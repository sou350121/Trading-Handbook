import contextlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import fetch_article as fetcher
import run_pipeline as runner


class DailyBudget(unittest.TestCase):
    def item(self, pos):
        return {'pos': pos, 'msgid': str(pos), 'itemidx': '1', 'url': 'https://source.invalid', 'title': 'Paper', 'date': '2026-10-20'}

    def environment(self, root):
        raw = root / 'data/raw'
        raw.mkdir(parents=True)
        return patch.multiple(fetcher, ROOT=root, RAW=raw, _stats={'ok': 0, 'skip': 0, 'fail': 0}, _failures=[], _deferred=[])

    def test_real_alarm_bounds_a_stalled_fetch_and_keeps_unverified_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.environment(root), patch.object(fetcher, 'process_one', side_effect=lambda _: time.sleep(2)):
                started = time.monotonic()
                fetcher.bounded_fetch([self.item(1), self.item(2)], 0.04)
                self.assertLess(time.monotonic() - started, 0.7)
                self.assertEqual({r['reason'] for r in fetcher._deferred}, {'batch_budget_interrupted', 'batch_budget_deferred'})
                self.assertEqual({r['pos'] for r in fetcher._deferred}, {1, 2})

    def test_interrupted_item_rotates_behind_unattempted_item(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.environment(root):
                visited = []
                def stop(a):
                    visited.append(a['pos'])
                    raise fetcher.BatchBudgetExpired()
                with patch.object(fetcher, 'process_one', side_effect=stop):
                    fetcher.bounded_fetch([self.item(1), self.item(2)], 10)
                    fetcher._deferred.clear()
                    fetcher.bounded_fetch([self.item(1), self.item(2)], 10)
                self.assertEqual(visited, [2, 1])

    def test_real_failure_is_not_duplicated_as_deferral(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.environment(root):
                def work(a):
                    if a['pos'] == 2:
                        fetcher._failures.append({**a, 'reason': 'original_challenge'})
                    else:
                        raise fetcher.BatchBudgetExpired()
                with patch.object(fetcher, 'process_one', side_effect=work):
                    fetcher.bounded_fetch([self.item(1), self.item(2)], 10)
                self.assertEqual([r['pos'] for r in fetcher._deferred], [1])

    def test_cached_body_does_not_spend_fetch_budget_or_rotate_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.environment(root):
                (fetcher.RAW / '1_1.json').write_text(json.dumps({'body': 'x' * 300}))
                with patch.object(fetcher, 'process_one') as work:
                    fetcher.bounded_fetch([self.item(1)], 0.04)
                    work.assert_not_called()
                self.assertEqual(fetcher._stats['skip'], 1)
                self.assertFalse((root / 'data/health/quantml-attempts.json').exists())

    def test_previous_alarm_handler_is_restored(self):
        previous = signal.getsignal(signal.SIGALRM)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.environment(root), patch.object(fetcher, 'process_one', side_effect=fetcher.BatchBudgetExpired):
                fetcher.bounded_fetch([self.item(1)], 0.04)
        self.assertIs(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_budget_expiry_writes_failed_not_empty_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.environment(root), patch.object(fetcher, 'INDEX', root / 'index.json'), \
                 patch.object(fetcher, 'process_one', side_effect=fetcher.BatchBudgetExpired), \
                 patch.object(sys, 'argv', ['fetch_article.py', '--workers', '1', '--budget-seconds', '0.04']), \
                 contextlib.redirect_stdout(io.StringIO()):
                fetcher.INDEX.write_text(json.dumps([self.item(1)]))
                fetcher.main()
                receipt = json.loads((root / 'data/health/quantml-fetch.json').read_text())
                self.assertEqual(receipt['status'], 'failed')
                self.assertEqual(len(receipt['deferred']), 1)

    def test_invalid_budget_or_parallel_budget_fails_before_any_fetch(self):
        for args in [['--budget-seconds', '-1'], ['--budget-seconds', 'nan'], ['--budget-seconds', 'inf'],
                     ['--budget-seconds', '10', '--workers', '2']]:
            with patch.object(sys, 'argv', ['fetch_article.py', *args]), patch.object(fetcher, 'process_one') as work, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    fetcher.main()
                self.assertEqual(error.exception.code, 2)
                work.assert_not_called()

    def test_timeout_of_fetch_cannot_block_verified_output_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def execute(command, seconds=600):
                calls.append((command, seconds))
                if any(str(arg).endswith('fetch_article.py') for arg in command):
                    raise RuntimeError('step_timeout')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'execute', side_effect=execute), \
                 patch.object(runner, 'dirty_paths', return_value=[]), patch.object(runner, 'pending'):
                incomplete = runner.process('daily', publish=False, allow_verified_partial=True)
            self.assertIn('fetch_article.py', incomplete)
            fetch = next((cmd, seconds) for cmd, seconds in calls if any(str(v).endswith('fetch_article.py') for v in cmd))
            self.assertIn('--budget-seconds', fetch[0])
            self.assertEqual(fetch[1], 210)
            self.assertTrue(any('scripts/handbook_audit.py' in cmd for cmd, _ in calls))
            phases = [(cmd, seconds) for cmd, seconds in calls if any(str(v).endswith(name) for name in
                       ['enumerate_album.py', 'fetch_article.py', 'distill_pass_a.py', 'fix_collisions.py', 'distill_pass_b.py'] for v in cmd)]
            self.assertLess(sum(seconds for _, seconds in phases), 1200)


if __name__ == '__main__':
    unittest.main()
