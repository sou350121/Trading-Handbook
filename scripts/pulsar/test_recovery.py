"""Regression tests for false success, history retention and provider failures."""
import io
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import _album
import _content_gate as gate
import _deepseek
import _health
import arxiv_radar


TEST_TMP = Path(__file__).resolve().parents[2] / "data/test-tmp"
TEST_TMP.mkdir(parents=True, exist_ok=True)


class AlbumTests(unittest.TestCase):
    def item(self, n):
        return {'msgid': str(n), 'itemidx': '1', 'title': 'Research',
                'url': 'https://mp.weixin.qq.com/s/test', 'pos_num': n, 'create_time': 1792368000}

    def test_complete_pagination(self):
        responses = iter([{'article_list': [self.item(1)], 'continue_flag': 1},
                          {'article_list': [self.item(2)], 'continue_flag': 0}])
        rows = _album.enumerate_all(lambda *_: next(responses), lambda _: None)
        self.assertEqual([x['msgid'] for x in rows], ['1', '2'])

    def test_malformed_last_item_cannot_be_cursor_or_healthy(self):
        with self.assertRaisesRegex(ValueError, 'item_invalid'):
            _album.enumerate_all(lambda *_: {'article_list': [self.item(1), 'bad'], 'continue_flag': 1}, lambda _: None)

    def test_empty_continuation_fails(self):
        with self.assertRaises(ValueError):
            _album.enumerate_all(lambda *_: {'article_list': [], 'continue_flag': 1}, lambda _: None)

    def test_empty_album_fails(self):
        with self.assertRaises(ValueError):
            _album.enumerate_all(lambda *_: {'article_list': [], 'continue_flag': 0}, lambda _: None)

    def test_same_count_different_history_cannot_replace(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            path = Path(directory) / 'index.json'
            path.write_text(json.dumps([{'msgid': '1', 'itemidx': '1'}]))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'history_missing'):
                _album.save_complete([{'msgid': '2', 'itemidx': '1'}], path)
            self.assertEqual(before, path.read_bytes())

    def test_repeated_cursor_fails(self):
        with self.assertRaisesRegex(ValueError, 'pagination_stalled'):
            _album.enumerate_all(lambda *_: {'article_list': [self.item(1)], 'continue_flag': 1}, lambda _: None)

    def test_album_path_identity_is_rejected(self):
        item = self.item(1); item['msgid'] = '../../escape'
        with self.assertRaisesRegex(ValueError, 'identity_invalid'):
            _album.enumerate_all(lambda *_: {'article_list': [item], 'continue_flag': 0}, lambda _: None)



class ProviderTests(unittest.TestCase):
    def response(self, content='{"ok":true}', reason='stop'):
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': reason, 'message': {'content': content}}]}).encode())

    def test_deepseek_only_json_and_thinking_switch(self):
        with patch.object(_deepseek, '_key', return_value='SECRET_SENTINEL'), patch.object(_deepseek.urllib.request, 'urlopen', return_value=self.response()) as opened:
            self.assertEqual(json.loads(_deepseek.ask('JSON', thinking=False, json_mode=True))['ok'], True)
            request = opened.call_args.args[0]
            self.assertEqual(request.full_url, 'https://api.deepseek.com/chat/completions')
            self.assertEqual(json.loads(request.data)['thinking'], {'type': 'disabled'})

    def test_401_not_retried_or_logged(self):
        error = urllib.error.HTTPError('https://example.invalid', 401, 'SECRET_SENTINEL', {}, io.BytesIO(b'SECRET_SENTINEL'))
        with patch.object(_deepseek, '_key', return_value='SECRET_SENTINEL'), patch.object(_deepseek.urllib.request, 'urlopen', side_effect=error) as opened:
            with self.assertRaisesRegex(RuntimeError, '^deepseek_http_401$'):
                _deepseek.ask('JSON')
            self.assertEqual(opened.call_count, 1)

    def test_truncation_rejected(self):
        with patch.object(_deepseek, '_key', return_value='fake'), patch.object(_deepseek.urllib.request, 'urlopen', return_value=self.response(reason='length')):
            with self.assertRaisesRegex(RuntimeError, 'incomplete_output'):
                _deepseek.ask('JSON')

    def test_empty_rejected(self):
        with patch.object(_deepseek, '_key', return_value='fake'), patch.object(_deepseek.urllib.request, 'urlopen', return_value=self.response(content='')):
            with self.assertRaisesRegex(RuntimeError, 'empty_output'):
                _deepseek.ask('JSON')

    def test_invalid_json_rejected(self):
        with patch.object(_deepseek, '_key', return_value='fake'), patch.object(_deepseek.urllib.request, 'urlopen', return_value=self.response(content='not json')):
            with self.assertRaises(ValueError):
                _deepseek.ask('JSON', json_mode=True)

    def test_non_deepseek_model_rejected(self):
        with self.assertRaises(ValueError):
            _deepseek.ask('test', model='gpt-or-qwen')


class TransactionTests(unittest.TestCase):
    def test_failure_restores_content_and_seen_together(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            (root / 'radar').mkdir(); (root / 'data').mkdir()
            page = root / 'radar/week.md'; state = root / 'data/radar_seen.json'
            page.write_text('old verified page'); state.write_text('["old"]')
            def broken():
                page.write_text('empty failed page'); state.write_text('["new"]')
                (root / 'radar/new.md').write_text('failed')
                _health.fail('source', 'classification_failed')
            with patch.object(_health, 'ROOT', root):
                self.assertEqual(_health.run('test', broken), 1)
            self.assertEqual(page.read_text(), 'old verified page')
            self.assertEqual(state.read_text(), '["old"]')
            self.assertFalse((root / 'radar/new.md').exists())
            self.assertEqual(json.loads((root / 'data/health/test.json').read_text())['status'], 'failed')


class SourceTests(unittest.TestCase):
    def test_arxiv_bad_xml_fails(self):
        with self.assertRaises(ValueError):
            arxiv_radar._parse_entries('<broken')

    def test_arxiv_http_200_error_feed_fails(self):
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#bad_query</id></entry></feed>'
        with self.assertRaises(ValueError):
            arxiv_radar._parse_entries(xml)

    def test_arxiv_empty_without_result_metadata_is_not_healthy_zero(self):
        with self.assertRaisesRegex(ValueError, 'result_count_missing'):
            arxiv_radar._parse_entries('<feed xmlns="http://www.w3.org/2005/Atom" />')

    def test_arxiv_query_has_official_date_bounds(self):
        captured = []
        def response(url):
            captured.append(url)
            return '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/"><o:totalResults>0</o:totalResults></feed>'
        with patch.object(arxiv_radar, '_http_get', side_effect=response):
            arxiv_radar._fetch_search('cat:q-fin.PM', arxiv_radar.datetime(2026, 10, 6, tzinfo=arxiv_radar.timezone.utc))
        from urllib.parse import urlsplit, parse_qs
        query = parse_qs(urlsplit(captured[0]).query)['search_query'][0]
        self.assertIn('submittedDate:[202610060000 TO ', query)



class QualityTests(unittest.TestCase):
    def page(self, metric='12.5%'):
        return '<!-- ontology-5axis data=test -->\n# Research\nhttps://source.invalid/paper\n' + ''.join(f'\n## §{i} · Section\n' + 'source-bound analysis ' * 18 + '\n' for i in range(1, 9)) + '\nResult: ' + metric

    def test_supported_measurement_passes(self):
        gate.validate_page(self.page(), {'body': 'Evidence 12.5% ' * 30}, {'url': 'https://source.invalid/paper'})

    def test_unsupported_measurement_fails(self):
        with self.assertRaisesRegex(ValueError, 'empirical_number'):
            gate.validate_page(self.page('99.9%'), {'body': 'Evidence 12.5% ' * 30}, {'url': 'https://source.invalid/paper'})

    def test_verified_bibliographic_id_is_not_an_empirical_number(self):
        raw = {'body': 'Evidence 12.5% ' * 30}
        d = {'url': 'https://source.invalid/paper', 'bibliographic_verification': {'arxiv': '2610.21846'}}
        gate.validate_page(self.page() + '\nReference: arXiv 2610.21846', raw, d)
        with self.assertRaisesRegex(ValueError, 'empirical_number'):
            gate.validate_page(self.page() + '\nReference: arXiv 2610.99999', raw, d)

    def test_abstract_is_not_fulltext(self):
        with self.assertRaisesRegex(ValueError, 'provenance_missing'):
            gate.validate_page(self.page(), {'body': 'Evidence ' * 100, 'source_kind': 'abstract-only'}, {'source': {'origin': 'arxiv'}})

    def test_metadata_bad_axis_fails(self):
        with self.assertRaises(ValueError):
            gate.validate_metadata({'zone': 'test', 'axes': {'data': 'invented'}}, {'data': ['valid']}, {'test'})

    def test_ontology_must_match_frozen_metadata(self):
        axes = {key: 'valid' for key in ('data', 'horizon', 'paradigm', 'alpha', 'autonomy')}
        with self.assertRaisesRegex(ValueError, 'ontology_metadata_mismatch'):
            gate.validate_page(self.page(), {'body': 'Evidence 12.5% ' * 30}, {'url': 'https://source.invalid/paper', 'axes': axes})

    def test_missing_sections_fail(self):
        with self.assertRaises(ValueError):
            gate.validate_page('<!-- ontology-5axis -->' + 'x' * 2000, {'body': 'x' * 500}, {'url': 'https://example.invalid'})


class GithubLinkTests(unittest.TestCase):
    def test_root_link_gets_github_extension_and_anchor(self):
        import github_links
        root = github_links.ROOT
        path = root / 'radar/overview.md'
        before = '[Home](/README#topic)'
        after, count = github_links.rewrite(before, path)
        self.assertEqual(after, '[Home](../README.md#topic)')
        self.assertEqual(count, 1)

    def test_code_and_external_links_are_unchanged(self):
        import github_links
        text = '```md\n[Home](/README)\n```\n[External](https://example.invalid/x)'
        after, count = github_links.rewrite(text, github_links.ROOT / 'README.md')
        self.assertEqual(after, text)
        self.assertEqual(count, 0)


@unittest.skipUnless(os.name == 'posix', 'production runner uses Linux process groups and flock')
class RunnerTests(unittest.TestCase):
    def test_real_cron_entrypoints_launch_expected_commands(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory); scripts = root / 'scripts/pulsar'; scripts.mkdir(parents=True)
            stub = 'import sys,json;print(json.dumps(sys.argv[1:]))'
            (scripts / 'run_pipeline.py').write_text(stub)
            (scripts / 'watch_pipeline.py').write_text(stub)
            expectations = {
                'daily_update.sh': ['--job', 'daily', '--publish', '--allow-verified-partial'],
                'radar_weekly.sh': ['--job', 'weekly', '--publish', '--allow-verified-partial'],
                'daily_watchdog.sh': ['--apply', '--notify'],
            }
            for name, expected in expectations.items():
                raw = (Path(__file__).parent / name).read_bytes()
                wrapper = root / name
                wrapper.write_bytes(raw.replace(b'/home/claudeuser/trading-handbook-work', str(root).encode()))
                result = subprocess.run(['bash', str(wrapper)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, name + result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)

    def test_subprocess_failure_is_nonzero(self):
        import run_pipeline
        with patch.object(run_pipeline, 'ROOT', TEST_TMP):
            with self.assertRaisesRegex(RuntimeError, 'step_failed_7'):
                run_pipeline.execute([sys.executable, '-c', 'raise SystemExit(7)'], 5)

    def test_timeout_is_failure_and_child_is_reaped(self):
        import run_pipeline
        with patch.object(run_pipeline, 'ROOT', TEST_TMP):
            with self.assertRaisesRegex(RuntimeError, 'step_timeout'):
                run_pipeline.execute([sys.executable, '-c', 'import time;time.sleep(15)'], .1)

    def test_automation_scope_rejects_source_and_private_bodies(self):
        from run_pipeline import allowed
        self.assertFalse(allowed('scripts/pulsar/_qwen.py'))
        self.assertFalse(allowed('README.md'))
        self.assertFalse(allowed('data/raw/body.json'))
        self.assertTrue(allowed('radar/arxiv/2026-W43.md'))

    def test_slug_override_cannot_escape_foundations(self):
        import distill_pass_b
        with self.assertRaisesRegex(ValueError, 'slug_override_invalid'):
            distill_pass_b.slugify({'slug_override': '../../README'})

    def test_prepared_or_partial_receipt_is_never_healthy(self):
        from watch_pipeline import valid_receipt, code_digest
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc); expected = now - timedelta(hours=2)
        receipt = {'status': 'prepared', 'published': False, 'sources_only': False, 'code_sha256': code_digest(),
                   'started_at': (now - timedelta(hours=1)).isoformat(), 'finished_at': now.isoformat()}
        self.assertFalse(valid_receipt(receipt, expected, now))
        receipt.update(status='success', published=True, sources_only=True)
        self.assertFalse(valid_receipt(receipt, expected, now))
        receipt['sources_only'] = False
        self.assertTrue(valid_receipt(receipt, expected, now))


if __name__ == '__main__':
    unittest.main()
