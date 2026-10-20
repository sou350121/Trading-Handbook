import contextlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import distill_pass_b as writer
import run_pipeline as runner
import _health as health
import arxiv_dissect as arxiv
from _content_gate import validate_page


class SourceContext(unittest.TestCase):
    def metadata(self, origin='arxiv'):
        return {'pos': 100000, 'msgid': 'arxiv_2610.23749', 'title': 'Source method',
                'zone': 'market-microstructure', 'url': 'https://arxiv.org/abs/2610.23749',
                'axes': {key: 'test' for key in ['data', 'horizon', 'paradigm', 'alpha', 'autonomy']},
                'source': {'origin': origin, 'framework': 'Source method'}}

    def test_primary_source_end_is_supplied_without_prefix_clipping(self):
        body = 'Source introduction ' * 1000 + '\nCAPACITY_ANALYSIS_AT_THE_END\n'
        prompt = writer.build(self.metadata(), {'body': body, 'source_kind': 'fulltext-html', 'extraction_version': 4})
        self.assertIn(body, prompt)
        self.assertIn('CAPACITY_ANALYSIS_AT_THE_END', prompt)
        self.assertNotIn('截斷', prompt)

    def test_primary_source_text_is_not_rewritten_by_template_labels(self):
        body = 'Source text 導讀 is an exact quote. ' * 100
        prompt = writer.build(self.metadata(), {'body': body, 'source_kind': 'fulltext-html', 'extraction_version': 4})
        self.assertIn(body, prompt)

    def test_quantml_complete_cache_is_not_confused_with_original_paper(self):
        body = 'Guide paragraph. ' * 1000 + 'GUIDE_END'
        prompt = writer.build(self.metadata('quantml'), {'body': body})
        self.assertIn(body, prompt)
        self.assertIn('僅讀導讀', prompt)
        self.assertIn('原論文作者限制未核', prompt)

    def test_abstract_cannot_use_primary_fulltext_label(self):
        with self.assertRaisesRegex(ValueError, 'fulltext_provenance_missing'):
            writer.build(self.metadata(), {'body': 'Abstract text. ' * 100, 'source_kind': 'abstract-only'})

    def test_legacy_merged_html_cache_cannot_reach_writer(self):
        with self.assertRaisesRegex(ValueError, 'source_extraction_requires_refresh'):
            writer.build(self.metadata(), {'body': 'Legacy fulltext ' * 100, 'source_kind': 'fulltext-html'})

    def test_oversize_source_fails_by_utf8_bytes_not_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, 'source_context_too_large'):
            writer.build(self.metadata(), {'body': '中' * 200001, 'source_kind': 'fulltext-html'})

    def test_oversize_source_does_not_spend_an_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.mkdir()
            (raw / 'arxiv_2610.23749_1.json').write_text(json.dumps({'body': 'x' * 600001, 'source_kind': 'fulltext-html'}))
            with patch.object(writer, 'ROOT', root), patch.object(writer, 'RAW', raw), \
                 patch.object(writer, '_n', {'ok': 0, 'skip': 0, 'fail': 0}), \
                 patch.object(writer, 'ask') as ask, contextlib.redirect_stderr(io.StringIO()):
                writer.do_one(self.metadata())
                ask.assert_not_called()
                self.assertEqual(writer._n['fail'], 1)


class CancellationRegression(unittest.TestCase):
    def test_source_cancellation_restores_seen_and_unpublished_page_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            pages = root / 'radar/test'
            pages.mkdir(parents=True)
            seen = root / 'data/test_seen.json'
            seen.write_text('["old"]')
            handlers = {}
            def register(number, callback): handlers[number] = callback
            def collect():
                seen.write_text('["old", "new"]')
                (pages / 'new.md').write_text('New unpublished page')
                handlers[signal.SIGTERM](None, None)
            with patch.object(health, 'ROOT', root), patch.object(health.signal, 'signal', side_effect=register), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = health.run('test', collect)
            self.assertEqual(code, 143)
            self.assertEqual(seen.read_text(), '["old"]')
            self.assertFalse((pages / 'new.md').exists())
            self.assertEqual(json.loads((root / 'data/health/test.json').read_text())['status'], 'failed')

    def test_actual_registered_sigterm_cannot_continue_partial_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            handlers = {}
            commands = []
            def register(number, callback): handlers[number] = callback
            def execute(command, *_):
                commands.append(command)
                if any(str(part).endswith('fetch_article.py') for part in command):
                    handlers[signal.SIGTERM](None, None)
            def git(*args):
                if args[0] == 'rev-parse': return 'verifiedhead\n'
                if args[0] == 'ls-remote': return 'verifiedhead\trefs/heads/main\n'
                return ''
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'code_digest', return_value='digest'), \
                 patch.object(runner.signal, 'signal', side_effect=register), \
                 patch.object(runner, 'execute', side_effect=execute), patch.object(runner, 'git', side_effect=git), \
                 patch.object(runner, 'dirty_paths', return_value=[]), patch.object(runner, 'pending'), \
                 patch.object(sys, 'argv', ['run_pipeline.py', '--job', 'daily', '--publish', '--allow-verified-partial']), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = runner.main()
            receipt = json.loads((root / 'data/health/pipeline-daily.json').read_text())
            self.assertEqual(code, 143)
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['reason'], 'runner_terminated')
            self.assertFalse(any(command[:2] in [['git', 'push'], ['git', 'commit']] for command in commands))


class ArxivSourceIntegrity(unittest.TestCase):
    def test_tex_percent_matches_visible_mathml_without_accepting_other_values(self):
        body = arxiv.html_to_text('<p>' + 'Source analysis ' * 50 + '</p><math alttext="17.6\\%"><semantics><mrow><mn>17.6</mn><mo>%</mo></mrow><annotation encoding="application/x-tex">17.6\\%</annotation></semantics></math>')
        page = ('<!-- ontology-5axis data=test -->\n# Study\nhttps://source.invalid/paper\n' +
                ''.join('\n## §' + str(i) + ' · Section\n' + 'Source-bound analysis ' * 18 for i in range(1, 9)) +
                '\nResult: 17.6%')
        raw = {'body': body, 'source_kind': 'fulltext-html', 'extraction_version': 4}
        metadata = {'url': 'https://source.invalid/paper', 'source': {'origin': 'arxiv'}}
        validate_page(page, raw, metadata)
        for unsupported in ['17.7%', '0.176']:
            with self.assertRaisesRegex(ValueError, 'unverified_empirical_number'):
                validate_page(page.replace('17.6%', unsupported), raw, metadata)

    def test_math_alttext_is_read_once_without_annotation_duplication(self):
        text = arxiv.html_to_text('<p>Bids <math alttext="1.0222"><semantics><mn>1.0222</mn><annotation encoding="application/x-tex">1.0222</annotation></semantics></math>; next <math alttext="-0.08"><mo>−</mo><mn>0.08</mn></math>.</p>')
        self.assertEqual(text.count('1.0222'), 1)
        self.assertIn(' -0.08 ', text)
        self.assertNotIn('1.02221.0222', text)

    def test_math_without_alttext_omits_duplicate_annotation(self):
        text = arxiv.html_to_text('<math><semantics><mn>0.267</mn><annotation encoding="application/x-tex">0.267</annotation></semantics></math>')
        self.assertEqual(text, '0.267')

    def test_adjacent_table_cells_preserve_real_measurement_tokens(self):
        body = arxiv.html_to_text('<p>' + 'Source analysis ' * 50 + '</p><table><tr><td>50%</td><td>111,102</td><td>388,737</td><td>56.5%</td><td>2.5%</td></tr></table>')
        self.assertNotIn('388,73756.5%', body)
        page = ('<!-- ontology-5axis data=test -->\n# Study\nhttps://source.invalid/paper\n' +
                ''.join('\n## §' + str(i) + ' · Section\n' + 'Source-bound analysis ' * 18 for i in range(1, 9)) +
                '\nResult: 56.5%')
        validate_page(page, {'body': body, 'source_kind': 'fulltext-html', 'extraction_version': 4},
                      {'url': 'https://source.invalid/paper', 'source': {'origin': 'arxiv'}})

    def cache_case(self, version=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'data/raw'
            raw.mkdir(parents=True)
            path = raw / 'arxiv_2610.23517_1.json'
            record = {'body': 'Legacy text ' * 100, 'source_kind': 'fulltext-html'}
            if version is not None:
                record['extraction_version'] = version
            path.write_text(json.dumps(record))
            before = path.read_bytes()
            with patch.object(arxiv, 'ROOT', root), patch.object(arxiv, 'RAW', raw), \
                 patch.object(arxiv, 'fetch_fulltext', return_value=('Refetched text ' * 100, 'fulltext-html')) as fetch:
                _, status = arxiv.ingest({'id': '2610.23517', 'title': 'Research', 'date': '2026-10-16', 'abs_url': 'https://arxiv.org/abs/2610.23517v1'}, 0)
                after = json.loads(path.read_text())
                backups = list((root / 'data/source-cache-backups').glob('*'))
                return status, after, fetch.call_count, [p.read_bytes() for p in backups], before

    def test_legacy_cache_is_backed_up_and_refetched(self):
        status, record, calls, backups, before = self.cache_case()
        self.assertEqual(status, 'fulltext-html')
        self.assertEqual(record['extraction_version'], 4)
        self.assertEqual(calls, 1)
        self.assertEqual(backups, [before])

    def test_current_cache_is_reused_without_network(self):
        status, _, calls, backups, _ = self.cache_case(4)
        self.assertEqual(status, 'cached')
        self.assertEqual(calls, 0)
        self.assertEqual(backups, [])

    def test_html_fetch_pins_explicit_paper_version(self):
        body = ('<p>arXiv:2610.23517v1</p><p>' + 'Paper text ' * 1000 + '</p>').encode()
        with patch.object(arxiv, '_fetch', side_effect=lambda url, **kwargs: (body, url)) as fetch:
            _, kind = arxiv.fetch_fulltext({'id': '2610.23517', 'abs_url': 'https://arxiv.org/abs/2610.23517v1'})
        self.assertEqual(kind, 'fulltext-html')
        fetch.assert_called_once_with('https://arxiv.org/html/2610.23517v1')

    def test_wrong_version_is_not_accepted_as_the_requested_paper(self):
        body = ('<p>arXiv:2610.23517v2</p><p>' + 'Paper text ' * 1000 + '</p>').encode()
        with patch.object(arxiv, '_fetch', side_effect=lambda url, **kwargs: (body, url)) as fetch:
            _, kind = arxiv.fetch_fulltext({'id': '2610.23517', 'abs_url': 'https://arxiv.org/abs/2610.23517v1'})
        self.assertEqual(kind, 'none')
        self.assertTrue(all(call.args[0].endswith('2610.23517v1') for call in fetch.call_args_list))


if __name__ == '__main__':
    unittest.main()
