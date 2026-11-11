"""Offline acceptance of factual rejection, provenance and protected writes."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from _fact_review import parse_review, review_page


class ReviewContract(unittest.TestCase):
    def test_plain_and_single_json_fence(self):
        for response in ['{"passed":true,"issues":[]}',
                         '```json\n{"passed":true,"issues":[]}\n```']:
            self.assertEqual(parse_review(response), {'passed': True, 'issues': []})

    def test_invalid_or_inconsistent_verdict_never_passes(self):
        for value in ['{}', '[]', '{"passed":1,"issues":[]}',
                      '{"passed":false,"issues":[]}',
                      '{"passed":true,"issues":["wrong metric"]}',
                      '{"passed":false,"issues":[""]}',
                      '{"passed":true,"issues":[],"extra":true}',
                      '{"passed":false,"passed":true,"issues":[]}',
                      'Approved: {"passed":true,"issues":[]}',
                      '```json\n{"passed":true,"issues":[]}\n``` trailing']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_review(value)

    def test_rejection_preserves_specific_issue(self):
        self.assertEqual(parse_review('{"passed":false,"issues":["model A result attributed to B"]}')['issues'],
                         ['model A result attributed to B'])

    def test_full_source_tail_and_date_authority_are_included(self):
        raw = {'body': 'start' + 'x' * 9000 + 'UNIQUE_SOURCE_END',
               'date': '2026-10-06', 'date_source': 'official_album_create_time',
               'create_time': 1791284850, 'url': 'https://example.test/article',
               'source_kind': 'fulltext_quantml'}
        caller = Mock(return_value='{"passed":true,"issues":[]}')
        review_page('candidate', raw, {'axes': {'horizon': 'editorial'}}, caller=caller)
        payload = json.loads(caller.call_args.args[0])
        self.assertEqual(payload['source_text'], raw['body'])
        self.assertEqual(payload['source_provenance']['create_time'], raw['create_time'])
        self.assertEqual(payload['source_provenance']['date'], raw['date'])
        self.assertEqual(caller.call_args.kwargs['retries'], 1)
        self.assertEqual(caller.call_args.kwargs['timeout'], 300)
        self.assertEqual(caller.call_args.kwargs['max_tokens'], 65536)
        self.assertTrue(caller.call_args.kwargs['thinking'])

    def test_oversize_source_is_rejected_before_api(self):
        caller = Mock()
        with self.assertRaises(ValueError):
            review_page('candidate', {'body': 'x' * 600001}, {}, caller=caller)
        caller.assert_not_called()

    def test_api_failure_is_not_an_approval(self):
        with self.assertRaises(RuntimeError):
            review_page('candidate', {'body': 'x' * 400}, {},
                        caller=Mock(side_effect=RuntimeError('unavailable')))


class DraftPublicationBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.pb = importlib.import_module('distill_pass_b')
        except ModuleNotFoundError:
            # Local test-only placeholder; production imports remain unchanged.
            with patch.dict(sys.modules, {'_qwen': types.SimpleNamespace(ask=Mock())}):
                cls.pb = importlib.import_module('distill_pass_b')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rawdir = self.root / 'data/raw'; self.rawdir.mkdir(parents=True)
        self.disdir = self.root / 'data/distill'; self.disdir.mkdir()
        self.d = {'msgid': '123', 'pos': 1, 'zone': 'zone', 'title': 'Paper',
                  'date': '2026-10-06', 'url': 'https://example.test/article',
                  'source': {'framework': 'Framework'},
                  'axes': dict.fromkeys(['data', 'horizon', 'paradigm', 'alpha', 'autonomy'], 'x')}
        self.meta = self.disdir / '123.json'; self.meta.write_text(json.dumps(self.d))
        self.source = self.rawdir / '123_1.json'
        self.source.write_text(json.dumps({'body': 'x' * 400, 'source_kind': 'fulltext_quantml'}))
        for key, val in [('ROOT', self.root), ('RAW', self.rawdir), ('DIS', self.disdir),
                         ('_n', {'ok': 0, 'skip': 0, 'fail': 0})]:
            ctx = patch.object(self.pb, key, val); ctx.start(); self.addCleanup(ctx.stop)
        self.mechanical = patch.object(self.pb, 'validate_page'); self.mechanical.start(); self.addCleanup(self.mechanical.stop)
        self.writer = patch.object(self.pb, 'ask', return_value='# Unpublished draft'); self.ask = self.writer.start(); self.addCleanup(self.writer.stop)

    def execute(self, audit):
        with patch.object(self.pb, 'review_page', audit):
            self.pb.do_one(self.d, (self.meta, self.meta.read_bytes()), self.source)

    def test_fact_rejection_blocks_file_even_when_numeric_gate_passes(self):
        audit = Mock(return_value={'passed': False, 'issues': ['wrong cost regime']})
        self.execute(audit)
        self.assertFalse(self.pb.output_path(self.d).exists())
        self.assertEqual(self.pb._n['fail'], 1)
        self.assertEqual(audit.call_count, 2)
        self.assertIn('wrong cost regime', self.ask.call_args.args[0])

    def test_pass_writes_page_and_bound_private_evidence(self):
        self.execute(Mock(return_value={'passed': True, 'issues': []}))
        md = self.pb.output_path(self.d).read_text()
        receipts = list((self.root / 'data/health/pass-b-fact-reviews').glob('*.json'))
        self.assertEqual(len(receipts), 1)
        proof = json.loads(receipts[0].read_text())
        self.assertEqual(proof['draft_sha256'], hashlib.sha256(md.encode()).hexdigest())
        self.assertEqual(proof['review'], {'passed': True, 'issues': []})

    def test_source_change_during_review_never_writes(self):
        def audit(*_):
            self.source.write_text(json.dumps({'body': 'changed' * 100}))
            return {'passed': True, 'issues': []}
        self.execute(audit)
        self.assertFalse(self.pb.output_path(self.d).exists())
        self.assertEqual(self.pb._n['fail'], 1)

    def test_existing_article_is_preserved_without_model_calls(self):
        out = self.pb.output_path(self.d); out.parent.mkdir(parents=True)
        out.write_text('manual article')
        audit = Mock()
        self.execute(audit)
        self.assertEqual(out.read_text(), 'manual article')
        self.ask.assert_not_called(); audit.assert_not_called()

    def test_article_created_during_review_is_not_overwritten(self):
        def audit(*_):
            out = self.pb.output_path(self.d); out.parent.mkdir(parents=True)
            out.write_text('other writer')
            return {'passed': True, 'issues': []}
        self.execute(audit)
        self.assertEqual(self.pb.output_path(self.d).read_text(), 'other writer')

    def test_malformed_audit_cannot_publish(self):
        self.execute(Mock(side_effect=ValueError('fact_review_invalid_json')))
        self.assertFalse(self.pb.output_path(self.d).exists())


if __name__ == '__main__':
    unittest.main()
