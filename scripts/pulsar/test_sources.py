import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import _health
import _nber
import fetch_article as article
import practitioner_radar as radar


class NberVerification(unittest.TestCase):
    ref = b'paper\tauthor\ttitle\tissue_date\tdoi\nw35898\tA Author\t"Quoted" title\t2026-10-15\tNULL\n'
    dates = b'paper\tissue_date\nw35898\t2026-10-15\n'
    rss = b'<rss><channel><item><title>"Quoted" title -- by A Author</title><link>https://www.nber.org/papers/w35898#fromrss</link></item></channel></rss>'

    def run_case(self, ref=None, dates=None, rss=None):
        return _nber.reconcile(ref or self.ref, dates or self.dates, rss or self.rss,
                               14, dt.date(2026, 10, 20))

    def test_real_tsv_literal_quotes_do_not_consume_later_records(self):
        ref = self.ref + b'w35899\tB Author\tSecond title\t2026-10-08\tNULL\n'
        dates = self.dates + b'w35899\t2026-10-08\n'
        rows, proof = self.run_case(ref=ref, dates=dates)
        self.assertEqual([row['paper'] for row in rows], ['w35899', 'w35898'])
        self.assertEqual(proof['rss_identities_verified'], 1)

    def test_missing_date_in_rss_uses_two_agreeing_official_tables(self):
        rows, _ = self.run_case()
        self.assertEqual(rows[0]['issue_date'], '2026-10-15')

    def test_wrong_title_is_not_matched_by_id_alone(self):
        with self.assertRaisesRegex(ValueError, 'title_mismatch'):
            self.run_case(rss=self.rss.replace(b'Quoted', b'Other'))

    def test_wrong_author_is_not_silently_stripped(self):
        with self.assertRaisesRegex(ValueError, 'author_mismatch'):
            self.run_case(rss=self.rss.replace(b'A Author', b'B Author'))

    def test_disagreeing_dates_fail(self):
        with self.assertRaisesRegex(ValueError, 'date_mismatch'):
            self.run_case(dates=self.dates.replace(b'10-15', b'10-16'))

    def test_unknown_date_is_not_retrieval_date(self):
        with self.assertRaisesRegex(ValueError, 'date_invalid'):
            self.run_case(ref=self.ref.replace(b'2026-10-15', b'0000-00-00'),
                          dates=self.dates.replace(b'2026-10-15', b'0000-00-00'))

    def test_future_date_fails(self):
        with self.assertRaisesRegex(ValueError, 'date_future'):
            self.run_case(ref=self.ref.replace(b'10-15', b'10-21'),
                          dates=self.dates.replace(b'10-15', b'10-21'))

    def test_missing_current_rss_paper_fails_snapshot(self):
        with self.assertRaisesRegex(ValueError, 'missing_current_rss'):
            self.run_case(rss=self.rss.replace(b'w35898', b'w35900'))

    def test_foreign_host_is_not_an_official_identity(self):
        with self.assertRaisesRegex(ValueError, 'identity_invalid'):
            self.run_case(rss=self.rss.replace(b'www.nber.org', b'example.org'))

    def test_bad_tsv_width_or_duplicate_fails(self):
        for ref in [self.ref.replace(b'\tNULL', b''), self.ref + self.ref.split(b'\n')[1] + b'\n']:
            with self.assertRaisesRegex(ValueError, 'record_invalid_or_duplicate'):
                self.run_case(ref=ref)

    def test_html_or_empty_feed_cannot_prove_snapshot_currency(self):
        for body in [b'<html/>', b'<rss><channel/></rss>']:
            with self.assertRaises(ValueError):
                self.run_case(rss=body)


class SourceIsolation(unittest.TestCase):
    def execute(self, partial=True, broken_item=False, cap=30, throw=False, all_failed=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            out = root / 'radar/practitioner'
            out.mkdir(parents=True)
            feeds = [{'id': x, 'name': x, 'url': 'https://example.org/' + x} for x in ['good', 'bad']]
            config = data / 'feeds.json'
            config.write_text(json.dumps(feeds))
            seen = data / 'radar_prac_seen.json'
            state = data / 'radar_prac_state.json'
            old = {'feed_id': 'bad', 'feed_name': 'bad', 'title': 'Old verified title',
                   'link': 'https://example.org/old', 'date': '2026-10-08',
                   'zone': 'meta', 'one_liner_zh': 'old', 'must_read': False}
            seen.write_text(json.dumps([radar.link_hash(old['link'])]))
            state.write_text(json.dumps({'2026-W43': [old]}))
            before = (seen.read_bytes(), state.read_bytes())
            def fetch(feed, _days):
                if all_failed:
                    return [], 'invalid-feed'
                if feed['id'] == 'bad' and not broken_item:
                    return [], 'invalid-feed'
                count = 2 if feed['id'] == 'bad' else 1
                return [{'feed_id': feed['id'], 'feed_name': feed['name'],
                         'title': feed['id'] + str(i), 'date': '2026-10-15',
                         'link': 'https://example.org/' + feed['id'] + str(i)} for i in range(count)], 'ok'
            def classify(item):
                if item['title'] == 'bad1':
                    raise ValueError('classification_invalid')
                return {'relevant': True, 'zone': 'meta', 'one_liner_zh': 'verified', 'must_read': False}
            argv = ['practitioner_radar.py', '--week', '2026-W43', '--cap', str(cap)]
            if partial:
                argv += ['--allow-verified-partial']
            with contextlib.ExitStack() as stack:
                for obj, name, value in [(_health, 'ROOT', root), (radar, 'ROOT', root),
                    (radar, 'DATA', data), (radar, 'OUT_DIR', out), (radar, 'FEEDS_FILE', config),
                    (radar, 'SEEN_FILE', seen), (radar, 'STATE_FILE', state),
                    (radar, 'fetch_feed', fetch), (radar, 'classify', classify), (sys, 'argv', argv)]:
                    stack.enter_context(patch.object(obj, name, value))
                if throw:
                    stack.enter_context(patch.object(radar, 'render_overview', side_effect=RuntimeError('late failure')))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                code = _health.run('practitioner', radar.main)
            page = (out / '2026-W43.md').read_text() if (out / '2026-W43.md').exists() else ''
            return code, json.loads(seen.read_text()), json.loads(state.read_text()), page, before, (seen.read_bytes(), state.read_bytes())

    def test_verified_feed_persists_but_partial_is_nonzero_and_labeled(self):
        code, seen, state, page, _, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertIn(radar.link_hash('https://example.org/good0'), seen)
        self.assertEqual(len(state['2026-W43']), 2)
        self.assertIn('來源覆蓋未完成', page)
        self.assertIn('Old verified title', page)

    def test_one_failed_classification_rolls_back_the_whole_feed(self):
        code, seen, state, _, _, _ = self.execute(broken_item=True)
        self.assertEqual(code, 2)
        self.assertNotIn(radar.link_hash('https://example.org/bad0'), seen)
        self.assertNotIn('bad0', [x['title'] for x in state['2026-W43']])
        self.assertIn(radar.link_hash('https://example.org/good0'), seen)

    def test_cap_rolls_back_the_affected_feed(self):
        code, seen, _, _, _, _ = self.execute(broken_item=True, cap=2)
        self.assertEqual(code, 2)
        self.assertNotIn(radar.link_hash('https://example.org/bad0'), seen)

    def test_default_rolls_back_everything_without_partial_permission(self):
        code, _, _, page, before, after = self.execute(partial=False)
        self.assertEqual(code, 1)
        self.assertEqual(before, after)
        self.assertEqual(page, '')

    def test_late_exception_cannot_keep_uncommitted_partial_outputs(self):
        code, _, _, page, before, after = self.execute(throw=True)
        self.assertEqual(code, 1)
        self.assertEqual(before, after)
        self.assertEqual(page, '')

    def test_no_complete_source_cannot_create_a_partial_empty_page(self):
        code, _, _, page, before, after = self.execute(all_failed=True)
        self.assertEqual(code, 1)
        self.assertEqual(before, after)
        self.assertEqual(page, '')


class ArticleFetchVerification(unittest.TestCase):
    def test_publisher_day_is_not_utc_day(self):
        timestamp = int(dt.datetime(2026, 10, 14, 23, tzinfo=dt.timezone.utc).timestamp())
        _, day, _ = article.extract('var ct = "' + str(timestamp) + '";')
        self.assertEqual(day, '2026-10-15')

    def test_http_200_challenge_is_never_article_success(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): return b'<html>wappoc_appmsgcaptcha token=PRIVATE</html>'
            def geturl(self): return 'https://mp.weixin.qq.com/mp/wappoc_appmsgcaptcha?token=PRIVATE'
        with patch.object(article.urllib.request, 'urlopen', return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, 'original_challenge') as raised:
                article.fetch('https://mp.weixin.qq.com/s?mid=1', retries=0)
            self.assertNotIn('PRIVATE', str(raised.exception))


class FeedVerification(unittest.TestCase):
    def fetch(self, content, mime, effective='https://www.validea.com/', feed_id='validea'):
        class Response:
            headers = {'Content-Type': mime}
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): return content
            def geturl(self): return effective
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(radar, 'DATA', root), patch.object(radar.urllib.request, 'urlopen', return_value=Response()):
                _health.ERRORS.clear()
                result = radar.fetch_feed({'id': feed_id, 'name': 'Validea', 'url': 'https://blog.validea.com/feed/'}, 14)
                proof = root / 'health' / ('feed-' + feed_id + '.json')
                return result, json.loads(proof.read_text()) if proof.exists() else {}, list(_health.ERRORS)

    def test_validea_homepage_redirect_is_not_a_successful_empty_feed(self):
        (items, status), proof, _ = self.fetch(b'<html>Home</html>', 'text/html')
        self.assertEqual(items, [])
        self.assertEqual(status, 'redirected-to-html')
        self.assertEqual(proof['effective_host'], 'www.validea.com')

    def test_missing_date_is_explicit_failure(self):
        body = b'<rss><channel><item><title>Valid title</title><link>https://example.org/a</link></item></channel></rss>'
        (items, _), _, errors = self.fetch(body, 'application/rss+xml')
        self.assertEqual(items, [])
        self.assertIn({'component': 'validea', 'reason': 'publication_date_missing'}, errors)

    def test_missing_title_cannot_be_silently_discarded(self):
        body = b'<rss><channel><item><link>https://example.org/a</link></item></channel></rss>'
        _, _, errors = self.fetch(body, 'application/rss+xml')
        self.assertIn({'component': 'validea', 'reason': 'article_title_missing'}, errors)

    def test_official_supplement_requires_matching_publisher_channel(self):
        (items, status), _, _ = self.fetch(b'<rss><channel><title>Other show</title></channel></rss>',
                                          'application/rss+xml', feed_id='validea-excess-returns')
        self.assertEqual(items, [])
        self.assertEqual(status, 'publisher-identity-mismatch')

    def test_official_supplement_keeps_explicit_dates_and_separate_identity(self):
        day = dt.date.today() - dt.timedelta(days=1)
        body = ('<rss><channel><title>Excess Returns </title><item><title>Research interview</title><link>https://creators.spotify.com/pod/show/excess-returns/a</link><pubDate>' + day.isoformat() + '</pubDate></item></channel></rss>').encode()
        (items, status), _, errors = self.fetch(body, 'application/rss+xml', feed_id='validea-excess-returns')
        self.assertEqual(status, 'ok')
        self.assertFalse(errors)
        self.assertEqual(items[0]['date'], day.isoformat())
        self.assertEqual(items[0]['feed_id'], 'validea-excess-returns')


if __name__ == '__main__':
    unittest.main()
