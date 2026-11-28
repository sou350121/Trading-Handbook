import datetime as dt,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_pipeline as r
class WeeklyBatch(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);(self.root/'data/distill').mkdir(parents=True);self.now=dt.datetime(2026,11,28,15,tzinfo=dt.timezone(dt.timedelta(hours=8)));self.ids=[f'2611.{10000+i}' for i in range(9)]
  self.save_queue(self.ids)
  self.p=patch.object(r,'ROOT',self.root);self.p.start();self.addCleanup(self.p.stop)
 def save_queue(self,ids):
  (self.root/'data/radar_queue.json').write_text(json.dumps([{'id':x,'date':f'2026-11-{28-i:02d}'} for i,x in enumerate(ids)]))
 def classify(self,ids):
  for x in ids:(self.root/'data/distill'/('arxiv_'+x+'.json')).write_text('{}')
 def manifest(self):return self.root/'data/health/weekly-arxiv-batch-2026-W48.json'
 def test_first_run_reserves_six_in_original_newest_order(self):self.assertEqual(r.weekly_arxiv_batch(self.now),self.ids[:6])
 def test_retry_after_classification_does_not_admit_next_six(self):
  r.weekly_arxiv_batch(self.now);self.classify(self.ids[:6]);self.assertEqual(r.weekly_arxiv_batch(self.now),[])
 def test_retry_keeps_original_unclassified_ids(self):
  r.weekly_arxiv_batch(self.now);self.classify(self.ids[:4]);self.save_queue(list(reversed(self.ids)));self.assertEqual(r.weekly_arxiv_batch(self.now),self.ids[4:6])
 def test_new_week_starts_new_batch(self):
  r.weekly_arxiv_batch(self.now);self.classify(self.ids[:6]);self.assertEqual(r.weekly_arxiv_batch(self.now+dt.timedelta(days=7)),self.ids[6:])
 def test_late_arrivals_can_fill_unspent_slots_only(self):
  self.save_queue(self.ids[:2]);r.weekly_arxiv_batch(self.now);self.classify(self.ids[:2]);self.save_queue(self.ids);self.assertEqual(r.weekly_arxiv_batch(self.now),self.ids[2:6])
 def test_duplicate_queue_items_do_not_consume_or_expand_quota(self):
  self.save_queue([self.ids[0]]*4+self.ids[1:]);self.assertEqual(r.weekly_arxiv_batch(self.now),self.ids[:6])
 def test_bad_manifest_never_silently_resets_quota(self):
  r.weekly_arxiv_batch(self.now)
  for state in [{'version':1,'week':'bad','ids':[]},{'version':1,'week':'2026-W48','ids':['../x']},{'version':1,'week':'2026-W48','ids':self.ids[:7]},{'version':1,'week':'2026-W48','ids':[self.ids[0]]*2}]:
   self.manifest().write_text(json.dumps(state))
   with self.assertRaisesRegex(RuntimeError,'weekly_arxiv_batch_invalid'):r.weekly_arxiv_batch(self.now)
 def test_invalid_queue_identity_does_not_become_command_argument(self):
  self.save_queue(['bad,--option'])
  with self.assertRaisesRegex(RuntimeError,'weekly_arxiv_queue_invalid'):r.weekly_arxiv_batch(self.now)
 def test_persisted_ids_do_not_need_to_remain_in_new_feed(self):
  r.weekly_arxiv_batch(self.now);self.save_queue([]);self.assertEqual(r.weekly_arxiv_batch(self.now),self.ids[:6])
 def test_no_classifier_or_model_runs_in_reservation(self):
  with patch.object(r,'execute',side_effect=AssertionError('unexpected subprocess')):self.assertEqual(r.weekly_arxiv_batch(self.now),self.ids[:6])

 def test_weekly_process_uses_reserved_ids_instead_of_implicit_new_intake(self):
  from test_postprocess_scope import scope_namespace
  ns,calls=scope_namespace([],[]);ns['weekly_arxiv_batch']=lambda:['2611.10001','2611.10002'];ns['process']('weekly')
  arxiv=[c for c in calls if 'scripts/pulsar/arxiv_dissect.py' in c and '--oa' not in c and '--refresh-stale' not in c]
  self.assertEqual(len(arxiv),1);self.assertIn('--ids',arxiv[0]);self.assertIn('2611.10001,2611.10002',arxiv[0])
 def test_completed_batch_keeps_backlog_gate_and_source_scans(self):
  from test_postprocess_scope import scope_namespace
  ns,calls=scope_namespace([],[]);ns['weekly_arxiv_batch']=lambda:[];hits=[];ns['pending']=lambda:hits.append(True);ns['process']('weekly')
  self.assertFalse(any('scripts/pulsar/arxiv_dissect.py' in c and '--oa' not in c and '--refresh-stale' not in c for c in calls))
  self.assertTrue(any('scripts/pulsar/arxiv_radar.py' in c for c in calls));self.assertTrue(any('scripts/pulsar/distill_pass_b.py' in c for c in calls));self.assertTrue(hits)
