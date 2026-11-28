import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import arxiv_dissect as a
class OAStatus(unittest.TestCase):
 def run_case(self,entries,classified=False):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);queue=root/'queue.json';queue.write_text(json.dumps(entries));dis=root/'distill';dis.mkdir()
   if classified:(dis/('oa_'+a.oa_key(entries[0]['doi'])+'.json')).write_text('{}')
   output=io.StringIO()
   with patch.multiple(a,OA_QUEUE=queue,DIS=dis,INGEST_ERRORS=[]),patch.object(a,'ingest_oa',side_effect=AssertionError('unexpected fetch')),contextlib.redirect_stdout(output):
    a.run_oa(6,1);errors=list(a.INGEST_ERRORS)
   return output.getvalue(),errors
 def test_exhausted_source_remains_failure_and_names_public_doi(self):
  text,errors=self.run_case([{'doi':'10.1111/mafi.70049','attempts':3}]);self.assertEqual(errors,['oa_retry_exhausted']);self.assertIn('OA_SOURCE_BLOCKED 10.1111/mafi.70049',text);self.assertIn('remain failed',text);self.assertNotIn('nothing to do',text)
 def test_genuinely_empty_queue_is_not_a_failure(self):
  text,errors=self.run_case([]);self.assertEqual(errors,[]);self.assertIn('nothing to do',text)
 def test_already_classified_source_does_not_repeat_exhausted_attempt(self):
  text,errors=self.run_case([{'doi':'10.1111/mafi.70049','attempts':3}],classified=True);self.assertEqual(errors,[]);self.assertIn('nothing to do',text)
