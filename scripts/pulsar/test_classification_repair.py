import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import distill_pass_a as a
class ClassificationRepair(unittest.TestCase):
 def valid(self):
  return dict(zone=next(iter(a.ZONE_SLUGS-{'unsorted'})),axes={k:v[0] for k,v in a.AXES.items()},rating='🔧',tldr='A specific method',key_trick='A concrete mechanism',source={'venue':None,'arxiv':None,'framework':'Test','confidence':'high'},tags=['測試'],page_worthy=True)
 def run_case(self,replies,configured='deepseek-v4-pro',body=None,times=None):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'raw.json';out=root/'out';out.mkdir()
   src.write_text(json.dumps(dict(msgid='arxiv_2611.12345',pos=1,title='Public paper',url='https://arxiv.org/abs/2611.12345',chars=400,body=body if body is not None else 'source '*60)))
   with patch.dict(a.os.environ,{'TRADING_DEEPSEEK_REPAIR_MODEL':configured}),patch.object(a,'OUT',out),patch.object(a,'ask',side_effect=replies) as ask,patch.object(a.time,'monotonic',side_effect=times) if times else contextlib.nullcontext(),contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
    a.do_one(src)
   dest=out/'arxiv_2611.12345.json'
   return ask.call_args_list,json.loads(dest.read_text()) if dest.exists() else None
 def test_invalid_enum_is_retried_with_original_constraints(self):
  good=self.valid();bad=dict(good,zone='invented-zone');calls,out=self.run_case([json.dumps(bad),json.dumps(good)],times=[0,0,80])
  self.assertEqual(len(calls),2);self.assertIsNone(calls[0].kwargs['model']);self.assertEqual(calls[1].kwargs['model'],'deepseek-v4-pro');self.assertEqual(calls[1].kwargs['timeout'],100);self.assertEqual(out['source']['arxiv'],'2611.12345')
 def test_valid_first_response_uses_one_call(self):
  calls,out=self.run_case([json.dumps(self.valid())]);self.assertEqual(len(calls),1);self.assertTrue(out)
 def test_two_invalid_responses_leave_no_metadata(self):
  calls,out=self.run_case(['{}','{}']);self.assertEqual(len(calls),2);self.assertIsNone(out)
 def test_transport_failure_is_not_schema_retry(self):
  calls,out=self.run_case([RuntimeError('http_401')]);self.assertEqual(len(calls),1);self.assertIsNone(out)
 def test_thin_source_does_not_call_model(self):
  calls,out=self.run_case([],body='short');self.assertEqual(len(calls),0);self.assertIsNone(out)
 def test_unknown_provider_configuration_does_not_call_model(self):
  calls,out=self.run_case([],configured='another-provider');self.assertEqual(len(calls),0);self.assertIsNone(out)
 def test_shared_deadline_is_not_reset(self):
  calls,out=self.run_case(['{}'],times=[0,0,181]);self.assertEqual(len(calls),1);self.assertIsNone(out)
 def test_default_does_not_enable_extra_attempt(self):
  calls,out=self.run_case(['{}'],configured='');self.assertEqual(len(calls),1);self.assertIsNone(out)
 def test_repair_does_not_force_page_worthy(self):
  good=dict(self.valid(),page_worthy=False,rating='📖');calls,out=self.run_case(['{}',json.dumps(good)]);self.assertEqual(len(calls),2);self.assertFalse(out['page_worthy'])
