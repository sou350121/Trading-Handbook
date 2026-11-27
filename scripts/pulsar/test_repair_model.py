import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import distill_pass_b as w

class RepairModelRoutingTests(unittest.TestCase):
 def exercise(self,configured,fail_first):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);d={'msgid':'test_1','pos':1,'zone':'methods','title':'Test'}
   meta=root/'metadata.json';raw=root/'source.json'
   meta.write_text(json.dumps(d));raw.write_text(json.dumps({'body':'public source '*40}))
   calls=[];validated=[]
   def ask(prompt,**options):
    calls.append(options.get('model'))
    return 'verified draft'
   def gate(*args):
    validated.append(1)
    if fail_first and len(validated)==1:raise ValueError('unverified_empirical_number:0.1')
   def review(*args,**options):
    if options.get('caller'):options['caller']('independent review')
    return {'passed':True,'issues':[]}
   env={'TRADING_DEEPSEEK_REPAIR_MODEL':configured}
   with patch.dict(w.os.environ,env),patch.object(w,'ROOT',root),patch.object(w,'ask',ask),patch.object(w,'build',lambda *a:'full source'),patch.object(w,'slugify',lambda d:'test'),patch.object(w,'validate_page',gate),patch.object(w,'review_page',review),patch.object(w,'save_review',lambda *a:None):
    w.do_one(d,(meta,meta.read_bytes()),raw)
   return calls,(root/'foundations/methods/test.md').is_file()
 def test_configured_repair_routes_generation_and_independent_review(self):
  calls,published=self.exercise('deepseek-v4-pro',True)
  self.assertEqual(calls,[None,'deepseek-v4-pro','deepseek-v4-pro'])
  self.assertTrue(published)
 def test_default_keeps_existing_route(self):
  calls,published=self.exercise('',True)
  self.assertEqual(calls,[None,None]);self.assertTrue(published)
 def test_good_first_draft_does_not_use_repair_model(self):
  calls,published=self.exercise('deepseek-v4-pro',False)
  self.assertEqual(calls,[None]);self.assertTrue(published)
 def test_unknown_repair_model_fails_before_external_call(self):
  calls,published=self.exercise('unsupported-model',False)
  self.assertEqual(calls,[]);self.assertFalse(published)
