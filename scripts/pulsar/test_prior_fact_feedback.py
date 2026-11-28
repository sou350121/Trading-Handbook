import json,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import distill_pass_b as w
class PriorFactFeedback(unittest.TestCase):
 def exercise(self,*,fingerprint='same-source',passed=False,draft_hash=None,issues=None,mutate=False):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);mid='arxiv_2611.12345';draft=b'Rejected public research draft';h=hashlib.sha256(draft).hexdigest();p=root/'data/rejected'/ (mid+'-attempt2.md');p.parent.mkdir(parents=True);p.write_bytes(draft)
   p=root/'data/health/pass-b-fact-reviews'/ (mid+'-'+h+'.json');p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(input_fingerprint=fingerprint,draft_sha256=draft_hash or h,review=dict(passed=passed,issues=['wrong experiment condition'] if issues is None else issues))))
   if mutate:(root/'data/rejected'/ (mid+'-attempt2.md')).write_text('changed')
   with patch.object(w,'ROOT',root):return w.previous_fact_feedback(mid,'same-source')
 def test_exact_source_and_draft_reuses_concrete_issues(self):
  t=self.exercise();self.assertIn('wrong experiment condition',t);self.assertIn('Rejected public research draft',t)
 def test_source_change_discards_prior_feedback(self):self.assertEqual(self.exercise(fingerprint='another-source'),'')
 def test_changed_draft_discards_prior_feedback(self):self.assertEqual(self.exercise(mutate=True),'')
 def test_success_is_not_a_rejection(self):self.assertEqual(self.exercise(passed=True),'')
 def test_mismatched_digest_discards_feedback(self):self.assertEqual(self.exercise(draft_hash='bad'),'')
 def test_malformed_or_empty_issues_are_ignored(self):
  for issues in [[],[{}],['x'*5001]]:self.assertEqual(self.exercise(issues=issues),'')
 def test_untrusted_identifier_cannot_read_other_files(self):self.assertEqual(w.previous_fact_feedback('../other','same-source'),'')
