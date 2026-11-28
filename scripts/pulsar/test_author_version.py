import json,unittest
import distill_pass_b as w
import _fact_review as f
import test_source_context as fixtures
class AuthorVersion(unittest.TestCase):
 def fixture(self):
  d=fixtures.SourceContext().metadata('oa');d.update(title='Author source',date='2025-10-27',url='https://arxiv.org/abs/2401.00103v2')
  raw=dict(title='Author source',body='Complete public paper text. '*100,source_kind='fulltext-pdf',source_variant='arxiv_author_version',source_version='2401.00103v2',source_version_date='2025-10-27',fulltext_url='https://arxiv.org/pdf/2401.00103v2',date='2025-10-27',bibliographic_doi='10.1111/mafi.70050',publisher_fulltext_obtained=False)
  return d,raw
 def test_complete_author_version_is_explicitly_distinguished_from_publisher(self):
  d,raw=self.fixture();prompt=w.build(d,raw);self.assertIn(raw['body'],prompt);self.assertIn('作者版本發布',prompt);self.assertIn('2401.00103v2',prompt);self.assertIn('出版社正式全文未取得',prompt);self.assertIn('不推定兩版相同',prompt)
 def test_wrong_version_url_or_date_is_not_accepted_as_author_provenance(self):
  for changes in [{'fulltext_url':'https://elsewhere.invalid/other.pdf'},{'source_version':'2401.00103'},{'date':'2026-08-25'}]:
   d,raw=self.fixture();raw.update(changes)
   with self.assertRaisesRegex(ValueError,'author_version_provenance_invalid'):w.build(d,raw)
 def test_independent_review_receives_complete_text_and_version_boundary(self):
  d,raw=self.fixture();seen=[]
  def caller(prompt,**kwargs):seen.append(json.loads(prompt));return '{"passed":true,"issues":[]}'
  self.assertEqual(f.review_page('Draft',raw,d,caller=caller),{'passed':True,'issues':[]});self.assertEqual(seen[0]['source_text'],raw['body']);self.assertFalse(seen[0]['source_provenance']['publisher_fulltext_obtained']);self.assertEqual(seen[0]['source_provenance']['source_version'],'2401.00103v2')
