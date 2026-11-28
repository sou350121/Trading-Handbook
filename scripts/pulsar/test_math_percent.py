import copy,unittest
from _source_markdown import numeric_source_text
from _content_gate import validate_page
def page(value):
 return '<!-- ontology-5axis data=x -->\nhttps://arxiv.org/abs/2611.10801v1\n'+''.join('## §%s · Part\n%s\n'%(i,'來源與限制。'*80) for i in range(1,9))+value
def source(body):
 return {'url':'https://arxiv.org/abs/2611.10801v1','body':body+' public context '*100,'source_kind':'fulltext-html','extraction_version':6}
def validate(claim,raw):
 return validate_page(page(claim),raw,{'url':raw['url'],'source':{'origin':'arxiv'}})
class MathPercent(unittest.TestCase):
 def test_actual_thin_space_percent_presentation(self):
  raw=source(r'Turnover 34\text{\,}\mathrm{%}; power 80\text{\,}\mathrm{%}.')
  validate('Turnover 34%; power 80%.',raw)
 def test_wrong_value_stays_rejected(self):
  with self.assertRaises(ValueError):validate('Turnover 35%.',source(r'34\text{\,}\mathrm{%}'))
 def test_plain_number_cannot_authorize_percent(self):
  with self.assertRaises(ValueError):validate('Turnover 34%.',source('34'))
 def test_wrong_sign_stays_rejected(self):
  with self.assertRaises(ValueError):validate('Return 34%.',source(r'-34\text{\,}\mathrm{%}'))
 def test_only_exact_presentation_macros(self):
  with self.assertRaises(ValueError):validate('Turnover 34%.',source(r'34\text{not a thin space}\mathrm{%}'))
 def test_markdown_and_other_source_kinds_untouched(self):
  body=r'34\text{\,}\mathrm{%}'
  self.assertEqual(numeric_source_text({'body':body,'source_kind':'fulltext-pdf'}),body)
  self.assertEqual(numeric_source_text({'body':body,'source_kind':'fulltext-html','extraction_version':4}),body)
 def test_supported_escaped_percent_and_decimal(self):
  validate('Return -3.4%.',source(r'-3.4\,\mathrm{\%}'))
 def test_original_source_and_hash_input_untouched(self):
  raw=source(r'34\text{\,}\mathrm{%}');before=copy.deepcopy(raw)
  self.assertIn('34%',numeric_source_text(raw));self.assertEqual(raw,before)
if __name__=='__main__':unittest.main()
