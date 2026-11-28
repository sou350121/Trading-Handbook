import unittest
import arxiv_dissect as arxiv
import distill_pass_b as writer
from _content_gate import NUMBER,validate_page
from test_math_percent import page,source,validate
MATH='<math alttext="-"><semantics><mo>−</mo><annotation encoding="application/x-tex">-</annotation></semantics></math>'
class MathSign(unittest.TestCase):
 def test_actual_same_cell_sign_then_text(self):
  text=arxiv.html_to_text('<span class="ltx_td">'+MATH+'82.7</span><span class="ltx_td">10.2</span>')
  self.assertIn('-82.7',NUMBER.findall(text));self.assertIn('\t',text)
 def test_native_table_cell_boundary_not_joined(self):
  text=arxiv.html_to_text('<table><tr><td>'+MATH+'</td><td>82.7</td></tr></table>')
  self.assertNotIn('-82.7',NUMBER.findall(text))
 def test_span_table_cell_boundary_not_joined(self):
  text=arxiv.html_to_text('<span class="ltx_td">'+MATH+'</span><span class="ltx_td">82.7</span>')
  self.assertNotIn('-82.7',NUMBER.findall(text))
 def test_plain_list_hyphen_not_reinterpreted(self):
  text=arxiv.html_to_text('<ul><li>- 82.7</li></ul>')
  self.assertNotIn('-82.7',NUMBER.findall(text))
 def test_internal_math_spacing_normalizes_visible_sign(self):
  text=arxiv.html_to_text('<math alttext="- 0.08"><mo>−</mo><mn>0.08</mn></math>')
  self.assertEqual(NUMBER.findall(text),['-0.08'])
 def test_subtraction_expression_is_not_reinterpreted_as_negative_measurement(self):
  text=arxiv.html_to_text('<math alttext="1 - 82.7"><mn>1</mn><mo>−</mo><mn>82.7</mn></math>')
  self.assertNotIn('-82.7',NUMBER.findall(text));self.assertIn('1 - 82.7',text)
 def test_formula_fraction_and_annotation_not_duplicated(self):
  text=arxiv.html_to_text('<math alttext="\\frac{1}{2}"><semantics><mfrac><mn>1</mn><mn>2</mn></mfrac><annotation>12</annotation></semantics></math>')
  self.assertIn('\\frac{1}{2}',text);self.assertNotIn('12',NUMBER.findall(text))
 def test_image_alt_cannot_supply_numeric_evidence(self):
  text=arxiv.html_to_text('<img alt="-82.7" src="figure.png"><p>Uninspected figure</p>')
  self.assertNotIn('-82.7',NUMBER.findall(text))
 def test_original_numeric_gate_accepts_correct_sign_rejects_wrong(self):
  text=arxiv.html_to_text('<span class="ltx_td">'+MATH+'82.7</span>')
  validate('Max drawdown -82.7.',source(text))
  with self.assertRaises(ValueError):validate('Max drawdown 82.7.',source(text))
 def test_version_five_requires_fresh_source(self):
  raw=source('Result 82.7');raw['extraction_version']=5
  with self.assertRaisesRegex(ValueError,'source_extraction_requires_refresh'):validate('Result 82.7.',raw)
 def test_source_authored_gap_is_preserved(self):
  text=arxiv.html_to_text('<span class="ltx_td">'+MATH+' 82.7</span>')
  self.assertNotIn('-82.7',NUMBER.findall(text))
if __name__=='__main__':unittest.main()
