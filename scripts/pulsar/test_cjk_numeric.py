import unittest
from _content_gate import NUMBER,validate_page
def page(claim):
 return '<!-- ontology-5axis data=x -->\nhttps://example.org/paper\n'+''.join('## §%s · Part\n%s\n'%(i,'可核對來源的研究與限制。'*35) for i in range(1,9))+claim
def validate(claim,source):
 return validate_page(page(claim),{'url':'https://example.org/paper','body':source+' 來源背景。'*100},{'url':'https://example.org/paper','source':{'origin':'quantml'}})
class CjkNumeric(unittest.TestCase):
 def test_attached_source_measurement(self):validate('夏普 1.62。','夏普為1.62，樣本有限。')
 def test_attached_claim_is_checked(self):
  with self.assertRaisesRegex(ValueError,'unverified_empirical_number:1.63'):validate('夏普為1.63。','夏普為1.62。')
 def test_chinese_percentage_source(self):validate('回撤為1.62%。','回撤為1.62%，不是改善量。')
 def test_chinese_percentage_wrong_unit_rejected(self):
  with self.assertRaises(ValueError):validate('回撤為1.62%。','原文統計量為1.62。')
 def test_chinese_negative_sign_preserved(self):
  self.assertIn('−1.62%',NUMBER.findall('回撤為−1.62%。'))
  with self.assertRaises(ValueError):validate('回報為1.62%。','回報為−1.62%。')
 def test_ascii_identifier_boundary_preserved(self):
  for text in ['abc1.62','model_1.62','b123.45']:
   self.assertNotIn('1.62',NUMBER.findall(text))
  with self.assertRaises(ValueError):validate('夏普 1.62。','模型abc1.62。')
 def test_existing_english_and_math_boundaries(self):
  self.assertEqual(NUMBER.findall('value -1.62% ; 1.62 ; x=0.5'),['-1.62%','1.62','0.5'])
 def test_han_context_variants(self):
  for prefix in ['為','为','率','值','達','达','中','最']:
   with self.subTest(prefix=prefix):
    self.assertIn('1.62%',NUMBER.findall(prefix+'1.62%'))
    with self.assertRaises(ValueError):validate(prefix+'9.876%。','率為1.62%。')
if __name__=='__main__':unittest.main()
