import copy,unittest
from test_math_percent import source,validate
from _source_markdown import numeric_source_text
class UnicodePercent(unittest.TestCase):
 def test_literal_nonbreaking_percent_rendering(self):
  for gap in ['\u00a0','\u2009','\u202f']:
   with self.subTest(gap=repr(gap)):
    raw=source('50.4'+gap+'%');old=copy.deepcopy(raw)
    validate('Success 50.4%.',raw);self.assertEqual(raw,old)
 def test_wrong_values_signs_or_units_still_rejected(self):
  for claim,body in [('Success 50.5%.','50.4\u2009%'),('Return 50.4%.','-50.4\u2009%'),('Success 50.4%.','50.4')]:
   with self.subTest(claim=claim,body=body),self.assertRaises(ValueError):validate(claim,source(body))
 def test_no_join_across_cells_or_lines(self):
  for gap in ['\t','\n',' ']:
   with self.subTest(gap=repr(gap)),self.assertRaises(ValueError):validate('Success 50.4%.',source('50.4'+gap+'%'))
if __name__=='__main__':unittest.main()
