import unittest
from _content_gate import validate_page


class SectionReferences(unittest.TestCase):
    def page(self, extra):
        return ('<!-- ontology-5axis data=test -->\n# Research\nhttps://source.invalid/paper\n' +
                ''.join('\n## §' + str(i) + ' · Section\n' + 'Source-bound analysis. ' * 18
                        for i in range(1, 9)) + '\n' + extra)

    def validate(self, extra):
        validate_page(self.page(extra), {'body': 'Evidence 12.5% ' * 30},
                      {'url': 'https://source.invalid/paper'})

    def test_real_rejected_cross_reference_is_navigation(self):
        self.validate('然後照 §2.5 手推一次。')
        self.validate('參見§2.5 的演算步驟。')
        self.validate('See §2.5 for the worked example.')

    def test_unsupported_measurement_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number:2.5%'):
            self.validate('參見 §2.5。Sharpe improved by 2.5%.')

    def test_metric_with_section_symbol_is_not_an_exemption(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number:2.5'):
            self.validate('Sharpe = §2.5')

    def test_percent_after_section_reference_still_fails(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number:2.5%'):
            self.validate('見 §2.5%')

    def test_template_bold_subheading_is_structural(self):
        self.validate('**7.1 可證偽預測（帶時點）**')

    def test_measurement_after_bold_subheading_is_still_checked(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number:2.5%'):
            self.validate('**7.1 可證偽預測：Sharpe = 2.5%**')


if __name__ == '__main__':
    unittest.main()
