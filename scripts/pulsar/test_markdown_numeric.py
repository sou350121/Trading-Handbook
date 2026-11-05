import copy
import unittest
from _content_gate import validate_page
from _source_markdown import numeric_source_text


class MarkdownNumericTests(unittest.TestCase):
    def raw(self, evidence, source_format='markdown'):
        return {'body': 'Evidence text. ' * 50 + '\n' + evidence,
                'source_format': source_format, 'url': 'https://example.invalid/source'}

    def page(self, claim):
        return '<!-- ontology-5axis data=x -->\n# Example\n' + '\n'.join(
            '## §' + str(i) + ' · Section\n' + 'Supported qualitative analysis. ' * 12
            for i in range(1, 9)) + '\nhttps://example.invalid/source\n' + claim

    def check(self, evidence, claim, passes=True, source_format='markdown'):
        raw = self.raw(evidence, source_format)
        before = copy.deepcopy(raw)
        if passes:
            validate_page(self.page(claim), raw, {'url': raw['url']})
        else:
            with self.assertRaises(ValueError):
                validate_page(self.page(claim), raw, {'url': raw['url']})
        self.assertEqual(raw, before)

    def test_escaped_decimal(self):
        self.check(r'Return 17\.12 bps', 'Return 17.12 bps')

    def test_escaped_percentage(self):
        self.check(r'RankIC 11\.22\%', 'RankIC 11.22%')

    def test_escaped_negative(self):
        self.check(r'Value \-0\.2795', 'Value −0.2795')

    def test_wrong_sign_still_rejected(self):
        self.check(r'Value 17\.12\%', 'Value −17.12%', False)

    def test_changed_value_still_rejected(self):
        self.check(r'Value 17\.12', 'Value 17.13', False)

    def test_changed_precision_still_rejected(self):
        self.check(r'Value 17\.12', 'Value 17.120', False)

    def test_unit_cannot_be_added(self):
        self.check(r'Value 17\.12', 'Value 17.12%', False)

    def test_plaintext_not_unescaped(self):
        self.check(r'Value 17\.12', 'Value 17.12', False, 'text')

    def test_double_backslash_not_recursively_decoded(self):
        self.check(r'Value 17\\.12', 'Value 17.12', False)

    def test_inline_code_keeps_literal_backslash(self):
        self.check(r'`17\.12`', 'Value 17.12', False)

    def test_fenced_code_keeps_literal_backslash(self):
        self.check('```\n17\\.12\n```', 'Value 17.12', False)

    def test_indented_code_keeps_literal_backslash(self):
        self.check('\n    17\\.12', 'Value 17.12', False)

    def test_indentation_does_not_turn_paragraph_continuation_into_code(self):
        self.check('    17\\.12', 'Value 17.12')

    def test_image_alt_not_numeric_evidence(self):
        self.check('![17.12](https://example.invalid/image.png)', 'Value 17.12', False)

    def test_link_destination_not_numeric_evidence(self):
        self.check('[report](https://example.invalid/17.12)', 'Value 17.12', False)

    def test_html_hidden_numbers_not_authorized(self):
        self.check('<span hidden>17.12</span>', 'Value 17.12', False)

    def test_nodes_do_not_join_into_fabricated_decimal(self):
        self.check('17**.12**', 'Value 17.12', False)

    def test_visible_entities(self):
        self.check('Value 17&#46;12', 'Value 17.12')

    def test_escape_does_not_create_markdown_structure(self):
        source = numeric_source_text(self.raw(r'\`17\.12\`'))
        self.assertIn('17.12', source)

    def test_percentage_point_delta(self):
        self.check(r'Baseline 9\.84\%; Method 11\.22\%', '| market | RankIC | 9.84% | 11.22% | +1.38pp |')

    def test_wrong_percentage_point_delta(self):
        self.check(r'Baseline 9\.84\%; Method 11\.22\%', '| market | RankIC | 9.84% | 11.22% | +1.39pp |', False)

    def test_pp_not_allowed_outside_calculation_column(self):
        self.check(r'Baseline 9\.84\%; Method 11\.22\%', 'Gain +1.38pp', False)

    def test_pp_not_allowed_for_unmatched_operand_units(self):
        self.check(r'Baseline 9\.84; Method 11\.22\%', '| market | RankIC | 9.84 | 11.22% | +1.38pp |', False)

if __name__ == '__main__':
    unittest.main()
