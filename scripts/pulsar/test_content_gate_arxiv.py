import unittest
from _content_gate import validate_page

class ArxivCitationGateTests(unittest.TestCase):
    def setUp(self):
        self.raw = {'body': 'Evidence text. ' * 50 + ' observed value 0.25 and 5.0%.', 'source_kind': 'fulltext-pdf', 'url': 'https://arxiv.org/abs/2610.39636v1'}
        self.meta = {'url': self.raw['url'], 'source': {'origin': 'arxiv'}}

    def page(self, extra):
        return '<!-- ontology-5axis data=x -->\n# Example\n' + '\n'.join('## §' + str(i) + ' · Section\n' + 'Supported qualitative analysis. ' * 12 for i in range(1, 9)) + '\n' + self.meta['url'] + '\n' + extra

    def test_pinned_source_accepts_unversioned_citation(self):
        validate_page(self.page('arXiv:2610.39636.'), self.raw, self.meta)

    def test_exact_pinned_version(self):
        validate_page(self.page('arXiv [2610.39636v1]'), self.raw, self.meta)

    def test_markdown_id_label(self):
        validate_page(self.page('arXiv [2610.39636](https://arxiv.org/abs/2610.39636v1)'), self.raw, self.meta)

    def test_other_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number'):
            validate_page(self.page('arXiv:2610.39637'), self.raw, self.meta)

    def test_other_version_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number'):
            validate_page(self.page('arXiv:2610.39636v2'), self.raw, self.meta)

    def test_longer_decimal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number'):
            validate_page(self.page('arXiv:2610.39636.9'), self.raw, self.meta)

    def test_same_number_without_citation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number'):
            validate_page(self.page('Observed metric 2610.39636'), self.raw, self.meta)

    def test_measurement_next_to_valid_citation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number'):
            validate_page(self.page('arXiv:2610.39636; observed gain 88.88%'), self.raw, self.meta)

    def test_unsupported_signed_number_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unverified_empirical_number'):
            validate_page(self.page('Observed value −0.25'), self.raw, self.meta)

    def test_supported_measurements_unchanged(self):
        validate_page(self.page('Observed value 0.25 and 5.0%.'), self.raw, self.meta)

if __name__ == '__main__':
    unittest.main()
