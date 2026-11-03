import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / 'scripts/pulsar/run_pipeline.py'


def scope_namespace(paths, tracked):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {'allowed', 'new_dissection_paths', 'process'}]
    calls = []
    ns = {'ALLOWED': ('foundations/', 'guides/', 'radar/'), 'SINGLES': set(),
          'git': lambda *a: '\0'.join(tracked), 'dirty_paths': lambda: paths,
          'execute': lambda command, *a, **kw: calls.append(command),
          'pending': lambda: None, 'PYTHON': 'python'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns, calls


class ScopeTests(unittest.TestCase):
    def test_existing_articles_are_excluded(self):
        paths = ['foundations/a/old.md', 'foundations/a/new.md']
        ns, _ = scope_namespace(paths, paths[:1])
        self.assertEqual(ns['new_dissection_paths'](), paths[1:])

    def test_all_existing_means_no_formatter_call(self):
        ns, calls = scope_namespace(['foundations/a/old.md'], ['foundations/a/old.md'])
        ns['process']('weekly')
        self.assertFalse(any('scripts/add_fingerprint.py' in c or 'scripts/pulsar/fix_github_render.py' in c for c in calls))

    def test_actual_process_passes_only_new_paths(self):
        ns, calls = scope_namespace(['foundations/a/old.md', 'foundations/a/new.md', 'foundations/a/overview.md'], ['foundations/a/old.md'])
        ns['process']('weekly')
        fmt = [c for c in calls if 'scripts/pulsar/fix_github_render.py' in c]
        fp = [c for c in calls if 'scripts/add_fingerprint.py' in c]
        self.assertEqual(fmt, [['python', 'scripts/pulsar/fix_github_render.py', 'foundations/a/new.md']])
        self.assertEqual(fp, [['python', 'scripts/add_fingerprint.py', '--paths', 'foundations/a/new.md']])

    def test_no_paths_still_generates_summaries_and_index(self):
        ns, calls = scope_namespace([], [])
        ns['process']('weekly')
        self.assertIn(['python', 'scripts/gen_overviews.py'], calls)
        self.assertIn(['python', 'scripts/pulsar/local_index.py', '--check'], calls)

    def test_unmanaged_and_traversal_excluded(self):
        paths = ['secrets.md', 'foundations/../outside.md', 'foundations/a/new.md', 'foundations/a/overview.md', 'guides/new.md']
        ns, _ = scope_namespace(paths, [])
        self.assertEqual(ns['new_dissection_paths'](), ['foundations/a/new.md'])


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('fp', BASE/'scripts/add_fingerprint.py')
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m.ROOT = self.root
        self.folder = self.root/'foundations/a';self.folder.mkdir(parents=True)
        self.body = '<!-- ontology-5axis data=price -->\n# title\n\n> source\n\nbody\n'
        for name in ['old.md','new.md']:(self.folder/name).write_text(self.body, encoding='utf-8')

    def run_args(self, *args):
        with patch.object(sys, 'argv', ['add_fingerprint.py', *args]):self.m.main()

    def test_only_explicit_new_file_is_changed(self):
        self.run_args('--paths', 'foundations/a/new.md')
        self.assertEqual((self.folder/'old.md').read_text(encoding='utf-8'), self.body)
        self.assertIn('五軸座標', (self.folder/'new.md').read_text(encoding='utf-8'))

    def test_explicit_empty_does_not_fall_back_to_all(self):
        self.run_args('--paths')
        self.assertEqual((self.folder/'old.md').read_text(encoding='utf-8'), self.body)
        self.assertEqual((self.folder/'new.md').read_text(encoding='utf-8'), self.body)

    def test_dry_run_leaves_bytes_unchanged(self):
        self.run_args('--dry-run', '--paths', 'foundations/a/new.md')
        self.assertEqual((self.folder/'new.md').read_text(encoding='utf-8'), self.body)

    def test_outside_path_is_rejected(self):
        with self.assertRaises(ValueError):self.run_args('--paths', '../outside.md')

    def test_repeated_run_does_not_churn(self):
        self.run_args('--paths', 'foundations/a/new.md')
        first = (self.folder/'new.md').read_bytes()
        self.run_args('--paths', 'foundations/a/new.md')
        self.assertEqual((self.folder/'new.md').read_bytes(), first)


if __name__ == '__main__':unittest.main()
