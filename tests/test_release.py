"""Release checks for the PDF-only report distribution."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import package_release, verify_release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = Path(__file__).resolve().parents[1]
        for directory in ('docs', 'experiments'):
            shutil.copytree(source / directory, self.root / directory)
        manifest = json.loads((self.root / 'docs/results/full_cuda/experiment.json').read_text())
        for name in [*manifest['source_sha256'], 'README.md', 'requirements-lock-windows.txt']:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        # Only the PDF signature is checked; this fixture is not a rendered report.
        (self.root / 'report.pdf').write_bytes(b'%PDF-1.7\nrelease test fixture\n')
        for attribute, value in (('ROOT', self.root), ('RESULTS', self.root / 'docs/results/full_cuda')):
            patcher = patch.object(verify_release, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_pdf_only_release_passes_without_removed_sources(self):
        self.assertFalse((self.root / 'main.tex').exists())
        self.assertFalse((self.root / 'docs/report.md').exists())
        with contextlib.redirect_stdout(io.StringIO()):
            verify_release.verify()

    def test_broken_document_link_still_fails(self):
        with (self.root / 'README.md').open('a') as stream:
            stream.write('\n[Missing source](main.tex)\n')
        with self.assertRaisesRegex(ValueError, 'Broken link in README.md: main.tex'):
            verify_release.verify()

    def test_missing_or_invalid_pdf_still_fails(self):
        report = self.root / 'report.pdf'
        report.unlink()
        with self.assertRaisesRegex(ValueError, 'report.pdf'):
            verify_release.verify()
        report.write_bytes(b'not a PDF')
        with self.assertRaisesRegex(ValueError, 'Invalid report PDF'):
            verify_release.verify()

    def test_report_archive_contains_pdf_without_latex_source(self):
        with patch.object(package_release, 'ROOT', self.root), \
                contextlib.redirect_stdout(io.StringIO()):
            package_release.main(include_checkpoints=False)
        release = self.root / 'output/release'
        with zipfile.ZipFile(release / 'AdversarialTraining-report.zip') as bundle:
            self.assertEqual(set(bundle.namelist()),
                             {'report.pdf', 'docs/results/full_cuda/comparison.png'})
            self.assertEqual(bundle.read('report.pdf'), (self.root / 'report.pdf').read_bytes())
        with zipfile.ZipFile(release / 'AdversarialTraining-source.zip') as bundle:
            self.assertIn('AdversarialTraining/report.pdf', bundle.namelist())


if __name__ == '__main__':
    unittest.main()
