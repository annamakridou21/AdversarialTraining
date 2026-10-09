"""Create a verified source release, report-source ZIP, and optional local run backup."""

import argparse
import hashlib
from pathlib import Path
import zipfile

from .verify_release import verify

ROOT = Path(__file__).resolve().parents[1]


def archive(path, entries):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for source, name in entries:
            bundle.write(source, name)
    with zipfile.ZipFile(path) as bundle:
        bad = bundle.testzip()
        if bad:
            raise ValueError(f'Archive integrity failure: {bad}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    path.with_suffix('.sha256').write_text(f'{digest}  {path.name}\n', encoding='utf-8')
    print(f'Verified {path.name}: {path.stat().st_size / 1024**2:.1f} MiB')


def main(include_checkpoints):
    verify()
    output = ROOT / 'output/release'
    output.mkdir(parents=True, exist_ok=True)
    suffixes = {'.py', '.md', '.json', '.yml', '.yaml', '.png', '.pdf', '.tex', '.txt', '.sha256', '.sh'}
    source = [p for p in ROOT.iterdir() if p.is_file() and
              (p.suffix in suffixes or p.name in ('.gitignore', '.gitattributes'))]
    for directory in ('.github', 'docs', 'experiments', 'scripts', 'tests'):
        source += [p for p in (ROOT / directory).rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.suffix in suffixes]
    archive(output / 'AdversarialTraining-source.zip',
            [(p, 'AdversarialTraining/' + p.relative_to(ROOT).as_posix()) for p in sorted(source)])
    report_files = [ROOT / 'main.tex', ROOT / 'docs/results/full_cuda/comparison.png']
    archive(output / 'AdversarialTraining-report-source.zip',
            [(p, p.relative_to(ROOT).as_posix()) for p in report_files])
    if include_checkpoints:
        run = ROOT / 'output/experiments/full_cuda'
        entries = [(p, 'full_cuda/' + p.relative_to(run).as_posix())
                   for p in sorted(run.rglob('*')) if p.is_file() and p.name != '.run.lock']
        if not any(p.suffix == '.pth' for p, _ in entries):
            raise FileNotFoundError('The original local checkpoints are unavailable')
        archive(output / 'AdversarialTraining-full-run-backup.zip', entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-checkpoints', action='store_true', help='Also back up the local completed run')
    args = parser.parse_args()
    main(args.include_checkpoints)
