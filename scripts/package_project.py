"""Package the current project and CIFAR-10 archive for another machine."""

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', 'venv', 'venv-wsl', '.cv-draft', 'output', 'data',
            '__pycache__', '.pytest_cache', '.idea', '.vscode'}


def main():
    output = ROOT / 'output/transfer'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'AdversarialTraining-windows.zip'
    temporary = archive.with_suffix('.zip.tmp')
    dataset = ROOT / 'data/cifar-10-python.tar.gz'
    if not dataset.is_file():
        raise FileNotFoundError('Download CIFAR-10 before packaging this transfer bundle')
    with dataset.open('rb') as stream:
        digest = hashlib.md5()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != 'c58f30108f718f92721af3b95e74349a':
        raise ValueError('CIFAR-10 archive checksum mismatch')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(ROOT.rglob('*')):
            relative = item.relative_to(ROOT)
            if any(part in EXCLUDED for part in relative.parts) or item.is_symlink():
                continue
            if not item.is_file() or item.name == '.DS_Store' or item.suffix in ('.pyc', '.pth', '.pt'):
                continue
            bundle.write(item, (Path('AdversarialTraining') / relative).as_posix())
        bundle.write(dataset, 'AdversarialTraining/data/cifar-10-python.tar.gz',
                     compress_type=zipfile.ZIP_STORED)
    temporary.replace(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    checksum = output / 'AdversarialTraining-windows.sha256'
    checksum.write_text(f'{digest.hexdigest()}  {archive.name}\n')
    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member:
            raise RuntimeError(f'Archive verification failed: {bad_member}')
    print(f'Created and verified {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)')
    print(f'SHA-256: {digest.hexdigest()}')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
