"""Verify and export the completed CUDA study with portable artifact paths."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def export(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if read(run / 'status.json') != {'state': 'complete', 'seeds': [42]}:
        raise ValueError('This export requires a completed seed 42 study')
    manifest = read(run / 'experiment.json')
    for name, expected in manifest['source_sha256'].items():
        if digest(ROOT / name.replace('\\', '/')) != expected:
            raise ValueError(f'Training-time source changed: {name}')
    if digest(ROOT / 'experiments/full.json') != manifest['protocol_sha256']:
        raise ValueError('Protocol hash mismatch')
    original_root = manifest['data_dir'].replace('\\', '/').rsplit('/', 1)[0]

    def portable(value):
        if isinstance(value, dict):
            return {key.replace('\\', '/'): portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace('\\', '/')
            if normalized.startswith(original_root + '/'):
                return normalized[len(original_root) + 1:]
        return value

    destination.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        content = json.dumps(portable(value), indent=2, allow_nan=False) + '\n'
        (destination / name).write_bytes(content.encode('utf-8'))

    records = []
    checkpoints = {}
    for branch, prefix in [('clean_control', 'standard'), ('adversarial', 'robust')]:
        record = read(run / 'seed_42/analysis' / f'{branch}.json')
        checkpoint = run / 'seed_42' / branch / 'models' / f'{prefix}_best.pth'
        actual = digest(checkpoint)
        if actual != record['checkpoint']['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {branch}')
        for metrics in record['results'].values():
            n, clean, robust = (metrics[key] for key in
                                ('total_samples', 'clean_correct', 'robust_correct'))
            if n != 10000 or not 0 <= robust <= clean <= n:
                raise ValueError('Invalid evaluation counts')
            expected = {'clean_accuracy': 100 * clean / n,
                        'robust_accuracy': 100 * robust / n,
                        'attack_success_rate': 100 * (clean - robust) / clean}
            for key, value in expected.items():
                if abs(metrics[key] - value) > 1e-9:
                    raise ValueError(f'Metric differs from counts: {key}')
        checkpoints[checkpoint.relative_to(ROOT).as_posix()] = actual
        save(f'{branch}.json', record)
        save(f'{branch}_history.json', read(run / 'seed_42' / branch / f'{prefix}_history.json'))
        records.append(record)
    for key in ('steps', 'restarts', 'alpha', 'epsilons', 'seed', 'eval_limit', 'batch_size'):
        if records[0]['config'][key] != records[1]['config'][key]:
            raise ValueError(f'Unmatched evaluation: {key}')
    initial = [r['checkpoint']['training_config']['pretrain_sha256'] for r in records]
    pretrained = run / 'seed_42/pretrain/models/standard_final.pth'
    if initial[0] != initial[1] or initial[0] != digest(pretrained):
        raise ValueError('Branches did not share the saved initialization')
    checkpoints[pretrained.relative_to(ROOT).as_posix()] = initial[0]
    save('pretrain_history.json', read(run / 'seed_42/pretrain/standard_history.json'))
    save('experiment.json', manifest)
    save('protocol.json', manifest['protocol'])
    save('checkpoint_hashes.json', checkpoints)
    save('status.json', read(run / 'status.json'))
    shutil.copyfile(run / 'seed_42/comparison.png', destination / 'comparison.png')
    summary = (run / 'seed_42/summary.md').read_text(encoding='utf-8')
    (destination / 'summary.md').write_bytes(summary.encode('utf-8'))
    for document in destination.glob('*.md'):
        document.write_bytes(document.read_text(encoding='utf-8').encode('utf-8'))
    checksums = ''.join(
        f'{digest(path)}  {path.name}\n' for path in sorted(destination.iterdir())
        if path.is_file() and path.name != 'checksums.sha256')
    (destination / 'checksums.sha256').write_bytes(checksums.encode('utf-8'))
    print(f'Verified source, protocol, checkpoints, matched initialization, and metrics: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='output/experiments/full_cuda')
    parser.add_argument('--output-dir', default='docs/results/full_cuda')
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)
