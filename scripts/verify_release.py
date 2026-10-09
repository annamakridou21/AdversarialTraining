"""Check the portable study's provenance, counts, checksums, and document links."""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'docs/results/full_cuda'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    load = lambda name: json.loads((RESULTS / name).read_text(encoding='utf-8'))
    require(load('status.json') == {'state': 'complete', 'seeds': [42]}, 'Run is incomplete')
    manifest = load('experiment.json')
    for name, expected in manifest['source_sha256'].items():
        require(sha256(ROOT / name) == expected, f'Training source mismatch: {name}')
    require(sha256(ROOT / 'experiments/full.json') == manifest['protocol_sha256'], 'Protocol mismatch')
    protocol = load('protocol.json')
    require(protocol == manifest['protocol'], 'Exported protocol differs')
    hashes = load('checkpoint_hashes.json')
    records = [load(f'{branch}.json') for branch in ('clean_control', 'adversarial')]
    initial = []
    for branch, record in zip(('clean_control', 'adversarial'), records):
        config, checkpoint = record['config'], record['checkpoint']
        require(record['evaluation_split'] == 'test', 'Wrong evaluation split')
        require(config['steps'] == 20 and config['restarts'] == 5 and config['eval_limit'] is None,
                'Unexpected evaluation protocol')
        require(config['epsilons'] == protocol['epsilons'] and config['alpha'] == protocol['alpha'],
                'Attack budget/step mismatch')
        require(config['seed'] == 42 and config['batch_size'] == 128, 'Unexpected evaluation settings')
        require(checkpoint['epoch'] == 4, 'Selected epoch changed')
        require(hashes[config['checkpoint']] == checkpoint['sha256'], 'Selected hash differs')
        initial.append(checkpoint['training_config']['pretrain_sha256'])
        history = load(f'{branch}_history.json')
        require(history['epochs'] == [1, 2, 3, 4, 5], 'Branch history is incomplete')
        require(history['learning_rate'] == [0.001, 0.001, 0.001, 0.0001, 0.00001], 'Learning-rate schedule differs')
        metric = 'validation_clean_acc' if branch == 'clean_control' else 'validation_robust_acc'
        require(history[metric].index(max(history[metric])) + 1 == checkpoint['epoch'],
                'Checkpoint not selected by recorded validation history')
        require(history['config']['sample_counts'] == {'train': 45000, 'validation': 5000, 'test': 10000},
                'Data split mismatch')
        for budget, metrics in record['results'].items():
            require(float(budget) in protocol['epsilons'], 'Unexpected result budget')
            n, clean, robust = (metrics[k] for k in ('total_samples', 'clean_correct', 'robust_correct'))
            require(n == 10000 and 0 <= robust <= clean <= n, 'Invalid metric counts')
            for key, expected in {'clean_accuracy': 100*clean/n, 'robust_accuracy': 100*robust/n,
                                  'attack_success_rate': 100*(clean-robust)/clean}.items():
                require(abs(metrics[key] - expected) < 1e-9, f'Incorrect metric: {key}')
        require(len(record['results']) == 4, 'Missing result budgets')
    require(initial[0] == initial[1], 'Initial weights differ')
    require(initial[0] in hashes.values(), 'Initialization is not in checkpoint manifest')
    require(load('pretrain_history.json')['epochs'] == list(range(1, 11)), 'Incomplete pre-training')

    checked = set()
    for line in (RESULTS / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        require(sha256(RESULTS / name) == expected, f'Export checksum mismatch: {name}')
        checked.add(name)
    require(checked == {p.name for p in RESULTS.iterdir() if p.is_file() and p.name != 'checksums.sha256'},
            'Checksum manifest does not cover every exported file')
    for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]:
        text = path.read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if re.match(r'^[a-z]+:', target) or target.startswith('#'):
                continue
            local = unquote(target.split('#', 1)[0])
            require((path.parent / local).exists(), f'Broken link in {path.name}: {target}')
    tex = (ROOT / 'main.tex').read_text(encoding='utf-8')
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex):
        require((ROOT / name).is_file(), f'Missing report figure: {name}')
    for token in ('acmDOI', 'acmISBN', '76.44', '36.40', '4.8M', 'eleven experiments'):
        require(token not in tex, f'Historical claim remains in report: {token}')
    report = ROOT / 'report.pdf'
    require(report.read_bytes().startswith(b'%PDF-'), 'Compiled report is missing')
    summary = (RESULTS / 'summary.md').read_text(encoding='utf-8')
    for record in records:
        for metrics in record['results'].values():
            require(f"{metrics['robust_accuracy']:.2f}" in summary, 'Summary misses a measured result')
    print('PASS: training-source/protocol hashes, counts, selection, shared initialization, export checksums, document links, and report assets.')


if __name__ == '__main__':
    verify()
