"""Run pre-training, a clean control, and adversarial fine-tuning from one protocol."""

import argparse
import json
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    import _common
else:
    from . import _common

from utils import file_sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def build_stages(protocol, seed, output, data_dir, device, workers):
    common = ['--batch-size', str(protocol['batch_size']), '--seed', str(seed),
              '--validation-size', str(protocol['validation_size']), '--device', device,
              '--num-workers', str(workers), '--data-dir', str(data_dir),
              '--eval-steps', str(protocol['validation_steps']), '--eval-alpha', str(protocol['alpha']),
              '--eval-epsilon', str(protocol['epsilon']), '--skip-test']
    for key in ('train_limit', 'eval_limit'):
        if protocol[key] is not None:
            common += ['--' + key.replace('_', '-'), str(protocol[key])]
    stages = []
    for name, robust in [('pretrain', False), ('clean_control', False), ('adversarial', True)]:
        pretrain = name == 'pretrain'
        phase = 'pretrain' if pretrain else 'finetune'
        prefix = 'robust' if robust else 'standard'
        directory = output / name
        command = [sys.executable, str(ROOT / f'main_{prefix}.py'), *common,
                   '--output-dir', str(directory), '--epochs', str(protocol[f'{phase}_epochs']),
                   '--lr', str(protocol[f'{phase}_lr']), '--lr-milestones',
                   *map(str, protocol[f'{phase}_milestones'])]
        if robust:
            command += ['--train-steps', str(protocol['train_steps']),
                        '--train-epsilon', str(protocol['epsilon']),
                        '--train-alpha', str(protocol['alpha']), '--adv-ratio', str(protocol['adv_ratio'])]
        latest = directory / 'models' / f'{prefix}_latest.pth'
        if latest.exists():
            command += ['--resume', str(latest)]
        elif not pretrain:
            # Both branches start from exactly the same final pre-training weights.
            command += ['--pretrain-path', str(output / 'pretrain/models/standard_final.pth')]
        stages.append((name, command))
    for name, prefix in [('clean_control', 'standard'), ('adversarial', 'robust')]:
        command = [sys.executable, '-m', 'scripts.analyze_robustness',
                   '--checkpoint', str(output / name / 'models' / f'{prefix}_best.pth'),
                   '--save-results', str(output / 'analysis' / f'{name}.json'),
                   '--data-dir', str(data_dir), '--device', device, '--num-workers', str(workers),
                   '--batch-size', str(protocol['batch_size']), '--seed', str(seed),
                   '--steps', str(protocol['analysis_steps']), '--restarts', str(protocol['analysis_restarts']),
                   '--alpha', str(protocol['alpha']), '--epsilons', *map(str, protocol['epsilons'])]
        if protocol['eval_limit'] is not None:
            command += ['--eval-limit', str(protocol['eval_limit'])]
        stages.append((f'evaluate_{name}', command))
    comparison = ['--standard', str(output / 'analysis/clean_control.json'),
                  '--robust', str(output / 'analysis/adversarial.json')]
    for module, filename in [('create_comparison_plots', 'comparison.png'),
                             ('create_summary_report', 'summary.md')]:
        stages.append((module, [sys.executable, '-m', f'scripts.{module}', *comparison,
                               '--output', str(output / filename)]))
    return stages


def execute_plan(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be distinct')
    protocol_path = ROOT / 'experiments' / f'{args.profile}.json'
    protocol = json.loads(protocol_path.read_text())
    output = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    manifest = dict(profile=args.profile, protocol=protocol, seeds=args.seeds,
                    device=args.device, num_workers=args.num_workers, data_dir=str(data_dir),
                    protocol_sha256=file_sha256(protocol_path),
                    source_sha256={str(path.relative_to(ROOT)): file_sha256(path)
                                   for path in sorted(list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*.py')))})
    manifest_path = output / 'experiment.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing experiment uses a different protocol; choose a new output directory')
    if args.execute:
        write_json(manifest_path, manifest)
    for seed in args.seeds:
        seed_output = output / f'seed_{seed}'
        stages = build_stages(protocol, seed, seed_output, data_dir, args.device, args.num_workers)
        for name, command in stages:
            print(f'seed={seed}, stage={name}: {subprocess.list2cmdline(command)}', flush=True)
            if not args.execute:
                continue
            log = seed_output / 'logs' / f'{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            write_json(output / 'status.json', dict(state='running', seed=seed, stage=name, log=str(log)))
            environment = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
            with log.open('a') as stream:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                        stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(output / 'status.json', dict(state='failed', seed=seed, stage=name,
                                                       log=str(log), exit_code=result.returncode))
                raise RuntimeError(f'{name} failed; see {log}')
    if args.execute:
        write_json(output / 'status.json', dict(state='complete', seeds=args.seeds))


@contextmanager
def experiment_lock(output):
    """Use the platform's file lock; process exit releases it automatically."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a+b') as lock:
        if sys.platform == 'win32':
            import msvcrt
            # Windows locks a byte range, so reserve a byte even for a new file.
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b'\0')
                lock.flush()
            lock.seek(0)
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            def release():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise RuntimeError('This experiment is already running') from error
            raise
        try:
            yield
        finally:
            release()


def main(args):
    if not args.execute:
        return execute_plan(args)
    with experiment_lock(args.output_dir):
        execute_plan(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['full', 'pilot'], default='full')
    parser.add_argument('--output-dir', default='output/experiments/full')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--execute', action='store_true', help='Execute the printed plan')
    main(parser.parse_args())
