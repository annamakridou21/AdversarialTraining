"""Compare measured robustness from two analysis JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_comparison(standard_path, robust_path):
    records = [json.loads(Path(path).read_text()) for path in (standard_path, robust_path)]
    for record in records:
        if record.get('format') != 'cifar10-pixel-space-v1':
            raise ValueError('Analysis must come from the corrected pixel-space pipeline')
    for key in ('steps', 'restarts', 'alpha', 'eval_limit', 'seed', 'batch_size'):
        if records[0]['config'][key] != records[1]['config'][key]:
            raise ValueError(f'Cannot compare different evaluation settings: {key}')
    for record in records:
        if record.get('evaluation_split') != 'test' or not record['results']:
            raise ValueError('Comparison requires non-empty test-set analyses')
    if records[0]['results'].keys() != records[1]['results'].keys():
        raise ValueError('Both analyses must use the same epsilon values')
    for key in records[0]['results']:
        if records[0]['results'][key]['total_samples'] != records[1]['results'][key]['total_samples']:
            raise ValueError('Both analyses must evaluate the same number of samples')
    return records


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--standard', default='output/analysis/standard_robustness.json')
    result.add_argument('--robust', default='output/analysis/robust_robustness.json')
    result.add_argument('--output', default='output/plots/robustness_comparison.png')
    return result


def main(args):
    records = load_comparison(args.standard, args.robust)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    any_asr = False
    for record, label in zip(records, ('Standard', 'Adversarial')):
        keys = sorted(record['results'], key=float)
        budgets = [float(key) * 255 for key in keys]
        axes[0].plot(budgets, [record['results'][k]['robust_accuracy'] for k in keys],
                     marker='o', label=label)
        asr_values = [record['results'][k]['attack_success_rate'] for k in keys]
        any_asr = any_asr or any(value is not None for value in asr_values)
        axes[1].plot(budgets, asr_values, marker='o', label=label)
    for axis, ylabel in zip(axes, ('Robust accuracy (%)', 'Attack success rate (%)')):
        axis.set(xlabel='Perturbation budget (1/255)', ylabel=ylabel, ylim=(0, 100))
        axis.set_xlim(min(budgets) - 0.25, max(budgets) + 0.25)
        axis.grid(alpha=0.2)
        axis.legend()
    if not any_asr:
        axes[1].text(0.5, 0.5, 'ASR undefined:\nno clean-correct samples',
                     ha='center', va='center', transform=axes[1].transAxes)
    config = records[0]['config']
    title = f"PGD-{config['steps']}, {config['restarts']} restart(s)"
    if config['eval_limit'] is not None:
        title += f" | Subset check: {config['eval_limit']} test samples"
    fig.suptitle(title)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main(parser().parse_args())
