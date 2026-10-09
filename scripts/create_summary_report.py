"""Write a Markdown comparison from measured analysis outputs."""

from pathlib import Path

if __package__ in (None, ''):
    from create_comparison_plots import load_comparison, parser
else:
    from .create_comparison_plots import load_comparison, parser


def main(args):
    records = load_comparison(args.standard, args.robust)
    lines = ['# CIFAR-10 robustness evaluation', '',
             'Measured on the test split. Robust accuracy requires clean and attacked correctness.', '',
             '| Model | Budget | Samples | Clean (%) | Robust (%) | ASR (%) |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for record, label in zip(records, ('Standard', 'Adversarial')):
        for epsilon in sorted(record['results'], key=float):
            m = record['results'][epsilon]
            asr = 'undefined' if m['attack_success_rate'] is None else f"{m['attack_success_rate']:.2f}"
            lines.append(f"| {label} | {float(epsilon)*255:g}/255 | {m['total_samples']} | "
                         f"{m['clean_accuracy']:.2f} | {m['robust_accuracy']:.2f} | {asr} |")
    config = records[0]['config']
    if config['eval_limit'] is not None:
        lines += ['', '**Subset evaluation only; these are not full CIFAR-10 benchmark results.**']
    lines += ['', '## Checkpoint provenance', '']
    for record, label in zip(records, ('Standard', 'Adversarial')):
        metadata = record.get('checkpoint')
        if metadata:
            lines += [f"- {label}: selected epoch {metadata['epoch']}, SHA-256 `{metadata['sha256']}`."]
            training = metadata['training_config']
            lines += [f"  Stage training budget: {training['epochs']} epoch(s); "
                      f"samples: {training.get('sample_counts', {})}."]
    lines += ['', f"PGD settings: {config['steps']} steps, {config['restarts']} restarts, "
              f"step size {config['alpha']*255:g}/255; seed {config['seed']}.", '',
              'These are empirical attack results, not a certificate of robustness.', '']
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text('\n'.join(lines))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    cli = parser()
    cli.description = __doc__
    cli.set_defaults(output='output/analysis/summary.md')
    main(cli.parse_args())
