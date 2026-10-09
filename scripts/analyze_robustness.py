"""Evaluate a checkpoint against PGD over a range of pixel-space budgets."""

if __package__ in (None, ''):
    from _common import evaluation_parser, load_evaluation
else:
    from ._common import evaluation_parser, load_evaluation

from attacks import PGDAttack
from utils import CHECKPOINT_FORMAT, evaluate_metrics, write_json


def main(args):
    model, loader, device = load_evaluation(args)
    results = {}
    for epsilon in args.epsilons:
        attack = PGDAttack(model, epsilon, args.alpha, args.steps, device, args.restarts)
        metrics = evaluate_metrics(model, loader, device, attack)
        results[str(epsilon)] = metrics
        print(f"epsilon={epsilon * 255:g}/255: clean={metrics['clean_accuracy']:.2f}%, "
              f"robust={metrics['robust_accuracy']:.2f}%, ASR={metrics['attack_success_rate']}")
    write_json(args.save_results, dict(format=CHECKPOINT_FORMAT, config=vars(args),
               evaluation_split='test', attack_selection='success',
               checkpoint=model.checkpoint_metadata, results=results))


if __name__ == '__main__':
    parser = evaluation_parser(__doc__)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[2/255, 4/255, 6/255, 8/255])
    parser.add_argument('--save-results', default='output/analysis/robustness.json')
    main(parser.parse_args())
