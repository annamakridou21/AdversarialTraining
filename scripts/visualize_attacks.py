"""Show consecutive test images and their model-specific PGD perturbations."""

from pathlib import Path
import matplotlib.pyplot as plt
import torch

if __package__ in (None, ''):
    from _common import evaluation_parser, load_evaluation
else:
    from ._common import evaluation_parser, load_evaluation

from attacks import PGDAttack, evaluating

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def visualize_adversarial_examples(model, test_loader, device, num_images=5,
                                    epsilon=4/255, steps=20, restarts=5, alpha=2/255):
    """Plot unfiltered examples; successful attacks are not selected in advance."""
    if num_images < 1:
        raise ValueError('num_images must be positive')
    batches, targets, count = [], [], 0
    for images, labels in test_loader:
        take = min(num_images - count, len(images))
        batches.append(images[:take])
        targets.append(labels[:take])
        count += take
        if count >= num_images:
            break
    if not batches:
        raise ValueError('Cannot visualize an empty dataset')
    images, labels = torch.cat(batches).to(device), torch.cat(targets).to(device)
    attack = PGDAttack(model, epsilon, alpha, steps, device, restarts)
    with evaluating(model):
        attacked = attack(images, labels)
        with torch.no_grad():
            clean_predictions = model(images).argmax(1).cpu()
            adv_predictions = model(attacked).argmax(1).cpu()
    images, attacked, labels = images.cpu(), attacked.cpu(), labels.cpu()
    fig, axes = plt.subplots(len(images), 3, figsize=(8, 2.5 * len(images)), squeeze=False)
    for i, (clean, adversarial) in enumerate(zip(images, attacked)):
        delta = adversarial - clean
        # A fixed scale makes perturbations comparable between examples.
        display_delta = (delta / (2 * epsilon) + 0.5).clamp(0, 1) if epsilon else delta + 0.5
        views = [clean, display_delta, adversarial]
        titles = [f'Clean: {CIFAR10_CLASSES[clean_predictions[i]]}',
                  f'Perturbation (scaled)\nmax |delta| = {delta.abs().max():.4f}',
                  f'PGD: {CIFAR10_CLASSES[adv_predictions[i]]}']
        for axis, view, title in zip(axes[i], views, titles):
            axis.imshow(view.permute(1, 2, 0).numpy())
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[i, 0].set_ylabel(f'True: {CIFAR10_CLASSES[labels[i]]}')
    fig.suptitle(f'PGD-{steps}, epsilon={epsilon * 255:g}/255, {restarts} restart(s)')
    fig.tight_layout()
    return fig


def main(args):
    model, loader, device = load_evaluation(args)
    fig = visualize_adversarial_examples(model, loader, device, args.num_images, args.epsilon,
                                         args.steps, args.restarts, args.alpha)
    output = Path(args.output_dir) / args.output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'Saved {output}')


def parser():
    result = evaluation_parser(__doc__)
    result.add_argument('--num-images', type=int, default=5)
    result.add_argument('--epsilon', type=float, default=4/255)
    result.add_argument('--output-dir', default='output/visualizations')
    result.add_argument('--output-name', default='adversarial_examples.png')
    result.add_argument('--show', action='store_true')
    return result


if __name__ == '__main__':
    main(parser().parse_args())
