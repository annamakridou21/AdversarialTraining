"""Plot measured training and validation history from a training run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(log_file, output_path='output/plots/training_curves.png'):
    history = json.loads(Path(log_file).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history['epochs'], history['train_loss'])
    axes[0].set(xlabel='Epoch', ylabel='Training loss')
    for key, label in [('train_acc', 'Training'), ('validation_clean_acc', 'Validation clean'),
                       ('validation_robust_acc', 'Validation robust')]:
        pairs = [(epoch, value) for epoch, value in zip(history['epochs'], history[key])
                 if value is not None]
        if pairs:
            epochs, values = zip(*pairs)
            axes[1].plot(epochs, values, marker='o', label=label)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', ylim=(0, 100))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-file', required=True)
    parser.add_argument('--output', default='output/plots/training_curves.png')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    fig = plot_training_curves(args.log_file, args.output)
    if args.show:
        plt.show()
    plt.close(fig)
