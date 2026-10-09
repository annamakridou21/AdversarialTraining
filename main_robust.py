"""Train with a weighted mixture of clean and PGD adversarial loss.

The adversarial loss weight is configurable; a weight of 1 recovers the pure
min-max objective. Fine-tuning initializes weights from a clean checkpoint.
"""

from training import build_parser, run_training, train_epoch


def train_robust(model, train_loader, optimizer, criterion, attack, device, epoch, adv_ratio=0.5):
    return train_epoch(model, train_loader, optimizer, criterion, device, epoch, attack, adv_ratio)


def main(args):
    run_training(args, robust=True)


if __name__ == '__main__':
    main(build_parser(robust=True).parse_args())
