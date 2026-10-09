"""Train the clean empirical-risk baseline on CIFAR-10."""

from training import build_parser, run_training, train_epoch


def train_standard(model, train_loader, optimizer, criterion, device, epoch):
    return train_epoch(model, train_loader, optimizer, criterion, device, epoch)


def main(args):
    run_training(args, robust=False)


if __name__ == '__main__':
    main(build_parser().parse_args())
