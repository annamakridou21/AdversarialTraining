"""Shared command-line options for checkpoint evaluation."""

import argparse
from pathlib import Path
import sys

# Support both `python scripts/name.py` and `python -m scripts.name`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_arch import CNN
from utils import (file_sha256, get_cifar10_test_loader, read_checkpoint,
                   seed_everything, select_device)


def evaluation_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-limit', type=int, help='Test subset size; for smoke tests only')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--restarts', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=2/255)
    return parser


def load_evaluation(args):
    seed_everything(args.seed)
    device = select_device(args.device)
    model = CNN().to(device)
    checkpoint = read_checkpoint(model, args.checkpoint, device)
    model.checkpoint_metadata = dict(sha256=file_sha256(args.checkpoint),
        epoch=checkpoint['epoch'], training_config=checkpoint['config'])
    loader = get_cifar10_test_loader(args.batch_size, args.num_workers, args.data_dir,
                                     eval_limit=args.eval_limit)
    model.eval()
    return model, loader, device
