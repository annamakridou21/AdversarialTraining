"""Time representative batches before scheduling full CIFAR-10 experiments."""

import argparse
import time

if __package__ in (None, ''):
    import _common
else:
    from . import _common

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attacks import PGDAttack
from model_arch import CNN
from training import train_epoch
from utils import seed_everything, select_device, write_json


def main(args):
    seed_everything(42)
    device = select_device(args.device)
    torch.set_num_threads(args.threads)
    images = torch.rand(args.batch_size, 3, 32, 32)
    labels = torch.randint(10, (args.batch_size,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size)
    timings = {}
    for mode in ('standard', 'robust'):
        model = CNN().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        attack = PGDAttack(model, num_iter=args.steps, device=device,
                           selection='loss') if mode == 'robust' else None
        elapsed = []
        for repeat in range(args.batches + 1):
            if device.type == 'mps':
                torch.mps.synchronize()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), device, repeat, attack)
            if device.type == 'mps':
                torch.mps.synchronize()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if repeat:
                elapsed.append(time.perf_counter() - start)
        timings[mode] = sum(elapsed) / len(elapsed)
        print(f'{mode}: {timings[mode]:.3f} seconds per batch', flush=True)
    write_json(args.output, dict(device=str(device), batch_size=args.batch_size,
                                pgd_steps=args.steps, seconds_per_batch=timings,
                                purpose='runtime_estimate_on_synthetic_images'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', default='output/verification/runtime.json')
    main(parser.parse_args())
