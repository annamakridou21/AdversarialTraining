"""Shared training workflow for clean and adversarial CIFAR-10 experiments."""

import argparse
import math
import time
import sys
from importlib.metadata import version
from pathlib import Path
import platform

import torch
import torch.nn as nn
from tqdm import tqdm

from attacks import PGDAttack
from model_arch import CNN
from utils import (atomic_write, capture_rng_state, restore_rng_state, file_sha256,
                   evaluate_clean, evaluate_metrics, get_cifar10_loaders,
                   read_checkpoint, save_checkpoint, seed_everything, select_device, write_json)


def train_epoch(model, loader, optimizer, criterion, device, epoch, attack=None, adv_ratio=0.5):
    """Return sample-weighted loss and accuracy (attacked accuracy when enabled)."""
    if not 0 <= adv_ratio <= 1:
        raise ValueError('adv_ratio must be in [0, 1]')
    model.train()
    loss_sum = correct = total = 0
    for batch_index, (images, labels) in enumerate(tqdm(
            loader, desc=f'Epoch {epoch}', leave=False, disable=not sys.stderr.isatty())):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if attack is not None and adv_ratio > 0:
            images_adv = attack(images, labels)
            outputs = model(images_adv)
            loss = adv_ratio * criterion(outputs, labels)
            if adv_ratio < 1:
                loss = loss + (1 - adv_ratio) * criterion(model(images), labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss; checkpoint was not updated')
        loss.backward()
        optimizer.step()
        total += labels.numel()
        correct += outputs.argmax(1).eq(labels).sum().item()
        loss_sum += loss.item() * labels.numel()
        if (batch_index + 1) % 50 == 0 and not sys.stderr.isatty():
            print(f'Epoch {epoch}, batch {batch_index + 1}/{len(loader)}: '
                  f'loss={loss_sum / total:.4f}', flush=True)
    if not total:
        raise ValueError('Cannot train on an empty dataset')
    return loss_sum / total, 100 * correct / total


def build_parser(robust=False):
    parser = argparse.ArgumentParser(description='CIFAR-10 adversarial training' if robust
                                     else 'CIFAR-10 standard training')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01 if robust else 0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr-milestones', type=int, nargs='+', default=[75, 90])
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--eval-epsilon', type=float, default=4/255)
    parser.add_argument('--eval-alpha', type=float, default=2/255)
    parser.add_argument('--eval-steps', type=int, default=10)
    parser.add_argument('--eval-restarts', type=int, default=1)
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--save-freq', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--resume', help='Resume an epoch-boundary latest checkpoint in its run directory')
    parser.add_argument('--pretrain-path', help='Initialize weights for clean continuation or adversarial fine-tuning')
    parser.add_argument('--skip-test', action='store_true', help='Reserve test evaluation for a later analysis run')
    parser.add_argument('--validation-size', type=int, default=5000)
    parser.add_argument('--train-limit', type=int, help='Subset size for smoke tests only')
    parser.add_argument('--eval-limit', type=int, help='Validation/test subset size for smoke tests')
    if robust:
        parser.add_argument('--train-epsilon', type=float, default=4/255)
        parser.add_argument('--train-alpha', type=float, default=2/255)
        parser.add_argument('--train-steps', type=int, default=10)
        parser.add_argument('--adv-ratio', type=float, default=0.5,
                            help='Adversarial loss weight; 1 means pure adversarial training')
    return parser


def run_training(args, robust=False):
    for key in ('epochs', 'batch_size', 'eval_freq', 'save_freq', 'eval_steps', 'eval_restarts'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if args.num_workers < 0:
        raise ValueError('num_workers must be non-negative')
    for key in ('lr', 'momentum', 'weight_decay', 'lr_gamma', 'dropout'):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f'{key} must be finite')
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.momentum < 1:
        raise ValueError('Require lr > 0, weight_decay >= 0, and momentum in [0, 1)')
    if not 0 <= args.dropout < 1 or not 0 < args.lr_gamma <= 1:
        raise ValueError('Require dropout in [0, 1) and lr_gamma in (0, 1]')
    if args.resume and args.pretrain_path:
        raise ValueError('Use either resume or pretrain-path, not both')
    if robust and not 0 <= args.adv_ratio <= 1:
        raise ValueError('adv_ratio must be in [0, 1]')
    seed_everything(args.seed)
    device = select_device(args.device)
    prefix = 'robust' if robust else 'standard'
    output = Path(args.output_dir)
    if not args.resume and (output / f'{prefix}_config.json').exists():
        raise FileExistsError(f'Run already exists in {output}; use a new output directory or --resume')
    config = dict(vars(args), training=prefix, input_space='pixels_[0,1]',
                  device_used=str(device), torch_version=str(torch.__version__),
                  python_version=platform.python_version(),
                  dependency_versions={name: version(name) for name in
                                       ('torch', 'torchvision', 'numpy', 'matplotlib', 'tqdm')})
    print(f'Training {prefix} model on {device}')
    model = CNN(dropout_rate=args.dropout).to(device)
    if args.pretrain_path:
        checkpoint = read_checkpoint(model, args.pretrain_path, device)
        config['pretrain_sha256'] = file_sha256(args.pretrain_path)
        old_config = checkpoint['config']
        if (old_config.get('seed'), old_config.get('validation_size')) != (
                args.seed, args.validation_size):
            raise ValueError('Pre-training and fine-tuning must use the same validation split')
    loaders = get_cifar10_loaders(args.batch_size, args.num_workers, args.data_dir,
                                  args.seed, args.validation_size, args.train_limit, args.eval_limit)
    train_loader, validation_loader, test_loader = loaders
    config['sample_counts'] = dict(zip(('train', 'validation', 'test'),
                                       (len(loader.dataset) for loader in loaders)))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones,
                                                     gamma=args.lr_gamma)
    train_attack = PGDAttack(model, args.train_epsilon, args.train_alpha, args.train_steps,
                             device, selection='loss') if robust else None
    eval_attack = PGDAttack(model, args.eval_epsilon, args.eval_alpha, args.eval_steps,
                            device, args.eval_restarts)
    history = dict(config=config, epochs=[], train_loss=[], train_acc=[],
                   validation_clean_acc=[], validation_robust_acc=[], learning_rate=[], epoch_seconds=[])
    best_accuracy = -1.0
    best_epoch = 0
    start_epoch = 1
    best_path = output / 'models' / f'{prefix}_best.pth'
    metric = 'validation_robust_accuracy' if robust else 'validation_clean_accuracy'
    if args.resume:
        resumed = read_checkpoint(model, args.resume, device)
        if 'training_state' not in resumed:
            raise ValueError('Resume requires a latest checkpoint with training state')
        old_config = resumed['config']
        if Path(old_config['output_dir']).resolve() != output.resolve():
            raise ValueError('Resume in the original output directory')
        required = ['training', 'batch_size', 'lr', 'momentum', 'weight_decay', 'dropout',
                    'lr_milestones', 'lr_gamma', 'eval_epsilon', 'eval_alpha', 'eval_steps',
                    'eval_restarts', 'eval_freq', 'seed', 'validation_size', 'train_limit',
                    'eval_limit', 'num_workers', 'device_used', 'dependency_versions']
        if robust:
            required += ['train_epsilon', 'train_alpha', 'train_steps', 'adv_ratio']
        for key in required:
            if old_config.get(key) != config.get(key):
                raise ValueError(f'Resume configuration differs: {key}')
        state = resumed['training_state']
        best_accuracy, best_epoch = state['best_accuracy'], state['best_epoch']
        if best_epoch:
            if 'best_checkpoint' in state:
                atomic_write(best_path, lambda stream: torch.save(state['best_checkpoint'], stream))
            best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
            if best_checkpoint['epoch'] != best_epoch:
                raise ValueError('Best checkpoint does not match the resumed run')
        optimizer.load_state_dict(resumed['optimizer_state_dict'])
        scheduler.load_state_dict(resumed['scheduler_state_dict'])
        history = state['history']
        start_epoch = resumed['epoch'] + 1
        if args.epochs < resumed['epoch']:
            raise ValueError('epochs cannot precede the resume checkpoint')
        config['pretrain_path'] = old_config.get('pretrain_path')
        if 'pretrain_sha256' in old_config:
            config['pretrain_sha256'] = old_config['pretrain_sha256']
        history['config'] = config
        restore_rng_state(state['rng'], loaders, device)
    write_json(output / f'{prefix}_config.json', config)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        loss, accuracy = train_epoch(model, train_loader, optimizer, nn.CrossEntropyLoss(),
                                     device, epoch, train_attack, args.adv_ratio if robust else 0)
        robust_accuracy = None
        if robust and (epoch % args.eval_freq == 0 or epoch == args.epochs):
            validation = evaluate_metrics(model, validation_loader, device, eval_attack)
            clean_accuracy = validation['clean_accuracy']
            robust_accuracy = validation['robust_accuracy']
        else:
            clean_accuracy, _ = evaluate_clean(model, validation_loader, device)
        for key, value in [('epochs', epoch), ('train_loss', loss), ('train_acc', accuracy),
                           ('validation_clean_acc', clean_accuracy),
                           ('validation_robust_acc', robust_accuracy),
                           ('learning_rate', optimizer.param_groups[0]['lr']),
                           ('epoch_seconds', time.perf_counter() - epoch_start)]:
            history[key].append(value)
        scheduler.step()
        selection_accuracy = robust_accuracy if robust else clean_accuracy
        if selection_accuracy is not None and selection_accuracy > best_accuracy:
            best_accuracy = selection_accuracy
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_accuracy, best_path,
                            config=config, scheduler=scheduler, metric=metric)
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            name = f'{prefix}_final.pth' if epoch == args.epochs else f'{prefix}_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, clean_accuracy, output / 'models' / name,
                            config=config, scheduler=scheduler, metric='validation_clean_accuracy')
        state = dict(best_accuracy=best_accuracy, best_epoch=best_epoch, history=history,
                     best_checkpoint=torch.load(best_path, map_location='cpu', weights_only=True),
                     rng=capture_rng_state(loaders, device))
        save_checkpoint(model, optimizer, epoch, clean_accuracy,
                        output / 'models' / f'{prefix}_latest.pth', config=config,
                        scheduler=scheduler, metric='validation_clean_accuracy', training_state=state)
        write_json(output / f'{prefix}_history.json', history)
        print(f'Epoch {epoch}: loss={loss:.4f}, validation clean={clean_accuracy:.2f}%, '
              f'validation robust={robust_accuracy}')
    if args.skip_test:
        print(f'Training complete; best validation checkpoint: {best_path}')
        return
    checkpoint = read_checkpoint(model, best_path, device)
    results = evaluate_metrics(model, test_loader, device, eval_attack)
    write_json(output / f'{prefix}_test.json', dict(config=config, checkpoint=str(best_path),
               selected_epoch=checkpoint['epoch'], checkpoint_sha256=file_sha256(best_path), metrics=results))
    print(f"Selected epoch {checkpoint['epoch']}; test clean={results['clean_accuracy']:.2f}%, "
          f"test robust={results['robust_accuracy']:.2f}%")
