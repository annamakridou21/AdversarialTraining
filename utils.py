"""Dataset splits, evaluation metrics, and experiment serialization."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from attacks import evaluating

CHECKPOINT_FORMAT = 'cifar10-pixel-space-v1'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_cifar10_loaders(batch_size=128, num_workers=2, data_dir='./data',
                        seed=42, validation_size=5000, train_limit=None,
                        eval_limit=None, download=True):
    """Return train/validation/test loaders; only training uses augmentation.

    Validation indices come from the training split. Limits support quick checks
    and must never be confused with full-dataset experiments.
    """
    if not 0 < validation_size < 50000:
        raise ValueError('validation_size must be between 1 and 49999')
    augment = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train = datasets.CIFAR10(data_dir, train=True, download=download, transform=augment)
    validation = datasets.CIFAR10(data_dir, train=True, download=download,
                                  transform=transforms.ToTensor())
    test = datasets.CIFAR10(data_dir, train=False, download=download,
                            transform=transforms.ToTensor())
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed)).tolist()
    validation = Subset(validation, indices[:validation_size])
    train = Subset(train, indices[validation_size:])

    def limit(dataset, count):
        if count is None:
            return dataset
        if count < 1 or count > len(dataset):
            raise ValueError(f'sample limit must be between 1 and {len(dataset)}')
        return Subset(dataset, range(count))

    train, validation, test = (limit(train, train_limit), limit(validation, eval_limit),
                               limit(test, eval_limit))
    return tuple(DataLoader(dataset, batch_size=batch_size, shuffle=(i == 0),
                            num_workers=num_workers,
                            generator=torch.Generator().manual_seed(seed + i))
                 for i, dataset in enumerate((train, validation, test)))


def get_cifar10_test_loader(batch_size=128, num_workers=2, data_dir='./data',
                            eval_limit=None, download=True):
    """Load only the held-out test split for standalone evaluation."""
    dataset = datasets.CIFAR10(data_dir, train=False, download=download,
                               transform=transforms.ToTensor())
    if eval_limit is not None:
        if not 1 <= eval_limit <= len(dataset):
            raise ValueError(f'eval_limit must be between 1 and {len(dataset)}')
        dataset = Subset(dataset, range(eval_limit))
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


def evaluate_metrics(model, loader, device, attack=None):
    """Measure clean and attacked accuracy on the same samples.

    Robust accuracy requires both clean and attacked predictions to be correct.
    ASR is conditional on clean correctness; it is undefined when that count is 0.
    """
    total = clean_correct = adv_correct = robust_correct = 0
    clean_loss = adv_loss = 0.0
    with evaluating(model):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                clean_ok = logits.argmax(1).eq(labels)
                clean_loss += F.cross_entropy(logits, labels, reduction='sum').item()
            attacked = attack(images, labels) if attack is not None else None
            with torch.no_grad():
                logits_adv = model(attacked) if attacked is not None else logits
                adv_ok = logits_adv.argmax(1).eq(labels)
                adv_loss += F.cross_entropy(logits_adv, labels, reduction='sum').item()
            total += labels.numel()
            clean_correct += clean_ok.sum().item()
            adv_correct += adv_ok.sum().item()
            robust_correct += (clean_ok & adv_ok).sum().item()
    if total == 0:
        raise ValueError('Cannot evaluate an empty dataset')
    return dict(total_samples=total, clean_correct=clean_correct,
                clean_accuracy=100 * clean_correct / total, clean_loss=clean_loss / total,
                adv_correct=adv_correct, adv_accuracy=100 * adv_correct / total,
                robust_correct=robust_correct, robust_accuracy=100 * robust_correct / total,
                adv_loss=adv_loss / total, successful_attacks=clean_correct - robust_correct,
                attack_success_rate=(100 * (clean_correct - robust_correct) / clean_correct
                                     if clean_correct else None))


def evaluate_clean(model, test_loader, device):
    metrics = evaluate_metrics(model, test_loader, device)
    return metrics['clean_accuracy'], metrics['clean_loss']


def evaluate_adversarial(model, test_loader, attack, device):
    metrics = evaluate_metrics(model, test_loader, device, attack)
    return metrics['robust_accuracy'], metrics['adv_loss']


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda stream: stream.write(
        (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()))


def atomic_write(path, writer):
    """Replace an artifact only after the complete new file has reached disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state(loaders, device):
    numpy_state = np.random.get_state()
    state = dict(python=random.getstate(), torch=torch.get_rng_state(),
                 numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                 loaders=[loader.generator.get_state() if loader.generator else None
                          for loader in loaders])
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    if device.type == 'mps':
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state, loaders, device):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state['torch'].cpu())
    for loader, generator_state in zip(loaders, state['loaders']):
        if generator_state is not None:
            loader.generator.set_state(generator_state.cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    if device.type == 'mps':
        torch.mps.set_rng_state(state['mps'].cpu())


def save_checkpoint(model, optimizer, epoch, accuracy, filepath, *, config=None,
                    scheduler=None, metric='clean_accuracy', training_state=None):
    checkpoint = dict(format=CHECKPOINT_FORMAT, epoch=epoch, accuracy=accuracy,
                      selection_metric=metric, model_state_dict=model.state_dict(),
                      optimizer_state_dict=optimizer.state_dict(), config=config or {},
                      scheduler_state_dict=scheduler.state_dict() if scheduler else None)
    if training_state is not None:
        checkpoint['training_state'] = training_state
    atomic_write(filepath, lambda stream: torch.save(checkpoint, stream))


def read_checkpoint(model, filepath, device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    if checkpoint.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Legacy checkpoint: preprocessing is unverified. Retrain with the '
                         'pixel-space pipeline; see docs/validation.md.')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def load_checkpoint(model, optimizer, filepath, device):
    checkpoint = read_checkpoint(model, filepath, device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['accuracy']
