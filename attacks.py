"""Untargeted L-infinity attacks on images in pixel space [0, 1]."""

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F


@contextmanager
def evaluating(model):
    """Freeze dropout and batch statistics without changing the caller's mode."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


class PGDAttack:
    """Projected gradient ascent with a random start and per-example selection.

    Training uses selection='loss' to approximate the inner maximization.
    Evaluation uses selection='success' to retain misclassified candidates,
    breaking ties by cross-entropy. Both include the clean input as a candidate.
    """

    def __init__(self, model, epsilon=4/255, alpha=2/255, num_iter=10,
                 device='cpu', restarts=1, selection='success'):
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError('epsilon must be finite and non-negative')
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError('alpha must be finite and non-negative')
        if not isinstance(num_iter, int) or num_iter < 1:
            raise ValueError('num_iter must be a positive integer')
        if not isinstance(restarts, int) or restarts < 1:
            raise ValueError('restarts must be a positive integer')
        if selection not in ('loss', 'success'):
            raise ValueError("selection must be 'loss' or 'success'")
        self.selection = selection
        self.model, self.device = model, device
        self.epsilon, self.alpha = epsilon, alpha
        self.num_iter, self.restarts = num_iter, restarts

    def project(self, x_adv, x_natural):
        delta = (x_adv - x_natural).clamp(-self.epsilon, self.epsilon)
        return (x_natural + delta).clamp(0, 1)

    def generate(self, x, y, random_start=True):
        x, y = x.detach().to(self.device), y.to(self.device)
        if x.numel() == 0 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError('Attacks require finite pixel values in [0, 1]')
        if self.epsilon == 0:
            return x.clone()
        with evaluating(self.model), torch.enable_grad():
            with torch.no_grad():
                logits = self.model(x)
                best_loss = F.cross_entropy(logits, y, reduction='none')
                best_wrong = logits.argmax(1).ne(y)
                best = x.clone()
            for _ in range(self.restarts):
                candidate = x.clone()
                if random_start:
                    candidate = self.project(
                        x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon), x)
                for step in range(self.num_iter + 1):
                    candidate = candidate.detach().requires_grad_(True)
                    logits = self.model(candidate)
                    losses = F.cross_entropy(logits, y, reduction='none')
                    with torch.no_grad():
                        wrong = logits.argmax(1).ne(y)
                        replace = losses > best_loss
                        if self.selection == 'success':
                            replace = (wrong & ~best_wrong) | (
                                (wrong == best_wrong) & replace)
                        best[replace] = candidate[replace]
                        best_loss[replace] = losses[replace]
                        best_wrong[replace] = wrong[replace]
                    if step < self.num_iter:
                        # Input gradients leave parameter gradients untouched.
                        gradient, = torch.autograd.grad(losses.sum(), candidate)
                        candidate = self.project(
                            candidate.detach() + self.alpha * gradient.sign(), x)
        return best.detach()

    def __call__(self, x, y, random_start=True):
        return self.generate(x, y, random_start)


class FGSMAttack(PGDAttack):
    """One gradient-sign step from the clean image, with no random start."""

    def __init__(self, model, epsilon=4/255, device='cpu'):
        super().__init__(model, epsilon, epsilon, 1, device)

    def generate(self, x, y, random_start=False):
        x, y = x.detach().to(self.device), y.to(self.device)
        if x.numel() == 0 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError('Attacks require finite pixel values in [0, 1]')
        if self.epsilon == 0:
            return x.clone()
        with evaluating(self.model), torch.enable_grad():
            candidate = x.clone().requires_grad_(True)
            loss = F.cross_entropy(self.model(candidate), y)
            gradient, = torch.autograd.grad(loss, candidate)
        return (x + self.epsilon * gradient.sign()).clamp(0, 1).detach()
