"""Regression checks for attack geometry, metrics, and training state."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attacks import PGDAttack, FGSMAttack
from model_arch import CNN
from training import train_epoch
from utils import atomic_write, evaluate_metrics, read_checkpoint, save_checkpoint


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(12, 16), nn.BatchNorm1d(16),
                                   nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 3))
        self.x = torch.rand(5, 3, 2, 2)
        self.y = torch.tensor([0, 1, 2, 0, 1])

    def test_attack_bounds_modes_and_parameter_gradients(self):
        self.model.train()
        self.model[2].eval()
        modes = [m.training for m in self.model.modules()]
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        for attack in (PGDAttack(self.model, num_iter=3, restarts=2), FGSMAttack(self.model)):
            with torch.no_grad():
                adversarial = attack(self.x, self.y)
            self.assertLessEqual((adversarial - self.x).abs().max().item(), 4/255 + 1e-7)
            self.assertGreaterEqual(adversarial.min().item(), 0)
            self.assertLessEqual(adversarial.max().item(), 1)
            self.assertFalse(adversarial.requires_grad)
            self.assertEqual(modes, [m.training for m in self.model.modules()])
            for parameter in self.model.parameters():
                self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))

    def test_zero_epsilon_is_identity(self):
        for attack in (PGDAttack(self.model, epsilon=0), FGSMAttack(self.model, epsilon=0)):
            torch.testing.assert_close(attack(self.x, self.y), self.x, rtol=0, atol=0)

    def test_gradient_direction_increases_linear_loss(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3)).eval()
        loss = nn.CrossEntropyLoss()
        for attack in (FGSMAttack(model), PGDAttack(model, num_iter=4)):
            adversarial = attack(self.x, self.y)
            self.assertGreater(loss(model(adversarial), self.y).item(),
                               loss(model(self.x), self.y).item())

    def test_fgsm_matches_the_single_step_definition(self):
        class CurvedClassifier(nn.Module):
            def forward(self, images):
                value = torch.sin(10 * images.flatten(1).mean(1))
                return torch.stack((value, -value), dim=1)
        model = CurvedClassifier()
        x = torch.full((2, 3, 2, 2), 0.4)
        y = torch.zeros(2, dtype=torch.long)
        candidate = x.clone().requires_grad_(True)
        gradient, = torch.autograd.grad(nn.CrossEntropyLoss()(model(candidate), y), candidate)
        expected = (x + 0.4 * gradient.sign()).clamp(0, 1)
        actual = FGSMAttack(model, epsilon=0.4)(x, y)
        torch.testing.assert_close(actual, expected)

    def test_pgd_matches_an_independent_reference(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.Tanh(), nn.Linear(8, 3))
        expected = self.x.clone()
        best = expected.clone()
        best_losses = nn.functional.cross_entropy(model(best), self.y, reduction='none').detach()
        for _ in range(4):
            expected.requires_grad_(True)
            loss = nn.functional.cross_entropy(model(expected), self.y, reduction='sum')
            gradient, = torch.autograd.grad(loss, expected)
            expected = expected.detach() + (2/255) * gradient.sign()
            expected = torch.maximum(torch.minimum(expected, self.x + 4/255), self.x - 4/255)
            expected = expected.clamp(0, 1)
            losses = nn.functional.cross_entropy(model(expected), self.y, reduction='none').detach()
            replace = losses > best_losses
            best[replace], best_losses[replace] = expected[replace], losses[replace]
        actual = PGDAttack(model, num_iter=4, selection='loss')(self.x, self.y, random_start=False)
        torch.testing.assert_close(actual, best, atol=1e-7, rtol=0)

    def test_atomic_write_preserves_previous_artifact_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'artifact'
            path.write_bytes(b'complete')
            def fail(stream):
                stream.write(b'partial')
                raise RuntimeError('simulated interruption')
            with self.assertRaises(RuntimeError):
                atomic_write(path, fail)
            self.assertEqual(path.read_bytes(), b'complete')
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_invalid_inputs(self):
        for options in ({'epsilon': -1}, {'num_iter': 0}, {'alpha': float('nan')}, {'restarts': 0}):
            with self.assertRaises(ValueError):
                PGDAttack(self.model, **options)
        with self.assertRaises(ValueError):
            PGDAttack(self.model)(self.x - 2, self.y)

    def test_metrics_weight_last_batch_and_preserve_mode(self):
        self.model.eval()
        loader = DataLoader(TensorDataset(self.x, self.y), batch_size=3)
        result = evaluate_metrics(self.model, loader, 'cpu', PGDAttack(self.model, num_iter=2))
        expected = nn.CrossEntropyLoss()(self.model(self.x), self.y).item()
        self.assertAlmostEqual(expected, result['clean_loss'], places=6)
        self.assertLessEqual(result['robust_accuracy'], result['clean_accuracy'])
        self.assertEqual(result['successful_attacks'], result['clean_correct'] - result['robust_correct'])
        self.assertFalse(self.model.training)

    def test_asr_without_clean_correct_samples(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
        for p in model.parameters():
            nn.init.zeros_(p)
        loader = DataLoader(TensorDataset(self.x, torch.ones(5, dtype=torch.long)), batch_size=5)
        self.assertIsNone(evaluate_metrics(model, loader, 'cpu')['attack_success_rate'])

    def test_clean_and_robust_training_update_weights(self):
        loader = DataLoader(TensorDataset(self.x[:4], self.y[:4]), batch_size=2)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        for ratio in (0, 0.5, 1):
            before = self.model[1].weight.detach().clone()
            loss, acc = train_epoch(self.model, loader, optimizer, nn.CrossEntropyLoss(),
                                    'cpu', 1, PGDAttack(self.model, num_iter=2), ratio)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertFalse(torch.equal(before, self.model[1].weight))
            self.assertTrue(self.model.training)

    def test_architecture_and_checkpoint_roundtrip(self):
        model = CNN().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9356554)
        x = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            expected = model(x)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            save_checkpoint(model, optimizer, 1, 25.0, path)
            restored = CNN().eval()
            read_checkpoint(restored, path, 'cpu')
            with torch.no_grad():
                torch.testing.assert_close(restored(x), expected)
            torch.save({'model_state_dict': model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'Legacy checkpoint'):
                read_checkpoint(restored, path, 'cpu')


if __name__ == '__main__':
    unittest.main()
