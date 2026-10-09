"""Offline integration tests; synthetic inputs are not benchmark results."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FakeData
from torchvision import transforms

import training
import utils
from scripts import _common, analyze_robustness, create_comparison_plots, create_summary_report
from scripts.plot_training import plot_training_curves
from scripts.visualize_attacks import visualize_adversarial_examples
from model_arch import CNN


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)

    def test_split_is_disjoint_seeded_and_pixel_space(self):
        def fake_cifar(root, train, download, transform):
            return FakeData(size=24 if train else 8, image_size=(3, 32, 32),
                            num_classes=10, transform=transform)
        with patch.object(utils.datasets, 'CIFAR10', side_effect=fake_cifar):
            loaders = utils.get_cifar10_loaders(num_workers=0, validation_size=6)
            again = utils.get_cifar10_loaders(num_workers=0, validation_size=6)
        train, validation, test = loaders
        self.assertEqual(len(train.dataset), 18)
        self.assertEqual(len(validation.dataset), 6)
        self.assertEqual(len(test.dataset), 8)
        self.assertFalse(set(train.dataset.indices) & set(validation.dataset.indices))
        self.assertEqual(train.dataset.indices, again[0].dataset.indices)
        image, _ = validation.dataset[0]
        self.assertGreaterEqual(image.min().item(), 0)
        self.assertLessEqual(image.max().item(), 1)
        self.assertIsInstance(validation.dataset.dataset.transform, transforms.ToTensor)

    def test_epoch_resume_matches_uninterrupted_training(self):
        class TinyClassifier(torch.nn.Module):
            def __init__(self, dropout_rate=0.5):
                super().__init__()
                self.layers = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 8),
                    torch.nn.ReLU(), torch.nn.Dropout(dropout_rate), torch.nn.Linear(8, 3))
            def forward(self, images):
                return self.layers(images)
        images, labels = torch.rand(8, 3, 2, 2), torch.arange(8) % 3
        def loaders(*args, **kwargs):
            return tuple(DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=(i == 0),
                         generator=torch.Generator().manual_seed(42 + i)) for i in range(3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(name, epochs, resume=False):
                cli = ['--epochs', str(epochs), '--lr', '0.001', '--device', 'cpu',
                       '--num-workers', '0', '--batch-size', '2', '--eval-steps', '1',
                       '--lr-milestones', '1', '--skip-test', '--output-dir', str(root / name)]
                if resume:
                    cli += ['--resume', str(root / name / 'models/standard_latest.pth')]
                with patch.object(training, 'CNN', TinyClassifier), \
                     patch.object(training, 'get_cifar10_loaders', side_effect=loaders):
                    training.run_training(training.build_parser().parse_args(cli))
            run('complete', 2)
            run('interrupted', 1)
            # Simulate a best-file write interrupted after the last complete epoch.
            (root / 'interrupted/models/standard_best.pth').write_bytes(b'incomplete')
            run('interrupted', 2, resume=True)
            reference = torch.load(root / 'complete/models/standard_latest.pth', weights_only=True)
            resumed = torch.load(root / 'interrupted/models/standard_latest.pth', weights_only=True)
            for key, value in reference['model_state_dict'].items():
                torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)
            self.assertEqual(reference['scheduler_state_dict'], resumed['scheduler_state_dict'])
            self.assertEqual(reference['training_state']['history']['train_loss'],
                             resumed['training_state']['history']['train_loss'])
            with self.assertRaises(FileExistsError):
                run('interrupted', 2)

    def test_train_finetune_analyze_and_plot(self):
        images = torch.rand(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for robust in (False, True):
                cli = ['--epochs', '1', '--lr', '0.001', '--output-dir', directory, '--num-workers', '0',
                       '--device', 'cpu', '--eval-steps', '1', '--batch-size', '2']
                if robust:
                    cli += ['--train-steps', '1', '--pretrain-path', str(root / 'models/standard_best.pth')]
                args = training.build_parser(robust).parse_args(cli)
                with patch.object(training, 'get_cifar10_loaders', return_value=(loader, loader, loader)):
                    training.run_training(args, robust)
                name = 'robust' if robust else 'standard'
                result = json.loads((root / f'{name}_test.json').read_text())
                self.assertEqual(result['metrics']['total_samples'], 4)
                self.assertEqual(result['selected_epoch'], 1)
                history = root / f'{name}_history.json'
                figure = plot_training_curves(history, str(root / f'{name}_training.png'))
                plt.close(figure)
                args = argparse.Namespace(checkpoint=str(root / f'models/{name}_best.pth'),
                    batch_size=2, num_workers=0, data_dir=directory, seed=42, eval_limit=4,
                    device='cpu', steps=1, restarts=1, alpha=2/255, epsilons=[0, 4/255],
                    save_results=str(root / f'{name}.json'))
                with patch.object(_common, 'get_cifar10_test_loader', return_value=loader):
                    analyze_robustness.main(args)
            comparison = argparse.Namespace(standard=str(root / 'standard.json'),
                robust=str(root / 'robust.json'), output=str(root / 'comparison.png'))
            create_comparison_plots.main(comparison)
            comparison.output = str(root / 'summary.md')
            create_summary_report.main(comparison)
            self.assertIn('Samples', Path(comparison.output).read_text())
            for epsilon in (0, 4/255):
                fig = visualize_adversarial_examples(CNN().eval(), loader, 'cpu', num_images=1,
                                                     epsilon=epsilon, steps=1, restarts=1)
                fig.savefig(root / f'images_{epsilon}.png')
                plt.close(fig)
            changed = json.loads((root / 'robust.json').read_text())
            changed['config']['steps'] = 100
            (root / 'robust.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'settings'):
                create_comparison_plots.load_comparison(root / 'standard.json', root / 'robust.json')


if __name__ == '__main__':
    unittest.main()
