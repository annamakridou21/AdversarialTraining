"""Check that the planned experiment controls training budgets and test access."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from scripts.run_experiment import ROOT, build_stages, experiment_lock


class ExperimentTests(unittest.TestCase):
    def test_control_and_robust_branches_share_initial_weights_and_budget(self):
        protocol = json.loads((ROOT / 'experiments/full.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            stages = dict(build_stages(protocol, 42, Path(directory), Path('data'), 'cpu', 0))
        clean, robust = stages['clean_control'], stages['adversarial']
        for argument in ('--pretrain-path', '--epochs', '--lr', '--batch-size', '--seed'):
            self.assertEqual(clean[clean.index(argument) + 1], robust[robust.index(argument) + 1])
        for name in ('pretrain', 'clean_control', 'adversarial'):
            self.assertIn('--skip-test', stages[name])
        self.assertNotIn('--train-limit', clean)
        self.assertNotIn('--eval-limit', stages['evaluate_clean_control'])

    def test_lock_blocks_concurrent_runner_and_releases_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'interrupted'):
                with experiment_lock(directory):
                    with self.assertRaisesRegex(RuntimeError, 'already running'):
                        with experiment_lock(directory):
                            self.fail('Concurrent lock was acquired')
                    raise ValueError('interrupted')
            with experiment_lock(directory):
                pass

    def test_windows_lock_reserves_and_unlocks_a_byte(self):
        locking = Mock()
        module = SimpleNamespace(locking=locking, LK_NBLCK=2, LK_UNLCK=0)
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.run_experiment.sys.platform', 'win32'), \
                 patch.dict('sys.modules', {'msvcrt': module}):
                with experiment_lock(directory):
                    self.assertEqual((Path(directory) / '.run.lock').stat().st_size, 1)
            self.assertEqual(locking.call_count, 2)
            self.assertEqual(locking.call_args_list[0].args[1:], (2, 1))
            self.assertEqual(locking.call_args_list[1].args[1:], (0, 1))

    def test_runner_resumes_latest_without_reapplying_pretraining(self):
        protocol = json.loads((ROOT / 'experiments/pilot.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            latest = output / 'adversarial/models/robust_latest.pth'
            latest.parent.mkdir(parents=True)
            latest.touch()
            stages = dict(build_stages(protocol, 42, output, Path('data'), 'cpu', 0))
        self.assertIn('--resume', stages['adversarial'])
        self.assertNotIn('--pretrain-path', stages['adversarial'])


if __name__ == '__main__':
    unittest.main()
