"""Regression checks for EMA validation, dropout and training randomness."""
import random
import unittest

import numpy as np
import torch

from diffusion_policy.common.policy_evaluation import evaluation_context, compute_validation_loss
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class EvaluationPolicy(BaseImagePolicy):
    def __init__(self, random_loss=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.dropout = torch.nn.Dropout(.9)
        self.random_loss = random_loss
        self.calls = []

    def compute_loss(self, batch):
        self.calls.append((self.training, self.dropout.training, torch.is_grad_enabled()))
        loss = self.dropout(batch['action']).mean() + self.weight
        if self.random_loss:
            loss = loss + torch.rand(()) + random.random() + float(np.random.random())
        return loss


def batch(size, value):
    return {'obs': {'state': torch.zeros(size, 2, 1)},
            'action': torch.full((size, 16, 52), float(value))}


class PolicyEvaluationTests(unittest.TestCase):
    def test_short_final_batch_is_weighted_by_sample_count(self):
        policy = EvaluationPolicy()
        result = compute_validation_loss(policy, [batch(4, 1), batch(1, 9)], 'cpu', seed=42)
        self.assertAlmostEqual(result, 2.6)
        self.assertEqual(policy.calls, [(False, False, False)] * 2)
        self.assertTrue(policy.training)
        self.assertTrue(policy.dropout.training)
        self.assertIsNone(policy.weight.grad)

    def test_fixed_seed_is_repeatable_and_does_not_change_training_rng(self):
        policy = EvaluationPolicy(random_loss=True)
        policy.dropout.eval()  # Preserve heterogeneous module modes too.
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        first = compute_validation_loss(policy, [batch(2, 1)], 'cpu', seed=42)
        second = compute_validation_loss(policy, [batch(2, 1)], 'cpu', seed=42)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), python_state)
        now = np.random.get_state()
        self.assertEqual(now[0], numpy_state[0])
        np.testing.assert_array_equal(now[1], numpy_state[1])
        self.assertEqual(now[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        self.assertTrue(policy.training)
        self.assertFalse(policy.dropout.training)
        third = compute_validation_loss(policy, [batch(2, 1)], 'cpu', seed=43)
        self.assertNotEqual(first, third)

    def test_context_restores_modes_rng_and_grad_mode_on_exception(self):
        policy = EvaluationPolicy()
        before = torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, 'stop evaluation'):
            with evaluation_context(policy, seed=42):
                self.assertFalse(policy.training)
                self.assertFalse(torch.is_grad_enabled())
                torch.rand(3)
                raise RuntimeError('stop evaluation')
        self.assertTrue(policy.training)
        self.assertTrue(policy.dropout.training)
        self.assertTrue(torch.is_grad_enabled())
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_empty_holdout_and_batch_limit(self):
        policy = EvaluationPolicy()
        self.assertIsNone(compute_validation_loss(policy, [], 'cpu', seed=42))
        result = compute_validation_loss(policy, [batch(4, 1), batch(1, 9)], 'cpu', max_batches=1, seed=42)
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
