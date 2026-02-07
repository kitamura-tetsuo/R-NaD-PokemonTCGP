import unittest
import torch
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.transformer import UnifiedModel

class TestModel(unittest.TestCase):
    def test_model_output_shapes(self):
        # Using the dimensions I found from the environment
        obs_dim = 30574
        action_dim = 25000
        batch_size = 4

        # Instantiate model with smaller hidden dim for speed
        model = UnifiedModel(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64, num_blocks=2)

        # Create dummy observation
        dummy_obs = torch.randn(batch_size, obs_dim)

        # Forward pass
        policy_logits, value = model(dummy_obs)

        # Check shapes
        self.assertEqual(policy_logits.shape, (batch_size, action_dim))
        self.assertEqual(value.shape, (batch_size, 1))

        # Check value range (should be between 0 and 1 because of sigmoid)
        self.assertTrue(torch.all(value >= 0))
        self.assertTrue(torch.all(value <= 1))

if __name__ == '__main__':
    unittest.main()
