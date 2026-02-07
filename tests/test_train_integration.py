import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.actor import SelfPlayWorker, MockDeckGym
from src.training.rnad import RNaDLearner, TrajectoryBuffer
from src.models.transformer import UnifiedModel

class TestTrainIntegration(unittest.TestCase):
    def test_single_iteration(self):
        # Configuration
        obs_dim = 30574
        action_dim = 25000
        hidden_dim = 32 # Small for speed
        num_blocks = 1
        learning_rate = 1e-4
        episodes_per_iteration = 2 # Small number
        batch_size = 2
        device = torch.device("cpu")

        # Initialize models
        model = UnifiedModel(obs_dim, action_dim, hidden_dim, num_blocks).to(device)
        fixed_point_model = UnifiedModel(obs_dim, action_dim, hidden_dim, num_blocks).to(device)
        fixed_point_model.load_state_dict(model.state_dict())

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize learner
        learner = RNaDLearner(
            model=model,
            fixed_point_model=fixed_point_model,
            optimizer=optimizer,
            device=device
        )

        # Initialize worker (will use MockDeckGym)
        worker = SelfPlayWorker(device=device)
        # Ensure it uses MockDeckGym
        self.assertIsInstance(worker.env, MockDeckGym)

        # Generate episodes
        buffer = TrajectoryBuffer(device=device)
        for _ in range(episodes_per_iteration):
            episode_data = worker.run_episode(model)
            buffer.add_episode(episode_data)

        self.assertEqual(len(buffer), episodes_per_iteration)

        # Sample batch
        batch = buffer.sample(batch_size=episodes_per_iteration)

        # Check batch keys
        self.assertIn('states', batch)
        self.assertIn('seq_mask', batch)

        # Update model
        metrics = learner.update(batch)
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('total_loss', metrics)

        # Update fixed point
        learner.update_fixed_point()

if __name__ == '__main__':
    unittest.main()
