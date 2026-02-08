import unittest
from unittest.mock import MagicMock, patch
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import os
import shutil
from src.rnad import RNaDLearner, RNaDConfig

class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "test_checkpoints"
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir)

    def tearDown(self):
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

    @patch('src.rnad.pyspiel')
    def test_save_and_load_checkpoint(self, mock_pyspiel):
        # Mock the game
        mock_game = MagicMock()
        mock_game.num_distinct_actions.return_value = 10
        mock_game.observation_tensor_shape.return_value = (20,)
        mock_pyspiel.load_game.return_value = mock_game

        config = RNaDConfig(batch_size=2, max_steps=10, hidden_size=32, num_blocks=1)
        learner = RNaDLearner("deckgym_ptcgp", config)

        # Init
        learner.init(jax.random.PRNGKey(0))

        # Save initial params
        initial_params = learner.params
        step = 5
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pkl")

        # Save checkpoint
        learner.save_checkpoint(ckpt_path, step)

        # Verify file exists
        self.assertTrue(os.path.exists(ckpt_path))

        # Modify params
        learner.params = jax.tree_util.tree_map(lambda x: x + 1.0, learner.params)

        # Verify params are different
        # We check one leaf
        k = list(initial_params.keys())[0]
        # initial_params[k] is a dict of arrays
        k2 = list(initial_params[k].keys())[0]
        self.assertFalse(jnp.array_equal(learner.params[k][k2], initial_params[k][k2]))

        # Load checkpoint
        loaded_step = learner.load_checkpoint(ckpt_path)

        # Verify step
        self.assertEqual(loaded_step, step)

        # Verify params are restored
        # Checking tree equality
        def check_eq(a, b):
            return jnp.array_equal(a, b)

        equality = jax.tree_util.tree_map(check_eq, learner.params, initial_params)
        # Flatten and check all True
        all_equal = all(jax.tree_util.tree_leaves(equality))
        self.assertTrue(all_equal)

if __name__ == '__main__':
    unittest.main()
