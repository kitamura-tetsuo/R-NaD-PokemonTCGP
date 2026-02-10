import jax
import jax.numpy as jnp
import numpy as np
import logging
import os
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.rnad import RNaDLearner, RNaDConfig

def test_unroll():
    # Use existing deck file
    deck_path = "deckgym-core/example_decks/mewtwoex.txt"
    if not os.path.exists(deck_path):
        print(f"Deck file not found at {deck_path}, creating dummy for test")
        # Create a dummy deck file if needed, or fail
        # But user environment should have it.
    
    config = RNaDConfig(
        batch_size=4,
        unroll_length=20, # Short for testing
        deck_id_1=deck_path,
        deck_id_2=deck_path,
        learning_rate=1e-3
    )
    
    print(f"Initializing learner with unroll_length={config.unroll_length}")
    learner = RNaDLearner("deckgym_ptcgp", config)
    
    # Verify game length matches configuration
    game_length = learner.game.max_game_length()
    print(f"Game max length: {game_length}")
    assert game_length == config.unroll_length, f"Expected game length {config.unroll_length}, got {game_length}"
    
    learner.init(jax.random.PRNGKey(0))
    
    print("Generating trajectories...")
    batch = learner.generate_trajectories(jax.random.PRNGKey(1))
    
    # Check shapes
    obs = batch['obs']
    print(f"Obs shape: {obs.shape}")
    assert obs.shape[0] == 20, f"Expected time dimension 20, got {obs.shape[0]}"
    assert obs.shape[1] == 4, f"Expected batch dimension 4, got {obs.shape[1]}"
    
    # Check bootstrap
    bootstrap = batch['bootstrap_value']
    print(f"Bootstrap shape: {bootstrap.shape}")
    assert bootstrap.shape == (4,), f"Expected bootstrap shape (4,), got {bootstrap.shape}"
    
    print(f"Bootstrap values: {bootstrap}")
    
    # Check update
    print("Running update step...")
    metrics = learner.update(batch, step=0)
    print("Metrics keys:", metrics.keys())
    
    assert 'total_loss' in metrics
    print("Test passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        test_unroll()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
