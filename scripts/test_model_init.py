import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import os
from src.rnad import RNaDLearner, RNaDConfig

def test_model_init():
    config = RNaDConfig(
        model_type="transformer",
        transformer_embed_dim=64,
        transformer_layers=2,
        transformer_heads=4
    )
    
    # Register game by importing
    import deckgym_openspiel
    # Needs to match the actual game name (short_name in game.py)
    game_name = "deckgym_ptcgp"
    
    try:
        learner = RNaDLearner(game_name, config)
        print(f"Learner initialized. Observation shape: {learner.obs_shape}")
        
        key = jax.random.PRNGKey(42)
        learner.init(key)
        print("Model initialized successfully.")
        
        # Test forward pass
        dummy_obs = jnp.zeros((1, *learner.obs_shape))
        logits, values = learner.network.apply(learner.params, key, dummy_obs)
        print(f"Forward pass successful. Logits shape: {logits.shape}, Values shape: {values.shape}")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Make sure we are in the right directory
    test_model_init()
