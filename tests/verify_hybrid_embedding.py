import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.models import CardTransformerNet

def test_card_transformer_hybrid():
    batch_size = 2
    obs_dim = 327 # From CardTransformerNet logic
    num_actions = 50
    
    # Dummy embedding matrix
    num_cards = 1000
    feature_dim = 26
    embedding_matrix = np.random.randn(num_cards, feature_dim).astype(np.float32)
    
    def forward(x):
        net = CardTransformerNet(
            num_actions=num_actions,
            embedding_matrix=embedding_matrix,
            hidden_size=64,
            num_blocks=2,
            num_heads=4
        )
        return net(x)
    
    net = hk.transform(forward)
    key = jax.random.PRNGKey(42)
    
    # Create dummy observation
    x = jnp.zeros((batch_size, obs_dim))
    
    # Set some card IDs to valid indices
    # Board card IDs are at 39 + i*32 + 11
    # Hand IDs are at 295:305
    # Discard IDs are at 307:317
    # Opp Discard IDs are at 317:327
    
    # Board IDs
    for i in range(8):
        x = x.at[:, 39 + i*32 + 11].set(i + 1)
        
    # Hand IDs
    x = x.at[:, 295:305].set(jnp.arange(10) + 10)
    
    # Init
    params = net.init(key, x)
    
    # Check if residual_path is in params and it is zero
    # residual_path is inside HybridEmbedding, which is used multiple times.
    # Haiku names them with suffixes if not careful, but here they are under different names like "emb_board/static_path", "emb_board/residual_path", etc.
    
    found_residual = False
    for module_name, param_dict in params.items():
        if "residual_path" in module_name:
            found_residual = True
            for param_name, value in param_dict.items():
                print(f"Checking {module_name}/{param_name}: mean={jnp.mean(value)}")
                assert jnp.all(value == 0.0), f"Residual path {module_name}/{param_name} should be initialized to zero"
    
    assert found_residual, "Could not find residual_path in params"
    
    # Apply
    logits, values = net.apply(params, key, x)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Values shape: {values.shape}")
    
    assert logits.shape == (batch_size, num_actions)
    assert values.shape == (batch_size, 1)
    
    print("Verification successful!")

if __name__ == "__main__":
    test_card_transformer_hybrid()
