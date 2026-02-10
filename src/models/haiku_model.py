import haiku as hk
import jax
import jax.numpy as jnp

class ResidualBlock(hk.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def __call__(self, x):
        input_val = x
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)
        return input_val + x

class DeckGymNet(hk.Module):
    def __init__(self, num_actions, hidden_size=256, num_blocks=4):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

    def __call__(self, x):
        # Flatten input if needed, though usually expected (B, obs_dim)
        if x.ndim > 2:
            x = jnp.reshape(x, (x.shape[0], -1))

        # Torso
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)

        for _ in range(self.num_blocks):
            x = ResidualBlock(self.hidden_size)(x)

        # Heads
        policy_logits = hk.Linear(self.num_actions)(x)
        value = hk.Linear(1)(x)
        return policy_logits, value


class TransformerBlock(hk.Module):
    def __init__(self, num_heads, key_size, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def __call__(self, x, is_training=False):
        # x: (B, SeqLen, EmbedDim)
        
        # Self-Attention
        attn_out = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init=hk.initializers.VarianceScaling(2.0),
        )(x, x, x)
        
        # Add & Norm
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + attn_out)
        
        # MLP
        mlp_out = hk.nets.MLP(
            [self.hidden_size * 2, self.hidden_size],
            activation=jax.nn.gelu
        )(x)
        
        # Add & Norm
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + mlp_out)
        
        return x

class TransformerNet(hk.Module):
    def __init__(self, num_actions, hidden_size=64, num_blocks=2, num_heads=4, seq_len=16):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size # Embed Dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.seq_len = seq_len

    def __call__(self, x, is_training=False):
        # x: (B, ObsDim)
        if x.ndim > 2:
            x = jnp.reshape(x, (x.shape[0], -1))
            
        batch_size = x.shape[0]
        obs_dim = x.shape[1]
        
        # Feature Projection: (B, ObsDim) -> (B, SeqLen * EmbedDim)
        # We project the flat observation into a latent sequence
        projection_size = self.seq_len * self.hidden_size
        x = hk.Linear(projection_size)(x)
        
        # Reshape to sequence: (B, SeqLen, EmbedDim)
        x = jnp.reshape(x, (batch_size, self.seq_len, self.hidden_size))
        
        # Add learned position embeddings (optional but good for latent sequence)
        # For simplicity in this "Latent Transformer" on flat data, we might skip or add them.
        # Let's add them.
        pos_emb = hk.get_parameter("pos_emb", [self.seq_len, self.hidden_size], init=hk.initializers.TruncatedNormal())
        x = x + pos_emb
        
        # Transformer Blocks
        for _ in range(self.num_blocks):
            x = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.hidden_size // self.num_heads,
                hidden_size=self.hidden_size
            )(x, is_training)
            
        # Global Pooling (Mean) -> (B, EmbedDim)
        x = jnp.mean(x, axis=1)
        
        # Heads
        # Policy
        policy_logits = hk.Linear(self.num_actions)(x)
        
        # Value
        value = hk.Linear(1)(x)
        
        return policy_logits, value
