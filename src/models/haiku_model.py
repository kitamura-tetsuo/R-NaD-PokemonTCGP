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
