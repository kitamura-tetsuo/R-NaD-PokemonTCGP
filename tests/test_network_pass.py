import jax
import jax.numpy as jnp
import haiku as hk
import pytest
from src.models import DeckGymNet

def test_deckgym_net_forward():
    batch_size = 4
    obs_dim = 100 # Dummy
    num_actions = 10 # Dummy

    def forward(x):
        net = DeckGymNet(num_actions, hidden_size=64, num_blocks=2)
        return net(x)

    net = hk.transform(forward)
    key = jax.random.PRNGKey(42)

    x = jnp.zeros((batch_size, obs_dim))

    # Init
    params = net.init(key, x)

    # Apply
    logits, values = net.apply(params, key, x)

    assert logits.shape == (batch_size, num_actions)
    assert values.shape == (batch_size, 1)

def test_deckgym_net_jit():
    batch_size = 4
    obs_dim = 100
    num_actions = 10

    def forward(x):
        net = DeckGymNet(num_actions, hidden_size=64, num_blocks=2)
        return net(x)

    net = hk.transform(forward)
    key = jax.random.PRNGKey(42)
    x = jnp.zeros((batch_size, obs_dim))
    params = net.init(key, x)

    apply_jit = jax.jit(net.apply)
    logits, values = apply_jit(params, key, x)

    assert logits.shape == (batch_size, num_actions)
    assert values.shape == (batch_size, 1)

if __name__ == "__main__":
    test_deckgym_net_forward()
    test_deckgym_net_jit()
