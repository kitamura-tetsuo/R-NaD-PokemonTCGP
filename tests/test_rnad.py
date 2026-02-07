import torch
import torch.nn as nn
from src.models.transformer import UnifiedModel
from src.training.rnad import RNaDLearner, TrajectoryBuffer

def test_rnad_learner_step():
    # Setup
    obs_dim = 10
    action_dim = 5
    hidden_dim = 8
    num_blocks = 2

    model = UnifiedModel(obs_dim, action_dim, hidden_dim, num_blocks)
    fixed_model = UnifiedModel(obs_dim, action_dim, hidden_dim, num_blocks)
    fixed_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    learner = RNaDLearner(
        model=model,
        fixed_point_model=fixed_model,
        optimizer=optimizer,
        alpha_rnad=0.1,
    )

    # Create dummy batch
    T = 4
    B = 2

    states = torch.randn(T, B, obs_dim)
    masks = torch.ones(T, B, action_dim)
    # Mask out some actions
    masks[0, 0, 0] = 0
    masks[1, 1, 4] = 0

    # Make sure sampled actions are valid to avoid gathering -inf
    actions = torch.randint(1, action_dim - 1, (T, B))

    rewards = torch.randn(T, B)
    log_probs = torch.randn(T, B) # Dummy log probs

    batch = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'log_probs': log_probs,
        'masks': masks,
        'seq_mask': torch.ones(T, B, dtype=torch.bool) # All valid
    }

    # Test compute_loss
    policy_loss, value_loss = learner.compute_loss(batch)
    assert isinstance(policy_loss, torch.Tensor)
    assert isinstance(value_loss, torch.Tensor)

    # Test update
    initial_params = [p.clone() for p in model.parameters()]
    stats = learner.update(batch)

    assert 'policy_loss' in stats
    assert 'value_loss' in stats
    assert 'total_loss' in stats

    # Check if params changed
    has_changed = False
    for p, initial_p in zip(model.parameters(), initial_params):
        if not torch.allclose(p, initial_p):
            has_changed = True
            break
    assert has_changed, "Model parameters should change after update"

def test_action_masking():
    obs_dim = 10
    action_dim = 5
    hidden_dim = 8
    num_blocks = 2

    model = UnifiedModel(obs_dim, action_dim, hidden_dim, num_blocks)

    states = torch.randn(1, 1, obs_dim)
    masks = torch.ones(1, 1, action_dim)
    masks[0, 0, 0] = 0 # Invalid action 0

    # Test with boolean mask
    bool_masks = masks.bool()

    logits, _ = model(states, action_mask=bool_masks)

    # Check if logits for invalid action are very small
    assert logits[0, 0, 0] < -1e8
    # Check valid actions are normal
    assert logits[0, 0, 1] > -1e8

    # Test with 0/1 mask (float or long)
    logits_float, _ = model(states, action_mask=masks)
    assert logits_float[0, 0, 0] < -1e8
    assert logits_float[0, 0, 1] > -1e8
