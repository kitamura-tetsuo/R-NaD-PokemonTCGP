import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from src.models.transformer import UnifiedModel

def v_trace(
    log_mu_a: torch.Tensor,
    log_pi_a: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    V-trace calculation.

    Args:
        log_mu_a: Log probability of actions taken under behavior policy. Shape: (T, B)
        log_pi_a: Log probability of actions taken under target policy. Shape: (T, B)
        values: Value estimates for states x_0...x_{T-1}. Shape: (T, B)
        bootstrap_value: Value estimate for state x_T. Shape: (B,)
        rewards: Rewards. Shape: (T, B)
        discounts: Discount factors. Shape: (T, B)
        clip_rho_threshold: Clipping threshold for importance weights (rho_bar).
        clip_pg_rho_threshold: Clipping threshold for PG importance weights (c_bar).

    Returns:
        vs: V-trace value targets. Shape: (T, B)
        pg_advantages: V-trace advantages for policy gradient. Shape: (T, B)
    """
    T, B = log_mu_a.shape

    # Importance weights
    log_rhos = log_pi_a - log_mu_a
    rhos = torch.exp(log_rhos)

    # Clip importance weights
    clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)

    # Values
    values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)

    # Deltas: rho * (r + gamma * V_{t+1} - V_t)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # V-trace targets (vs)
    # Recursively: (vs_t - V_t) = delta_t + gamma * c_t * (vs_{t+1} - V_{t+1})

    acc = torch.zeros_like(bootstrap_value)
    vs_minus_v = []

    for t in range(T - 1, -1, -1):
        delta = deltas[t]
        gamma = discounts[t]
        c = clipped_rhos[t] # using rho_bar for c_bar (common practice)

        acc = delta + gamma * c * acc
        vs_minus_v.append(acc)

    vs_minus_v = torch.stack(vs_minus_v[::-1])
    vs = vs_minus_v + values

    # Advantages for policy gradient
    # DeepMind: pg_advantage = rho_pg * (r + gamma * vs_next - V)
    vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    return vs, pg_advantages


class TrajectoryBuffer:
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.episodes = []
        self.device = device

    def add_episode(self, episode_data: Dict[str, torch.Tensor]):
        """
        episode_data contains tensors for 'states', 'actions', 'rewards', 'masks', 'log_probs'.
        All should have shape (T, ...).
        """
        self.episodes.append(episode_data)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        import random
        indices = random.sample(range(len(self.episodes)), min(batch_size, len(self.episodes)))
        batch_episodes = [self.episodes[i] for i in indices]

        # Collate and pad
        keys = batch_episodes[0].keys()
        batch = {}

        # Find max length
        max_len = max(ep['states'].shape[0] for ep in batch_episodes)

        for key in keys:
            tensors = [ep[key] for ep in batch_episodes]
            # Pad sequences
            # Assuming shape (T, ...)
            # We pad with 0.
            padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=False, padding_value=0)
            batch[key] = padded.to(self.device)

        # Add a mask for valid steps (sequence lengths)
        lengths = torch.tensor([ep['states'].shape[0] for ep in batch_episodes], device=self.device)
        # Create a mask (T, B) where True is valid
        # padded shape is (T, B, ...)
        T, B = batch['states'].shape[:2]
        seq_mask = torch.arange(T, device=self.device).unsqueeze(1) < lengths.unsqueeze(0)
        batch['seq_mask'] = seq_mask

        return batch

    def __len__(self):
        return len(self.episodes)


class RNaDLearner:
    def __init__(
        self,
        model: UnifiedModel,
        fixed_point_model: UnifiedModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cpu'),
        alpha_rnad: float = 0.1,
        clip_rho_threshold: float = 1.0,
        clip_pg_rho_threshold: float = 1.0,
        discount_factor: float = 0.99,
    ):
        self.model = model
        self.fixed_point_model = fixed_point_model
        self.optimizer = optimizer
        self.device = device
        self.alpha_rnad = alpha_rnad
        self.clip_rho_threshold = clip_rho_threshold
        self.clip_pg_rho_threshold = clip_pg_rho_threshold
        self.discount_factor = discount_factor

        # Ensure fixed point model is in eval mode and doesn't require grad
        self.fixed_point_model.eval()
        for param in self.fixed_point_model.parameters():
            param.requires_grad = False

    def update_fixed_point(self):
        """Updates the fixed point model with the current model's weights."""
        self.fixed_point_model.load_state_dict(self.model.state_dict())
        self.fixed_point_model.eval()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        log_probs_mu = batch['log_probs'] # Log prob of action taken under behavior policy
        masks = batch['masks']
        seq_mask = batch['seq_mask']

        T, B = states.shape[:2]

        # Flatten for model processing
        obs_dim = states.shape[-1]
        flat_states = states.view(-1, obs_dim)
        # masks usually (T, B, action_dim)
        flat_masks = masks.view(-1, masks.shape[-1]) if masks is not None else None

        # Forward pass
        curr_logits, curr_values = self.model(flat_states, action_mask=flat_masks)
        with torch.no_grad():
            fixed_logits, _ = self.fixed_point_model(flat_states, action_mask=flat_masks)

        # Reshape back
        curr_logits = curr_logits.view(T, B, -1)
        curr_values = curr_values.view(T, B) # ValueHead returns (B, 1) -> squeeze(-1) if needed?
        # ValueHead: return torch.sigmoid(self.fc(x)) -> shape (B, 1)
        # So view(T, B, 1) or view(T*B, 1).
        if curr_values.dim() == 2 and curr_values.shape[1] == 1:
            curr_values = curr_values.squeeze(-1)
        curr_values = curr_values.view(T, B)

        fixed_logits = fixed_logits.view(T, B, -1)

        # Compute log probs
        log_probs_pi = F.log_softmax(curr_logits, dim=-1)
        log_probs_fixed = F.log_softmax(fixed_logits, dim=-1)

        # Gather log probs for actions taken
        actions_unsqueezed = actions.unsqueeze(-1)
        log_pi_a = log_probs_pi.gather(-1, actions_unsqueezed).squeeze(-1)
        log_pi_reg_a = log_probs_fixed.gather(-1, actions_unsqueezed).squeeze(-1)

        # Regularized rewards for V-trace targets
        # r_reg = r - alpha * (log_pi - log_pi_reg)
        # We detach log_pi_a to treat it as fixed for V-trace target calculation
        r_reg = rewards - self.alpha_rnad * (log_pi_a.detach() - log_pi_reg_a)

        # Bootstrap value (assume 0 for end of episode)
        bootstrap_value = torch.zeros(B, device=self.device)

        # V-trace
        vs, pg_advantages = v_trace(
            log_mu_a=log_probs_mu,
            log_pi_a=log_pi_a,
            values=curr_values,
            bootstrap_value=bootstrap_value,
            rewards=r_reg,
            discounts=torch.full_like(rewards, self.discount_factor),
            clip_rho_threshold=self.clip_rho_threshold,
            clip_pg_rho_threshold=self.clip_pg_rho_threshold
        )

        # Mask for valid steps
        valid_mask = seq_mask.float()

        # Value loss
        # Target vs should be detached
        value_loss = 0.5 * (vs.detach() - curr_values).pow(2) * valid_mask
        value_loss = value_loss.sum() / valid_mask.sum()

        # Policy loss
        # R-NaD policy gradient: minimize - E[ log_pi * advantage ]
        # Advantage here is regularized advantage from V-trace. It must be detached.
        policy_loss = - (log_pi_a * pg_advantages.detach()) * valid_mask
        policy_loss = policy_loss.sum() / valid_mask.sum()

        return policy_loss, value_loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        policy_loss, value_loss = self.compute_loss(batch)
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
