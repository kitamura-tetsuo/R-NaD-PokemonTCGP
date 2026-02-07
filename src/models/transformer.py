import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.fc1(x)

        x = self.norm2(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x + residual

class Torso(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_blocks):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return x

class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # We return logits
        return self.fc(x)

class ValueHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Return win probability [0, 1]
        return torch.sigmoid(self.fc(x))

class UnifiedModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.torso = Torso(obs_dim, hidden_dim, num_blocks)
        self.policy_head = PolicyHead(hidden_dim, action_dim)
        self.value_head = ValueHead(hidden_dim)

    def forward(self, x, action_mask=None):
        features = self.torso(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        if action_mask is not None:
            # We assume action_mask is 1 for valid actions and 0 for invalid actions.
            # We mask invalid actions (where mask is 0 or False).
            if action_mask.dtype == torch.bool:
                mask = ~action_mask
            else:
                mask = action_mask == 0
            policy_logits = policy_logits.masked_fill(mask, -1e9)
        return policy_logits, value
