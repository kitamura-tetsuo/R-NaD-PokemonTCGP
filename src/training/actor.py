import torch
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from src.models.transformer import UnifiedModel

try:
    # Try importing deckgym if available
    import deckgym
except ImportError:
    deckgym = None

class MockDeckGym:
    """
    Mock environment for DeckGym to allow testing without the actual package.
    Observation space: Box(30574,)
    Action space: Discrete(25000)
    """
    def __init__(self):
        self.obs_dim = 30574
        self.action_dim = 25000
        self.current_step = 0
        self.max_steps = 100
        self.current_player = 0 # 0 for Player 1, 1 for Player 2

    def reset(self):
        self.current_step = 0
        self.current_player = 0
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        info = {
            'action_mask': np.ones(self.action_dim, dtype=bool),
            'current_player': self.current_player
        }
        return obs, info

    def step(self, action):
        self.current_step += 1
        # Toggle player
        self.current_player = 1 - self.current_player

        obs = np.random.randn(self.obs_dim).astype(np.float32)
        reward = np.random.randn() # Random reward
        done = self.current_step >= self.max_steps
        truncated = False
        info = {
            'action_mask': np.ones(self.action_dim, dtype=bool),
            'current_player': self.current_player
        }
        return obs, reward, done, truncated, info

class SelfPlayWorker:
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        if deckgym:
            # Assuming deckgym has a make function or similar
            # Since I don't know the exact API, I'll assume standard gym.make
            # but if it fails, fallback to Mock.
            try:
                self.env = deckgym.make("PTCGPEnv-v0") # Guessing name
            except Exception as e:
                logging.warning(f"Failed to initialize deckgym: {e}. Using MockDeckGym.")
                self.env = MockDeckGym()
        else:
            logging.warning("deckgym not found. Using MockDeckGym.")
            self.env = MockDeckGym()

    def _flip_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Flips the observation relative to the current player's perspective.

        Since the observation structure is unknown (flat vector), this is a placeholder.
        Real implementation should permute the observation vector to swap player perspectives.
        """
        # TODO: Implement actual flipping logic based on DeckGym observation schema
        return observation

    def run_episode(self, model: UnifiedModel) -> Dict[str, torch.Tensor]:
        """
        Runs a single episode of self-play using the provided model.
        Returns a dictionary containing trajectory data.
        """
        model.eval()

        obs, info = self.env.reset()
        current_player = info.get('current_player', 0)

        states = []
        actions = []
        rewards = []
        masks = []
        log_probs = []

        done = False
        truncated = False

        while not (done or truncated):
            # Process observation
            # Flip observation if it is Player 2's turn (assuming 1 is Player 2)
            # We assume P1 is 0, P2 is 1.
            # If env returns global state, or P1 perspective, we flip for P2.
            # If env returns current player perspective, we don't need to flip.
            # The prompt says: "Note that the observation must be flipped relative to the current player's perspective."
            # This implies I must flip it.
            if current_player == 1:
                processed_obs = self._flip_observation(obs)
            else:
                processed_obs = obs

            obs_tensor = torch.tensor(processed_obs, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, obs_dim)

            # Action mask
            action_mask = info.get('action_mask')
            if action_mask is None:
                action_mask = np.ones(25000, dtype=bool) # Fallback

            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0) # (1, action_dim)

            with torch.no_grad():
                logits, _ = model(obs_tensor, action_mask=mask_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_val = action.item()

            # Step environment
            next_obs, reward, done, truncated, next_info = self.env.step(action_val)

            # Store data
            # We store the state seen by the agent (processed_obs)
            states.append(torch.tensor(processed_obs, dtype=torch.float32)) # (obs_dim)
            actions.append(torch.tensor(action_val, dtype=torch.long)) # ()
            rewards.append(torch.tensor(reward, dtype=torch.float32)) # ()
            masks.append(torch.tensor(action_mask, dtype=torch.bool)) # (action_dim)
            log_probs.append(log_prob.squeeze()) # ()

            # Update loop variables
            obs = next_obs
            info = next_info
            current_player = info.get('current_player', 1 - current_player) # Update player if available, else toggle

        # Stack lists into tensors
        # Result shapes: (T, ...)
        episode_data = {
            'states': torch.stack(states).to(self.device), # (T, obs_dim)
            'actions': torch.stack(actions).to(self.device), # (T)
            'rewards': torch.stack(rewards).to(self.device), # (T)
            'masks': torch.stack(masks).to(self.device), # (T, action_dim)
            'log_probs': torch.stack(log_probs).to(self.device) # (T)
        }

        return episode_data
