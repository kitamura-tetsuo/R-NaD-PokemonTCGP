import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pyspiel
import numpy as np
import logging
import time
import pickle
import os
import re
import deckgym
from typing import NamedTuple, Tuple, List, Dict, Any, Optional
from src.models import DeckGymNet
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import deckgym_openspiel
except ImportError:
    logging.warning("deckgym_openspiel not found. Assuming environment is available via pyspiel.")

# Try importing deckgym (needed for PyGameState if not available in deckgym_openspiel)
# This is used for exact evaluation in evaluate_against_baseline
try:
    import deckgym
    if hasattr(deckgym, 'deckgym') and hasattr(deckgym.deckgym, 'PyGameState'):
        PyGameState = deckgym.deckgym.PyGameState
    elif hasattr(deckgym, 'PyGameState'):
        PyGameState = deckgym.PyGameState
    else:
        # Fallback
        from deckgym.deckgym import PyGameState
except ImportError:
    PyGameState = None
    pass

# Fix deckgym import if necessary
try:
    import deckgym
    if not hasattr(deckgym, 'PyBatchedSimulator'):
        if hasattr(deckgym, 'deckgym') and hasattr(deckgym.deckgym, 'PyBatchedSimulator'):
             deckgym.PyBatchedSimulator = deckgym.deckgym.PyBatchedSimulator
        else:
             try:
                 from deckgym.deckgym import PyBatchedSimulator
                 deckgym.PyBatchedSimulator = PyBatchedSimulator
             except ImportError:
                 pass
except ImportError:
    pass

class LeagueConfig(NamedTuple):
    decks: List[str] = ["deckgym-core/example_decks/mewtwoex.txt"]
    rates: List[float] = [1.0]
    fixed_decks: List[str] = [] # Decks that always participate in matches

    def sample_decks(self, batch_size: int) -> Tuple[List[str], List[str]]:
        """Samples deck pairs for a batch."""
        p = np.array(self.rates) / sum(self.rates)
        
        # If no fixed decks, sample both players from league
        if not self.fixed_decks:
            decks_1 = np.random.choice(self.decks, size=batch_size, p=p).tolist()
            decks_2 = np.random.choice(self.decks, size=batch_size, p=p).tolist()
            return decks_1, decks_2
        
        # If we have fixed decks, we ensure one player always uses a fixed deck.
        # For simplicity, let's say we alternate or sample which player gets the fixed deck,
        # or just fix player 2 to be one of the fixed decks.
        # The user said "常に試合に参加する固定デッキの指定も可能にして下さい。固定デッキは指定しない場合もあります。"
        # "Fixed decks" means these decks are always present in the match.
        # If multiple fixed decks are specified, we sample from them.
        
        # Sample opponent decks from league
        league_decks = np.random.choice(self.decks, size=batch_size, p=p).tolist()
        # Sample fixed decks
        fixed_sampled = np.random.choice(self.fixed_decks, size=batch_size).tolist()
        
        # Randomly assign fixed deck to player 1 or 2
        final_1 = []
        final_2 = []
        for i in range(batch_size):
            if np.random.rand() < 0.5:
                final_1.append(fixed_sampled[i])
                final_2.append(league_decks[i])
            else:
                final_1.append(league_decks[i])
                final_2.append(fixed_sampled[i])
        
        return final_1, final_2

class RNaDConfig(NamedTuple):
    batch_size: int = 128
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    max_steps: int = 1000
    entropy_schedule_start: float = 0.1
    entropy_schedule_end: float = 0.01
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    hidden_size: int = 256
    num_blocks: int = 4
    log_interval: int = 100
    save_interval: int = 1000
    deck_id_1: str = "deckgym-core/example_decks/mewtwoex.txt"
    deck_id_2: str = "deckgym-core/example_decks/mewtwoex.txt"
    league_config: Optional[LeagueConfig] = None
    win_reward: float = 1.0
    point_reward: float = 0.0
    damage_reward: float = 0.0
    enable_profiler: bool = False
    profiler_dir: str = "runs/profile"
    profile_start_step: int = 10
    profile_num_steps: int = 10
    past_self_play: bool = False
    test_interval: int = 10
    test_interval: int = 10
    test_games: int = 8
    unroll_length: int = 200

def v_trace(
    v_tm1: jnp.ndarray, # (T, B)
    r_t: jnp.ndarray,   # (T, B)
    rho_t: jnp.ndarray, # (T, B)
    gamma: float = 0.99,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    bootstrap_value: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes V-trace targets and advantages.
    Assumes v_tm1 contains values for steps 0 to T-1.
    """
    T = v_tm1.shape[0]

    # Clip importance weights
    rho_bar_t = jnp.minimum(rho_t, clip_rho_threshold)
    c_bar_t = jnp.minimum(rho_t, clip_pg_rho_threshold)

    # We need a bootstrap value for V_T.
    # v_tm1 has shape (T, B). bootstrap_value has shape (B,) or (1, B).
    if bootstrap_value is None:
        bootstrap_value = jnp.zeros((1, v_tm1.shape[1]))
    else:
        # Ensure shape (1, B)
        if bootstrap_value.ndim == 1:
            bootstrap_value = bootstrap_value[None, :]
            
    v_all = jnp.concatenate([v_tm1, bootstrap_value], axis=0) # (T+1, B)

    def scan_body(carry, x):
        acc = carry
        rho_bar, c_bar, r, v_curr, v_next = x

        delta = rho_bar * (r + gamma * v_next - v_curr)
        acc = delta + gamma * c_bar * acc
        return acc, acc + v_curr

    # Scan from T-1 down to 0
    xs = (rho_bar_t, c_bar_t, r_t, v_all[:-1], v_all[1:])
    xs_rev = jax.tree_util.tree_map(lambda x: x[::-1], xs)

    init_acc = jnp.zeros_like(v_tm1[0])
    _, vs_rev = jax.lax.scan(scan_body, init_acc, xs_rev)

    vs = vs_rev[::-1]

    # Advantages for Policy Gradient
    vs_plus_1 = jnp.concatenate([vs[1:], jnp.zeros((1, vs.shape[1]))], axis=0)
    rho_pg = jnp.minimum(rho_t, clip_pg_rho_threshold)
    pg_advantages = rho_pg * (r_t + gamma * vs_plus_1 - v_all[:-1])

    return vs, pg_advantages

def loss_fn(params, fixed_params, batch, apply_fn, config: RNaDConfig, alpha_rnad: float):
    obs = batch['obs'] # (T, B, dim)
    act = batch['act'] # (T, B)
    rew = batch['rew'] # (T, B)
    log_prob_behavior = batch['log_prob'] # (T, B)

    T, B, _ = obs.shape
    flat_obs = obs.reshape(-1, obs.shape[-1])

    # Forward pass (Current Policy)
    logits, values = apply_fn(params, jax.random.PRNGKey(0), flat_obs)
    logits = logits.reshape(T, B, -1)
    values = values.reshape(T, B)

    # Forward pass (Fixed Policy)
    fixed_logits, _ = apply_fn(fixed_params, jax.random.PRNGKey(0), flat_obs)
    fixed_logits = fixed_logits.reshape(T, B, -1)

    # Log Probs
    log_probs = jax.nn.log_softmax(logits)
    fixed_log_probs = jax.nn.log_softmax(fixed_logits)

    # Select log prob of taken actions
    act_one_hot = jax.nn.one_hot(act, logits.shape[-1])
    log_pi_a = jnp.sum(log_probs * act_one_hot, axis=-1)
    log_pi_fixed_a = jnp.sum(fixed_log_probs * act_one_hot, axis=-1)

    # Regularized Reward: r_reg = r - alpha * (log_pi - log_pi_fixed)
    r_reg = rew - alpha_rnad * (log_pi_a - log_pi_fixed_a)

    # Importance Sampling Weights
    log_rho = log_pi_a - log_prob_behavior
    rho = jnp.exp(log_rho)

    # V-trace
    vs, pg_adv = v_trace(
        values,
        r_reg,
        rho,
        gamma=config.discount_factor,
        clip_rho_threshold=config.clip_rho_threshold,
        clip_pg_rho_threshold=config.clip_pg_rho_threshold,
        bootstrap_value=batch.get('bootstrap_value', None)
    )

    # Value Loss
    value_loss = 0.5 * jnp.mean((jax.lax.stop_gradient(vs) - values) ** 2)

    # Policy Loss
    policy_loss = -jnp.mean(log_pi_a * jax.lax.stop_gradient(pg_adv))

    # Metrics
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    mean_entropy = jnp.mean(entropy)

    kl = jnp.sum(probs * (log_probs - fixed_log_probs), axis=-1)
    mean_kl = jnp.mean(kl)

    y_true = jax.lax.stop_gradient(vs)
    y_pred = values
    var_y = jnp.var(y_true)
    var_resid = jnp.var(y_true - y_pred)
    explained_variance = 1 - var_resid / (var_y + 1e-8)

    total_loss = policy_loss + value_loss

    return total_loss, (policy_loss, value_loss, mean_entropy, mean_kl, explained_variance)

class RNaDLearner:
    def __init__(self, game_name: str, config: RNaDConfig):
        self.game = pyspiel.load_game(
            game_name,
            {
                "deck_id_1": config.deck_id_1,
                "deck_id_2": config.deck_id_2,
                "max_game_length": config.unroll_length
            }
        )
        self.config = config
        self.num_actions = self.game.num_distinct_actions()
        self.obs_shape = self.game.observation_tensor_shape()

        def forward(x):
            net = DeckGymNet(
                num_actions=self.num_actions,
                hidden_size=config.hidden_size,
                num_blocks=config.num_blocks
            )
            return net(x)

        self.network = hk.transform(forward)
        self.params = None
        self.fixed_params = None
        self.opt_state = None
        self.optimizer = optax.adam(learning_rate=config.learning_rate)

        # JIT the update function
        self._update_fn = jax.jit(partial(self._update_pure, apply_fn=self.network.apply, config=self.config, optimizer=self.optimizer))

        # Batched Simulator in Rust
        self.batched_sim = deckgym.PyBatchedSimulator(
            config.deck_id_1,
            config.deck_id_2,
            config.batch_size,
            config.win_reward,
            config.point_reward,
            config.damage_reward
        )

        # JIT the inference function for speed
        @jax.jit
        def _inference_fn(params, obs):
            logits, _ = self.network.apply(params, jax.random.PRNGKey(0), obs)
            return logits
        self._inference_fn = _inference_fn

        # JIT the value function for bootstrapping
        @jax.jit
        def _value_fn(params, obs):
            _, values = self.network.apply(params, jax.random.PRNGKey(0), obs)
            return values
        self._value_fn = _value_fn

    def save_checkpoint(self, path: str, step: int):
        data = {
            'params': self.params,
            'fixed_params': self.fixed_params,
            'opt_state': self.opt_state,
            'step': step,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Checkpoint saved to {path} at step {step}")

    def load_checkpoint(self, path: str) -> int:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.params = data['params']
        self.fixed_params = data['fixed_params']
        self.opt_state = data['opt_state']
        step = data['step']
        # We generally trust the loaded config to match roughly or ignore it,
        # or we could assert config compatibility.
        # For now, we assume the user knows what they are doing.
        logging.info(f"Checkpoint loaded from {path}, resuming from step {step}")
        return step

    def init(self, key):
        dummy_obs = jnp.zeros((1, *self.obs_shape))
        self.params = self.network.init(key, dummy_obs)
        self.fixed_params = self.params
        self.opt_state = self.optimizer.init(self.params)

    @staticmethod
    def _update_pure(params, fixed_params, opt_state, batch, apply_fn, config, optimizer, alpha_rnad):
        # Wrapper to allow jax.value_and_grad
        def loss_wrapper(p):
            return loss_fn(p, fixed_params, batch, apply_fn, config, alpha_rnad)

        (total_loss, (p_loss, v_loss, ent, kl, ev)), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, {
            'total_loss': total_loss,
            'policy_loss': p_loss,
            'value_loss': v_loss,
            'policy_entropy': ent,
            'approx_kl': kl,
            'explained_variance': ev
        }

    def update(self, batch, step: int):
        # Anneal alpha_rnad
        progress = min(1.0, step / self.config.max_steps)
        alpha = self.config.entropy_schedule_start + progress * (self.config.entropy_schedule_end - self.config.entropy_schedule_start)

        self.params, self.opt_state, metrics = self._update_fn(
            self.params, self.fixed_params, self.opt_state, batch, alpha_rnad=alpha
        )
        metrics['alpha'] = alpha

        if 'stats' in batch:
            metrics.update(batch['stats'])

        return metrics

    def update_fixed_point(self):
        self.fixed_params = self.params

    def train_step(self, key, step: int):
        # 1. Generate Trajectories
        batch = self.generate_trajectories(key)

        # 2. Update
        metrics = self.update(batch, step)

        return metrics

    def generate_trajectories(self, key):
        # Sample decks
        if self.config.league_config:
            decks_1, decks_2 = self.config.league_config.sample_decks(self.config.batch_size)
        else:
            decks_1 = [self.config.deck_id_1] * self.config.batch_size
            decks_2 = [self.config.deck_id_2] * self.config.batch_size

        reset_result = self.batched_sim.reset(
            seed=int(jax.random.randint(key, (), 0, 1000000)),
            deck_ids_1=decks_1,
            deck_ids_2=decks_2
        )
        
        # New signature: reset(..) -> (obs, current_players)
        initial_obs, initial_current_players = reset_result

        # Determine if we use past self-play
        use_past_self_play = False
        past_params = None
        
        if self.config.past_self_play:
            # Check for checkpoints
            checkpoint_dir = getattr(self.config, 'checkpoint_dir', 'checkpoints')
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
                if checkpoints and len(checkpoints) > 0:
                     use_past_self_play = True
                     # Pick a random checkpoint
                     chosen_cp = np.random.choice(checkpoints)
                     try:
                         with open(os.path.join(checkpoint_dir, chosen_cp), 'rb') as f:
                             data = pickle.load(f)
                             past_params = data['params']
                     except Exception as e:
                         logging.warning(f"Failed to load past checkpoint {chosen_cp}: {e}")
                         use_past_self_play = False
            
            if not use_past_self_play and self.config.past_self_play:
                 logging.warning("Past self-play requested but no checkpoints found. Falling back to self-play.")

        # Assign agent identity (0 or 1) per environment
        # agent_player_ids[i] = 0 means Agent is Player 0, Past (or Self) is Player 1
        agent_player_ids = np.zeros(self.config.batch_size, dtype=int)
        if use_past_self_play:
             # Randomly assign Agent to P1 (0) or P2 (1)
             agent_player_ids = np.random.randint(0, 2, size=self.config.batch_size)

        # Fixed size buffers
        max_len = self.config.unroll_length
        batch_size = self.config.batch_size
        
        obs_buf = np.zeros((max_len, batch_size, *self.obs_shape), dtype=np.float32)
        act_buf = np.zeros((max_len, batch_size), dtype=np.int32)
        rew_buf = np.zeros((max_len, batch_size), dtype=np.float32)
        log_prob_buf = np.zeros((max_len, batch_size), dtype=np.float32)
        
        ptrs = np.zeros(batch_size, dtype=int) # Position in buffer for each env
        
        active_mask = np.ones(batch_size, dtype=bool)
        current_obs = np.array(initial_obs)
        current_players = np.array(initial_current_players)

        # Track episode outcomes
        # 0: Tie, 1: P1 Win, 2: P2 Win
        episode_outcomes = np.zeros(batch_size, dtype=int)
        outcome_recorded = np.zeros(batch_size, dtype=bool)
        
        # Stats tracking
        episode_lengths = np.zeros(batch_size, dtype=int) # Game steps (not stored steps)

        for i_step in range(max_len):
            if not active_mask.any():
                break

            # 1. Inference
            t0 = time.time()
            logits = self._inference_fn(self.params, current_obs)
            logits.block_until_ready()
            t1 = time.time()
            logits_np = np.array(logits)
            t2 = time.time()

            if use_past_self_play:
                past_logits = self._inference_fn(past_params, current_obs)
                past_logits_np = np.array(past_logits)
                
                mask_agent = (current_players == agent_player_ids)
                final_logits = np.where(mask_agent[:, None], logits_np, past_logits_np)
            else:
                final_logits = logits_np

            # 2. Sample and Step in Rust
            t3 = time.time()
            (next_obs, rewards, dones, _, valid_mask, actions, log_probs, next_current_players) = \
                self.batched_sim.sample_and_step(final_logits)
            t4 = time.time()

            if i_step % 10 == 0:
                 logging.info(f"Step {i_step}: Inference(TPU)={t1-t0:.4f}s, Transfer(D->H)={t2-t1:.4f}s, Sim(Rust)={t4-t3:.4f}s")
            
            # 3. Store transitions
            for i in range(batch_size):
                if active_mask[i]:
                    if valid_mask[i]:
                        episode_lengths[i] += 1
                        
                        # Determine if we should store this transition
                        should_store = True
                        if use_past_self_play:
                            if current_players[i] != agent_player_ids[i]:
                                should_store = False
                        
                        if should_store:
                            idx = ptrs[i]
                            if idx < max_len:
                                obs_buf[idx, i] = current_obs[i]
                                act_buf[idx, i] = actions[i]
                                rew_buf[idx, i] = rewards[i]
                                log_prob_buf[idx, i] = log_probs[i]
                                ptrs[i] += 1
                        
                        if dones[i]:
                            active_mask[i] = False
                            if not outcome_recorded[i]:
                                # Determine outcome based on reward
                                actor = current_players[i]
                                r = rewards[i]

                                if abs(r) > 1e-3:
                                    if r > 0:
                                        episode_outcomes[i] = 1 if actor == 0 else 2
                                    else:
                                        episode_outcomes[i] = 2 if actor == 0 else 1
                                else:
                                    episode_outcomes[i] = 0 # Tie

                                outcome_recorded[i] = True

                    else:
                        active_mask[i] = False

            current_obs = np.array(next_obs)
            current_players = np.array(next_current_players)

        # Calculate stats
        mean_episode_length = np.mean(episode_lengths)
        max_episode_length_stat = np.max(episode_lengths)

        p1_wins = np.sum(episode_outcomes == 1)
        p2_wins = np.sum(episode_outcomes == 2)
        ties = np.sum(episode_outcomes == 0)
        total_games = batch_size

        p1_win_rate = p1_wins / total_games
        decisive_rate = (p1_wins + p2_wins) / total_games
        tie_rate = ties / total_games

        # Bootstrap value for truncated episodes
        # If active_mask[i] is True, the game is truncated at `current_obs`.
        bootstrap_value = np.zeros(batch_size, dtype=np.float32)
        if active_mask.any():
            vals = self._value_fn(self.params, current_obs)
            vals_np = np.array(vals).reshape(-1)
            bootstrap_value = np.where(active_mask, vals_np, 0.0)

        batch = {
            'obs': jnp.array(obs_buf),
            'act': jnp.array(act_buf),
            'rew': jnp.array(rew_buf),
            'log_prob': jnp.array(log_prob_buf),
            'bootstrap_value': jnp.array(bootstrap_value),
            'stats': {
                'mean_episode_length': mean_episode_length,
                'max_episode_length': max_episode_length_stat,
                'p1_win_rate': p1_win_rate,
                'decisive_rate': decisive_rate,
                'tie_rate': tie_rate
            }
        }
        return batch

import queue
import threading

class TrajectoryGenerator(threading.Thread):
    def __init__(self, learner, num_workers=1, max_queue_size=10):
        super().__init__()
        self.learner = learner
        self.num_workers = num_workers
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stopped = False
        self.daemon = True

    def run(self):
        worker_id = 0
        while not self.stopped:
            try:
                key = jax.random.PRNGKey(worker_id)
                batch = self.learner.generate_trajectories(key)
                self.queue.put(batch)
                worker_id += 1
            except Exception as e:
                logging.error(f"Error in TrajectoryGenerator: {e}")
                break

    def stop(self):
        self.stopped = True

    def get_batch(self):
        return self.queue.get()

def evaluate_against_baseline(learner: RNaDLearner, baseline_params: Any, config: RNaDConfig, num_games: int) -> Dict[str, float]:
    """
    Evaluates the current learner against a baseline (e.g., step 0) using PyGameState.
    Iterates through all deck combinations configured in LeagueConfig or default decks.
    """
    if PyGameState is None:
        logging.warning("PyGameState not available. Skipping evaluation.")
        return {}
        
    decks_1 = [config.deck_id_1]
    decks_2 = [config.deck_id_2]
    
    if config.league_config:
        # Sort decks by rates descending, stable sort preserves original order for ties
        # User requested: "test で使用するデッキは重み順の上位3デッキに変更して下さい。"
        zipped = list(zip(config.league_config.decks, config.league_config.rates))
        zipped.sort(key=lambda x: x[1], reverse=True)
        top_decks = [d for d, r in zipped[:3]]
        
        decks_1 = top_decks
        decks_2 = top_decks
        
    # We want to test every combination of d1 vs d2
    # For each pair, we run num_games
    # We want to know:
    # 1. Win rate of Current (P1) vs Baseline (P2)
    # 2. Win rate of Current (P2) vs Baseline (P1) ?? 
    # The user request says: "step==0 の時のチェックポイントの結果と対戦をします。" 
    # and "リーグであれば組み合わせ毎の勝率と、全組み合わせの平均の勝率の計算をしてmlflowのtestパラメータとして保存して下さい。"
    
    results = {}
    total_wins = 0
    total_games = 0
    
    # Pre-compile inference functions
    # learner.network is already transformed
    
    # Helper to run a batch of games for a specific deck pair
    def run_pair(d1_path, d2_path, current_params_p1, baseline_params_p2, n_games):
        # We can reuse the learner's network definition
        # But we need to handle the loop carefully to avoid JIT overhead if possible,
        # or just rely on the fact that the network is the same.
        
        # We'll use a simple batched loop similar to plot_winrates.py
        # but integrated here.
        
        batch_size = min(config.batch_size, n_games)
        wins = 0
        ties = 0
        played = 0
        
        while played < n_games:
            current_batch = min(batch_size, n_games - played)
            
            # Init games
            games = [PyGameState(d1_path, d2_path, None) for _ in range(current_batch)]
            
            active_mask = np.ones(current_batch, dtype=bool)
            current_obs = np.array([g.encode_observation() for g in games], dtype=np.float32)
            
            while active_mask.any():
                # Inference
                # We need to obtain logits for both players
                # P1 = current_params_p1
                # P2 = baseline_params_p2
                
                # Get current players
                p1_indices = []
                p2_indices = []
                active_indices = np.where(active_mask)[0]
                
                for idx in active_indices:
                    cp = games[idx].get_state().current_player
                    if cp == 0:
                        p1_indices.append(idx)
                    else:
                        p2_indices.append(idx)
                
                batch_logits = np.zeros((current_batch, learner.num_actions), dtype=np.float32)
                
                if p1_indices:
                    obs_p1 = current_obs[p1_indices]
                    logits_p1 = learner._inference_fn(current_params_p1, obs_p1)
                    batch_logits[p1_indices] = np.array(logits_p1)
                
                if p2_indices:
                    obs_p2 = current_obs[p2_indices]
                    logits_p2 = learner._inference_fn(baseline_params_p2, obs_p2)
                    batch_logits[p2_indices] = np.array(logits_p2)

                # Step
                actions = []
                for i in range(current_batch):
                    if not active_mask[i]:
                        actions.append(0)
                        continue
                        
                    legal = games[i].legal_actions()
                    logits = batch_logits[i]
                    # Mask illegal
                    # We can do a simpler mask here since we are in Python loop
                    # robust softmax sampling
                    
                    # Create a safe logits array
                    safe_logits = np.full_like(logits, -1e9)
                    safe_logits[legal] = logits[legal]
                    
                    l_max = np.max(safe_logits[legal])
                    exp_l = np.exp(safe_logits[legal] - l_max)
                    probs = exp_l / np.sum(exp_l)
                    
                    # Sample
                    a_legal_idx = np.random.choice(len(legal), p=probs)
                    a = legal[a_legal_idx]
                    actions.append(a)
                
                # Apply actions
                next_obs_list = []
                for i in range(current_batch):
                    if active_mask[i]:
                        try:
                            done, p0_won = games[i].step_with_id(actions[i])
                            if done:
                                active_mask[i] = False
                                outcome = games[i].get_state().winner
                                if outcome is not None:
                                    if outcome.winner == 0:
                                        wins += 1
                                    elif outcome.is_tie:
                                        ties += 1
                                next_obs_list.append(np.zeros_like(current_obs[0]))
                            else:
                                next_obs_list.append(games[i].encode_observation())
                        except Exception as e:
                            logging.error(f"Error in eval step: {e}")
                            active_mask[i] = False
                            next_obs_list.append(np.zeros_like(current_obs[0]))
                    else:
                        next_obs_list.append(np.zeros_like(current_obs[0]))
                
                current_obs = np.array(next_obs_list, dtype=np.float32)

            played += current_batch
            
        return wins, ties

    import itertools
    for d1, d2 in itertools.product(decks_1, decks_2):
        d1_name = os.path.splitext(os.path.basename(d1))[0]
        d2_name = os.path.splitext(os.path.basename(d2))[0]
        
        # Eval: Current (P1) vs Baseline (P2)
        wins, ties = run_pair(d1, d2, learner.params, baseline_params, num_games)
        
        # Calculate win rate (ignoring ties for win/loss ratio, or counting as 0.5?
        # Standard: wins / total
        wr = wins / num_games
        results[f"test_winrate_{d1_name}_vs_{d2_name}"] = wr
        
        total_wins += wins
        total_games += num_games
        
    if total_games > 0:
        results["test_winrate_mean"] = total_wins / total_games
    
    return results

def train_loop(config: RNaDConfig, experiment_manager: Optional[Any] = None, checkpoint_dir: str = "checkpoints", resume_checkpoint: Optional[str] = None):
    learner = RNaDLearner("deckgym_ptcgp", config)
    learner.init(jax.random.PRNGKey(42))

    start_step = 0

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            start_step = learner.load_checkpoint(resume_checkpoint)
            start_step += 1 # Resume from next step
        else:
            logging.error(f"Checkpoint {resume_checkpoint} not found!")
            return
    else:
        # Check for latest checkpoint in directory
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
        if checkpoints:
            # Extract step numbers
            steps = []
            for cp in checkpoints:
                match = re.search(r"checkpoint_(\d+).pkl", cp)
                if match:
                    steps.append(int(match.group(1)))

            if steps:
                latest_step = max(steps)
                latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{latest_step}.pkl")
                start_step = learner.load_checkpoint(latest_checkpoint)
                start_step += 1
                logging.info(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")

    if experiment_manager:
        experiment_manager.log_params(config)

    logging.info(f"Starting training loop from step {start_step}...")

    # Save interval
    save_interval = config.save_interval if hasattr(config, 'save_interval') else 1000

    # Start Trajectory Generator
    generator = TrajectoryGenerator(learner)
    generator.start()
    
    # Capture baseline parameters (Step 0)
    baseline_params = None
    # If we are at step 0, we can use current params
    if start_step == 0:
        baseline_params = learner.params
        # We might want to save this explicitly if not already saved
        # But we can also rely on checkpoint_0.pkl being created/loaded
    else:
        # Try to load checkpoint 0
        try:
            cp0 = os.path.join(checkpoint_dir, "checkpoint_0.pkl")
            if os.path.exists(cp0):
                with open(cp0, 'rb') as f:
                    data = pickle.load(f)
                baseline_params = data['params']
                logging.info("Loaded baseline parameters from checkpoint_0.pkl")
            else:
                logging.warning("checkpoint_0.pkl not found. Baseline evaluation will be skipped until it is found/created.")
        except Exception as e:
            logging.warning(f"Failed to load baseline parameters: {e}")

    for step in range(start_step, config.max_steps):
        if config.enable_profiler:
            if step == config.profile_start_step:
                logging.info(f"Starting JAX profiler trace at step {step}. Saving to {config.profiler_dir}")
                jax.profiler.start_trace(config.profiler_dir)
            elif step == config.profile_start_step + config.profile_num_steps:
                logging.info(f"Stopping JAX profiler trace at step {step}.")
                jax.profiler.stop_trace()

        start_time = time.time()

        # 1. Get batch from background generator
        batch = generator.get_batch()

        # 2. Update
        metrics = learner.update(batch, step)

        end_time = time.time()

        # Calculate SPS
        total_steps = metrics.get('mean_episode_length', 0) * config.batch_size
        step_time = end_time - start_time
        sps = total_steps / (step_time + 1e-6)
        metrics['sps'] = sps

        if step % 10 == 0:
            logging.info(f"Step {step}: {metrics}")

        if experiment_manager and step % config.log_interval == 0:
            experiment_manager.log_metrics(step, metrics)

        if experiment_manager and step % config.save_interval == 0:
            experiment_manager.save_model(step, learner.params)

        if step % 100 == 0:
            learner.update_fixed_point()
            logging.info("Updated fixed point.")

        if step % save_interval == 0:
             ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
             learner.save_checkpoint(ckpt_path, step)
        
        # Periodic Evaluation against Baseline
        if config.test_interval > 0 and step % config.test_interval == 0:
             if baseline_params is None:
                 # Try loading again (maybe it was created in this run)
                 try:
                    cp0 = os.path.join(checkpoint_dir, "checkpoint_0.pkl")
                    if os.path.exists(cp0):
                        with open(cp0, 'rb') as f:
                            data = pickle.load(f)
                        baseline_params = data['params']
                 except Exception:
                     pass
            
             if baseline_params is not None:
                 logging.info(f"Step {step}: Running evaluation against baseline...")
                 eval_metrics = evaluate_against_baseline(
                     learner, 
                     baseline_params, 
                     config, 
                     num_games=config.test_games
                 )
                 # Log to experiment manager
                 if experiment_manager:
                     experiment_manager.log_metrics(step, eval_metrics)
                 logging.info(f"Step {step}: Evaluation results: {eval_metrics}")

    # Save final checkpoint
    final_step = config.max_steps - 1
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{final_step}.pkl")
    learner.save_checkpoint(ckpt_path, final_step)
    logging.info("Training complete.")

if __name__ == "__main__":
    train_loop(RNaDConfig())
