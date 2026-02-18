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
import optuna # Added for HPO
from typing import NamedTuple, Tuple, List, Dict, Any, Optional
from src.models import DeckGymNet, TransformerNet, CardTransformerNet
from functools import partial

# Enable bfloat16 for TPU v6e
jax.config.update("jax_default_matmul_precision", "bfloat16")

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

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Finds the latest checkpoint in the given directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
    if not checkpoints:
        return None

    # Extract step numbers
    steps = []
    checkpoint_map = {}
    for cp in checkpoints:
        match = re.search(r"checkpoint_(\d+).pkl", cp)
        if match:
            step = int(match.group(1))
            steps.append(step)
            checkpoint_map[step] = cp

    if not steps:
        return None

    latest_step = max(steps)
    return os.path.join(checkpoint_dir, checkpoint_map[latest_step])

class LeagueConfig(NamedTuple):
    decks: List[str] = ["deckgym-core/example_decks/mewtwoex.txt"]
    rates: List[float] = [1.0]
    fixed_decks: List[str] = [] # Decks that always participate in matches
    
    # New fields for student/teacher league
    student_decks: Optional[List[str]] = None
    student_rates: Optional[List[float]] = None
    teacher_decks: Optional[List[str]] = None
    teacher_rates: Optional[List[float]] = None

    @staticmethod
    def from_csv(student_csv: Optional[str], teacher_csv: Optional[str]) -> 'LeagueConfig':
        """Loads league configuration from CSV files."""
        def load_one(csv_path):
            if not csv_path or not os.path.exists(csv_path):
                return None, None
            
            decks = []
            rates = []
            csv_dir = os.path.dirname(csv_path)
            
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Expecting headers: signature, lower_ci (others ignored)
                    sig = row.get('signature')
                    weight = row.get('lower_ci')
                    if not sig or not weight:
                        continue
                    
                    # Deck file is expected in the same directory as CSV
                    deck_path = os.path.join(csv_dir, f"{sig}.txt")
                    if os.path.exists(deck_path):
                        decks.append(deck_path)
                        rates.append(float(weight))
                    else:
                        logging.warning(f"Deck file {deck_path} not found, skipping.")
            
            return decks, rates

        s_decks, s_rates = load_one(student_csv)
        t_decks, t_rates = load_one(teacher_csv)
        
        # If student/teacher leagues are provided, they override the default decks/rates
        return LeagueConfig(
            decks=s_decks or ["deckgym-core/example_decks/mewtwoex.txt"],
            rates=s_rates or [1.0],
            student_decks=s_decks,
            student_rates=s_rates,
            teacher_decks=t_decks,
            teacher_rates=t_rates
        )

    def sample_decks(self, batch_size: int) -> Tuple[List[str], List[str]]:
        """Samples deck pairs for a batch."""
        
        # Student vs Teacher case
        if self.student_decks and self.teacher_decks:
            p_s = np.array(self.student_rates) / sum(self.student_rates)
            p_t = np.array(self.teacher_rates) / sum(self.teacher_rates)
            
            decks_1 = np.random.choice(self.student_decks, size=batch_size, p=p_s).tolist()
            decks_2 = np.random.choice(self.teacher_decks, size=batch_size, p=p_t).tolist()
            return decks_1, decks_2

        # Standard league case
        p = np.array(self.rates) / sum(self.rates)
        
        # If no fixed decks, sample both players from league
        if not self.fixed_decks:
            decks_1 = np.random.choice(self.decks, size=batch_size, p=p).tolist()
            decks_2 = np.random.choice(self.decks, size=batch_size, p=p).tolist()
            return decks_1, decks_2
        
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
    test_games: int = 8
    unroll_length: int = 200
    num_buffers: int = 2
    accumulation_steps: int = 1
    update_batch_size: Optional[int] = None
    model_type: str = "transformer" # "mlp" or "transformer"
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_embed_dim: int = 64
    transformer_seq_len: int = 16
    timeout_reward: Optional[float] = None
    seed: int = 42
    num_workers: int = 4


def v_trace(
    v_tm1: jnp.ndarray, # (T, B)
    v_tp1: jnp.ndarray, # (T, B)
    r_t: jnp.ndarray,   # (T, B)
    rho_t: jnp.ndarray, # (T, B)
    gamma: float = 0.99,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes V-trace targets and advantages.
    Assumes v_tm1 contains values for steps 0 to T-1.
    Assumes v_tp1 contains values for steps 1 to T (bootstrapped).
    """
    T = v_tm1.shape[0]

    # Clip importance weights
    rho_bar_t = jnp.minimum(rho_t, clip_rho_threshold)
    c_bar_t = jnp.minimum(rho_t, clip_pg_rho_threshold)

    def scan_body(carry, x):
        acc = carry
        rho_bar, c_bar, r, v_curr, v_next = x

        delta = rho_bar * (r + gamma * v_next - v_curr)
        acc = delta + gamma * c_bar * acc
        return acc, acc + v_curr

    # Scan from T-1 down to 0
    xs = (rho_bar_t, c_bar_t, r_t, v_tm1, v_tp1)
    xs_rev = jax.tree_util.tree_map(lambda x: x[::-1], xs)

    init_acc = jnp.zeros_like(v_tm1[0])
    _, vs_rev = jax.lax.scan(scan_body, init_acc, xs_rev)

    vs = vs_rev[::-1]

    # Advantages for Policy Gradient (uses v_tp1 as the lookahead value)
    rho_pg = jnp.minimum(rho_t, clip_pg_rho_threshold)
    pg_advantages = rho_pg * (r_t + gamma * v_tp1 - v_tm1)

    return vs, pg_advantages

def merge_leaf(t_leaf, s_leaf):
    """Merges a single leaf parameter, handling shape mismatch by slicing/padding."""
    if not hasattr(t_leaf, 'shape') or not hasattr(s_leaf, 'shape'):
         return s_leaf
         
    if t_leaf.shape == s_leaf.shape:
        return s_leaf
        
    logging.warning(f"Shape mismatch: target {t_leaf.shape}, source {s_leaf.shape}. Merging...")
    
    out = t_leaf
    ndim = out.ndim
    if s_leaf.ndim != ndim:
         logging.warning(f"Rank mismatch: {t_leaf.shape} vs {s_leaf.shape}. Returning target implementation.")
         return t_leaf
         
    s_slice = tuple(slice(0, min(dc, do)) for dc, do in zip(out.shape, s_leaf.shape))
    try:
         out = out.at[s_slice].set(s_leaf[s_slice])
         return out
    except Exception as e:
         logging.error(f"Merge failed: {e}. Returning target.")
         return t_leaf

def merge_recursive(target, source):
    """Recursively merges target and source parameters/state."""
    if isinstance(target, dict) and isinstance(source, dict):
        out = {}
        for k, v_t in target.items():
            if k in source:
                out[k] = merge_recursive(v_t, source[k])
            else:
                out[k] = v_t 
        return out
    
    # Handle NamedTuple and active tuples/lists
    if isinstance(target, (list, tuple)) and isinstance(source, (list, tuple)):
         # Handle NamedTuple
         if hasattr(target, '_fields') and hasattr(source, '_fields'):
             if type(target) == type(source):
                  return type(target)(*(merge_recursive(getattr(target, f), getattr(source, f)) for f in target._fields))
         
         if len(target) != len(source):
              return target
         
         return type(target)(*(merge_recursive(t, s) for t, s in zip(target, source)))

    if hasattr(target, 'shape') and hasattr(source, 'shape'):
         return merge_leaf(target, source)
         
    return source

def loss_fn(params, fixed_params, batch, apply_fn, config: RNaDConfig, alpha_rnad: float):
    obs = batch['obs'] # (T, B, dim)
    mask = batch.get('mask') # (T, B, action_dim)
    act = batch['act'] # (T, B)
    rew = batch['rew'] # (T, B, 2) -- Absolute rewards [r_p1, r_p2]
    log_prob_behavior = batch['log_prob'] # (T, B)
    bootstrap_value = batch.get('bootstrap_value', None) # (B,) -- Value [v_learner]
    player_id = batch['player_id'] # (T, B) -- Learner ID (0 or 1)
    
    T, B, _ = obs.shape
    flat_obs = obs.reshape(-1, obs.shape[-1])
    flat_mask = mask.reshape(-1, mask.shape[-1]) if mask is not None else None

    # Forward pass (Current Policy)
    logits, values = apply_fn(params, jax.random.PRNGKey(0), flat_obs, mask=flat_mask)
    logits = logits.reshape(T, B, -1)
    values = values.reshape(T, B) # (T, B) -- [v_learner]

    # Forward pass (Fixed Policy)
    fixed_logits, _ = apply_fn(fixed_params, jax.random.PRNGKey(0), flat_obs, mask=flat_mask)
    fixed_logits = fixed_logits.reshape(T, B, -1)

    # Log Probs
    log_probs = jax.nn.log_softmax(logits)
    fixed_log_probs = jax.nn.log_softmax(fixed_logits)

    # Select log prob of taken actions
    act_one_hot = jax.nn.one_hot(act, logits.shape[-1])
    log_pi_a = jnp.sum(log_probs * act_one_hot, axis=-1)
    log_pi_fixed_a = jnp.sum(fixed_log_probs * act_one_hot, axis=-1)

    # Determine if it's the learner's turn
    # learner_id is provided by player_id
    # current_player is whose turn it actually was
    current_player = batch.get('current_player') # (T, B)
    if current_player is None:
        # Fallback (should be present now)
        current_player = player_id 
    
    is_my_turn = (current_player == player_id) # (T, B) boolean

    # Regularization Penalty (applied to acting player)
    penalty = alpha_rnad * (log_pi_a - log_pi_fixed_a)
    
    # Select Learner's Reward
    # If learner_id == 0 -> r_p1 (index 0)
    # If learner_id == 1 -> r_p2 (index 1)
    # Note: rew is (T, B, 2)
    # We select based on player_id which is (T, B)
    r_learner = jnp.where(player_id == 0, rew[..., 0], rew[..., 1])
    
    # Penalty is applied to reward ONLY if I took the action (is_my_turn).
    # If not my turn, no penalty (deviation by opponent doesn't cost me).
    penalty_masked = jnp.where(is_my_turn, penalty, 0.0)
    
    r_reg = r_learner - penalty_masked

    # Importance Sampling Weights
    log_rho = log_pi_a - log_prob_behavior
    
    # If not my turn, force rho = 1.0 (log_rho = 0.0)
    # This treats off-turn transitions as environment dynamics from learner's perspective.
    log_rho = jnp.where(is_my_turn, log_rho, 0.0)
    rho = jnp.exp(log_rho)

    # --- V-trace on Learner Stream ---
    
    # Bootstrap setup
    if bootstrap_value is None:
        bootstrap_value = jnp.zeros((B,))
    
    # Prepare v_next
    v_next = jnp.concatenate([values[1:], bootstrap_value[None, :]], axis=0)
    
    vs, pg_adv = v_trace(
        v_tm1=values,
        v_tp1=v_next,
        r_t=r_reg,
        rho_t=rho,
        gamma=config.discount_factor,
        clip_rho_threshold=config.clip_rho_threshold,
        clip_pg_rho_threshold=config.clip_pg_rho_threshold,
    )

    # Value Loss
    value_loss = 0.5 * jnp.mean((jax.lax.stop_gradient(vs) - values) ** 2)

    # Policy Loss
    # Maximize advantage on MY turns.
    # Mask policy loss -> is_my_turn
    raw_policy_loss = -log_pi_a * jax.lax.stop_gradient(pg_adv)
    policy_loss = jnp.mean(jnp.where(is_my_turn, raw_policy_loss, 0.0))

    # Metrics
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    mean_entropy = jnp.mean(jnp.where(is_my_turn, entropy, 0.0))

    kl = jnp.sum(probs * (log_probs - fixed_log_probs), axis=-1)
    mean_kl = jnp.mean(jnp.where(is_my_turn, kl, 0.0))

    # Metric variance
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
                "seed": config.seed,
                "max_game_length": config.unroll_length
            }
        )
        self.config = config
        self.num_actions = self.game.num_distinct_actions()
        self.obs_shape = self.game.observation_tensor_shape()

        # Load embedding matrix
        emb_path = "card_embeddings.npy"
        if os.path.exists(emb_path):
            self.embedding_matrix = jnp.array(np.load(emb_path))
            logging.info(f"Loaded embedding matrix from {emb_path}, shape: {self.embedding_matrix.shape}")
        else:
            logging.warning(f"Embedding matrix not found at {emb_path}. Using zero matrix.")
            self.embedding_matrix = jnp.zeros((10000, 26))

        def forward(x, mask=None):
            if config.model_type == "transformer":
                net = CardTransformerNet(
                    num_actions=self.num_actions,
                    embedding_matrix=self.embedding_matrix,
                    hidden_size=config.transformer_embed_dim,
                    num_blocks=config.transformer_layers,
                    num_heads=config.transformer_heads,
                )
            else:
                net = DeckGymNet(
                    num_actions=self.num_actions,
                    hidden_size=config.hidden_size,
                    num_blocks=config.num_blocks
                )
            return net(x, mask=mask)

        self.network = hk.transform(forward)
        self.params = None
        self.fixed_params = None
        self.opt_state = None
        
        # Determine SGD batch size and micro-batching
        self.sgd_batch_size = config.update_batch_size or config.batch_size
        
        if self.sgd_batch_size > config.batch_size:
            raise ValueError(f"update_batch_size ({self.sgd_batch_size}) cannot be larger than batch_size ({config.batch_size}). "
                             "Micro-batching is intended to use smaller batches for updates to save memory.")

        if config.batch_size % self.sgd_batch_size != 0:
            raise ValueError(f"batch_size ({config.batch_size}) must be divisible by update_batch_size ({self.sgd_batch_size}).")

        self.num_micro_batches = config.batch_size // self.sgd_batch_size
        total_accumulation_steps = (config.batch_size * config.accumulation_steps) // self.sgd_batch_size
        logging.info(f"Using sgd_batch_size: {self.sgd_batch_size}, num_micro_batches_per_batch: {self.num_micro_batches}, total_accumulation_steps: {total_accumulation_steps}")

        base_optimizer = optax.adam(learning_rate=config.learning_rate)
        if total_accumulation_steps > 1:
            self.optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=total_accumulation_steps)
        else:
            self.optimizer = base_optimizer

        # JIT the update function
        self._update_fn = jax.jit(partial(self._update_pure, apply_fn=self.network.apply, config=self.config, optimizer=self.optimizer))

        # Batched Simulator in Rust


        # JIT the inference function for speed
        @jax.jit
        def _inference_fn(params, obs, mask=None):
            # TPU optimization: cast to bfloat16
            obs = obs.astype(jnp.bfloat16)
            logits, _ = self.network.apply(params, jax.random.PRNGKey(0), obs, mask=mask)
            # Logits should remain float32 for stability usually, but let's check if output is already f32.
            # Usually network outputs f32 if not configured otherwise.
            return logits
        self._inference_fn = _inference_fn

        # JIT the value function for bootstrapping
        @jax.jit
        def _value_fn(params, obs):
            # TPU optimization: cast to bfloat16
            obs = obs.astype(jnp.bfloat16)
            _, values = self.network.apply(params, jax.random.PRNGKey(0), obs)
            return values
        self._value_fn = _value_fn

    def save_checkpoint(self, path: str, step: int, metadata: Dict[str, Any] = None):
        data = {
            'params': self.params,
            'fixed_params': self.fixed_params,
            'opt_state': self.opt_state,
            'step': step,
            'config': self.config,
            'metadata': metadata or {}
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Checkpoint saved to {path} at step {step}")

    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, Any]]:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        logging.info(f"Loading checkpoint from {path}...")
        
        # 1. Initialize fresh params/opt_state to get correct shapes/structure for current code
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, *self.obs_shape))
        dummy_mask = jnp.zeros((1, self.num_actions))
        
        current_params = self.network.init(rng, dummy_obs, mask=dummy_mask)
        current_opt_state = self.optimizer.init(current_params)
        
        # 2. Merge loaded params into current params
        self.params = merge_recursive(current_params, data['params'])
        self.fixed_params = merge_recursive(current_params, data['fixed_params'])
        
        # 3. Merge opt_state
        try:
             self.opt_state = merge_recursive(current_opt_state, data['opt_state'])
        except Exception as e:
             logging.warning(f"Failed to merge opt_state: {e}. Resetting optimizer state.")
             self.opt_state = current_opt_state
             
        step = data['step']
        metadata = data.get('metadata', {})
        
        logging.info(f"Checkpoint loaded, resuming from step {step}")
        return step, metadata

    def init(self, key):
        dummy_obs = jnp.zeros((1, *self.obs_shape))
        dummy_mask = jnp.zeros((1, self.num_actions))
        self.params = self.network.init(key, dummy_obs, mask=dummy_mask)
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
            'mean_entropy': ent,
            'mean_kl': kl,
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
        # Optimization: eliminate num_buffers loop and switch to streaming processing
        # ※ config.num_buffers is ignored here and effectively treated as 1
        # ※ Increase config.accumulation_steps as needed
        
        batch_size = self.config.batch_size
        max_len = self.config.unroll_length

        # === 1. Deck sampling ===
        if self.config.league_config:
            decks_1, decks_2 = self.config.league_config.sample_decks(batch_size)
        else:
            decks_1 = [self.config.deck_id_1] * batch_size
            decks_2 = [self.config.deck_id_2] * batch_size

        # === 2. Load past model (Self-Play opponent) ===
        use_past_self_play = False
        past_params = None
        if self.config.past_self_play:
            checkpoint_dir = getattr(self.config, 'checkpoint_dir', 'checkpoints')
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
                if checkpoints:
                    use_past_self_play = True
                    chosen_cp = np.random.choice(checkpoints)
                    try:
                        with open(os.path.join(checkpoint_dir, chosen_cp), 'rb') as f:
                            data = pickle.load(f)
                            past_params = data['params']
                    except Exception as e:
                        logging.warning(f"Failed to load past checkpoint {chosen_cp}: {e}")
                        use_past_self_play = False

        # === 3. Simulator initialization ===
        sim = deckgym.PyBatchedSimulator(
            self.config.deck_id_1,
            self.config.deck_id_2,
            batch_size,
            self.config.win_reward,
            self.config.point_reward,
            self.config.damage_reward
        )
        
        # Reset returns (obs_0, obs_1), cp, mask
        (initial_obs_0, initial_obs_1), initial_current_players, initial_mask = sim.reset(
            seed=int(jax.random.randint(key, (), 0, 1000000)),
            deck_ids_1=decks_1,
            deck_ids_2=decks_2
        )

        # === Assign Learner ID per Episode (0 or 1) ===
        # Randomly assign whether the Learner network represents Player 0 or Player 1 for this game.
        # This determines which observation stream we follow for the "Subjective Evaluation".
        learner_ids = np.random.randint(0, 2, size=batch_size) 
        
        # Select initial subjective observation
        initial_obs_0_np = np.array(initial_obs_0)
        initial_obs_1_np = np.array(initial_obs_1)
        
        # Choose obs based on learner_id (0 or 1)
        # learner_ids is (B,)
        # initial_obs_0_np is (B, Obs)
        # We need to broadcast learner_ids to (B, Obs) or compatible.
        # learner_ids[:, None] -> (B, 1). This broadcasts against (B, Obs) to produce (B, Obs).
        current_obs_learner = np.where(learner_ids[:, None] == 0, initial_obs_0_np, initial_obs_1_np)
        
        # Verify shape to prevent issues
        if current_obs_learner.shape != initial_obs_0_np.shape:
            logging.error(f"Shape mismatch in obs selection! LearnerObs: {current_obs_learner.shape}, InitObs: {initial_obs_0_np.shape}")
        
        # State management buffer (single)
        sb = {
            'obs_0': initial_obs_0_np,
            'obs_1': initial_obs_1_np,
            'current_players': np.array(initial_current_players),
            'current_mask': np.array(initial_mask),
            'active_mask': np.ones(batch_size, dtype=bool),
            'episode_outcomes': np.zeros(batch_size, dtype=int),
            'outcome_recorded': np.zeros(batch_size, dtype=bool),
            'episode_lengths': np.zeros(batch_size, dtype=int)
        }

        # Trajectory buffer (single)
        tb = {
            'obs_buf': np.zeros((max_len, batch_size, *self.obs_shape), dtype=np.float32),
            'mask_buf': np.zeros((max_len, batch_size, self.num_actions), dtype=np.float32),
            'act_buf': np.zeros((max_len, batch_size), dtype=np.int32),
            'rew_buf': np.zeros((max_len, batch_size, 2), dtype=np.float32), 
            'log_prob_buf': np.zeros((max_len, batch_size), dtype=np.float32),
            'player_buf': np.zeros((max_len, batch_size), dtype=np.int32), # This will now store LEARNER_ID
            'current_player_buf': np.zeros((max_len, batch_size), dtype=np.int32), # This stores WHOSE TURN
            'ptrs': np.zeros(batch_size, dtype=int)
        }

        agent_ids = np.zeros(batch_size, dtype=int)
        if use_past_self_play:
            # If using past self play, we need to know if the opponent is the past model
            # Let's say learner_id always uses `self.params`.
            # Opponent uses `past_params`.
            # If current_player == learner_id -> self.params.
            # If current_player != learner_id -> past_params.
            # This logic is handled in the loop.
            pass

        # === 4. First inference (Learner Stream) ===
        # Continuous Subjective Evaluation: Always running Learner inference on Learner's Obs
        # Send request to GPU
        future_learner = self._inference_fn(self.params, current_obs_learner, mask=sb['current_mask'])

        # === 5. Main loop ===
        for i_step in range(max_len):
             if not sb['active_mask'].any():
                 break

             # A. Inference result reception (Subjective/Learner)
             logits_learner = future_learner
             logits_learner.block_until_ready()
             logits_learner_np = np.array(logits_learner)

             # B. Inference for Actor (Who determines the action?)
             # We need to know who is acting to decide which network to query for the ACTION.
             # obs_active: Observation of the player whose turn it is.
             # sb['current_players'] is (B,). Expand to (B, 1) to broadcast with (B, ObsDim).
             obs_active = np.where(sb['current_players'][:, None] == 0, sb['obs_0'], sb['obs_1'])
             
             # Determine Policy for Action Selection
             # If current_player == learner_id: Use logits_learner (which is computed on obs_learner == obs_active)
             # If current_player != learner_id: Use Opponent Policy.
             
             # Optimization: If we are in self-play (not past), opponent uses same params.
             # But obs_active != obs_learner (usually), so we need another inference call anyway.
             
             is_my_turn = (sb['current_players'] == learner_ids)
             
             future_actor = None
             if use_past_self_play:
                 # Opponent uses past_params
                 # We only need to run actor inference if it is NOT my turn.
                 # If it IS my turn, logits_learner is sufficient (since obs_learner == obs_active).
                 # Wait, we need to mix.
                 future_actor = self._inference_fn(past_params, obs_active, mask=sb['current_mask'])
             else:
                 # Standard Self-Play: Opponent uses same params
                 future_actor = self._inference_fn(self.params, obs_active, mask=sb['current_mask'])
             
             logits_actor = future_actor
             logits_actor_np = np.array(logits_actor)
             
             # Combine Logits for Step
             # If is_my_turn -> logits_learner
             # Else -> logits_actor
             final_logits = np.where(is_my_turn[:, None], logits_learner_np, logits_actor_np)

             # C. Simulation execution (Rust)
             # step returns ((obs_0, obs_1), rewards, ...)
             ((next_obs_0, next_obs_1), rewards, dones, _, valid_mask, actions, log_probs, next_current_players, next_mask) = \
                 sim.sample_and_step(final_logits)
             
             # Calculate rewards for stats
             rewards_np = np.array(rewards) # (B, 2)
             
             # === D. Pipeline optimization and data saving ===
             
             # Save current state for saving
             old_obs_learner = current_obs_learner
             
             # We store the mask corresponding to the LEARNER?
             # Actually mask is usually valid actions for the ACTIVE player.
             # For off-turn, mask is irrelevant for policy loss, but maybe useful for architecture.
             # Let's just use the simulator's mask (which is for active player) or old mask.
             old_mask = sb['current_mask']
             old_active = sb['active_mask'].copy()
             old_players = sb['current_players'].copy() 

             # Update to next state
             nb_obs_0_np = np.array(next_obs_0)
             nb_obs_1_np = np.array(next_obs_1)
             
             sb['obs_0'] = nb_obs_0_np
             sb['obs_1'] = nb_obs_1_np
             
             # Select next learner obs
             current_obs_learner = np.where(learner_ids[:, None] == 0, nb_obs_0_np, nb_obs_1_np)
             
             sb['current_players'] = np.array(next_current_players)
             sb['current_mask'] = np.array(next_mask)
             
             # Valid determination
             step_valid = valid_mask.astype(bool)
             
             # Active determination (deactivate if completed or invalid)
             still_active = old_active & (~dones) & step_valid
             sb['active_mask'] = still_active

             # Next inference immediately (GPU runs while CPU saves)
             # Always run learner inference
             if sb['active_mask'].any():
                 future_learner = self._inference_fn(self.params, current_obs_learner, mask=sb['current_mask'])

             # Data saving (vectorized for speed)
             # We save ALL steps now (Subjective Stream)
             write_mask = old_active & step_valid
             
             write_indices = np.where(write_mask)[0]
             if write_indices.size > 0:
                 current_ptrs = tb['ptrs'][write_indices]
                 valid_ptr_mask = current_ptrs < max_len
                 final_indices = write_indices[valid_ptr_mask]
                 final_ptrs = current_ptrs[valid_ptr_mask]
                 
                 if final_indices.size > 0:
                     tb['obs_buf'][final_ptrs, final_indices] = old_obs_learner[final_indices]
                     tb['mask_buf'][final_ptrs, final_indices] = old_mask[final_indices]
                     tb['act_buf'][final_ptrs, final_indices] = actions[final_indices]
                     tb['rew_buf'][final_ptrs, final_indices] = rewards_np[final_indices] # Absolute rewards
                     # log_prob from the ACTOR (behavior policy) required for V-trace?
                     # Yes, rho = pi_learner / mu_behavior.
                     # If I acted, mu = pi_learner (so rho=1).
                     # If Opponent acted, mu = pi_opponent. pi_learner is irrelevant (masked).
                     # So we just store the log_prob returned by sample_and_step.
                     tb['log_prob_buf'][final_ptrs, final_indices] = log_probs[final_indices]
                     
                     # Store Learner ID and Current Player
                     tb['player_buf'][final_ptrs, final_indices] = learner_ids[final_indices]
                     tb['current_player_buf'][final_ptrs, final_indices] = old_players[final_indices]
                     
                     tb['ptrs'][final_indices] += 1
                     sb['episode_lengths'][final_indices] += 1

             # Record win/loss (vectorized)
             just_finished = old_active & dones
             finished_indices = np.where(just_finished)[0]
             if finished_indices.size > 0:
                 needs_record_mask = ~sb['outcome_recorded'][finished_indices]
                 rec_indices = finished_indices[needs_record_mask]
                 if rec_indices.size > 0:
                     # Use Absolute rewards for simpler stats
                     # If stats tracking expects P1 perspective, we use rewards_np[:, 0]
                     r_p1 = rewards_np[rec_indices, 0]
                     
                     outcomes = np.zeros(len(rec_indices), dtype=int)
                     
                     # P1 Win: r_p1 > 0
                     # P2 Win: r_p2 > 0 (and r_p1 < 0)
                     # Tie: r_p1 < 0 AND r_p2 < 0 (due to Rust change returning -1 for Tie)
                     
                     win_cond = r_p1 > 1e-3
                     # Check r_p2 to distinguish P2 Win (1) from Tie (-1)
                     r_p2 = rewards_np[rec_indices, 1]
                     p2_win_cond = (r_p1 < -1e-3) & (r_p2 > 1e-3)
                     
                     outcomes[win_cond] = 1 # P1 Win
                     outcomes[p2_win_cond] = 2 # P2 Win
                     # Tie remains 0 (default)
                     
                     sb['episode_outcomes'][rec_indices] = outcomes
                     sb['outcome_recorded'][rec_indices] = True
        
        # === 6. Result aggregation and return ===
        # Bootstrap values (not finished episodes' final state values)
        merged_bootstrap = np.zeros((batch_size,), dtype=np.float32) # (B,)
        if sb['active_mask'].any():
            if self.config.timeout_reward is not None:
                 vals_np = np.full((batch_size,), self.config.timeout_reward, dtype=np.float32)
            else:
                 vals = self._value_fn(self.params, current_obs_learner)
                 vals_np = np.array(vals).reshape(batch_size) # [v_learner]
            
            active_idxs = np.where(sb['active_mask'])[0]
            if active_idxs.size > 0:
                merged_bootstrap[active_idxs] = vals_np[active_idxs]

        # Stats
        p1_wins = np.sum(sb['episode_outcomes'] == 1)
        p2_wins = np.sum(sb['episode_outcomes'] == 2)
        ties = np.sum(sb['episode_outcomes'] == 0)
        
        batch_out = {
            'obs': np.array(tb['obs_buf']),
            'mask': np.array(tb['mask_buf']),
            'act': np.array(tb['act_buf']),
            'rew': np.array(tb['rew_buf']),
            'log_prob': np.array(tb['log_prob_buf']),
            'player_id': np.array(tb['player_buf']),
            'current_player': np.array(tb['current_player_buf']),
            'bootstrap_value': np.array(merged_bootstrap),
            'stats': {
                'mean_episode_length': np.mean(sb['episode_lengths']),
                'max_episode_length': np.max(sb['episode_lengths']),
                'p1_win_rate': p1_wins / batch_size,
                'decisive_rate': (p1_wins + p2_wins) / batch_size,
                'tie_rate': ties / batch_size
            }
        }
        return batch_out

def slice_batch(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Slices a trajectory batch for micro-batching."""
    sliced = {}
    for k, v in batch.items():
        if k == 'stats':
            sliced[k] = v
        elif k == 'bootstrap_value':
            sliced[k] = v[start:end]
        elif isinstance(v, (jnp.ndarray, np.ndarray)):
            # T, B, ... -> slice at dim 1
            sliced[k] = v[:, start:end]
        else:
            sliced[k] = v
    return sliced

import queue
import threading

class TrajectoryGenerator:
    """
    Many workers generate trajectories in parallel to maximize GPU utilization.
    """
    def __init__(self, learner, num_workers=4, max_queue_size=20):
        self.learner = learner
        self.num_workers = num_workers 
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.threads = []

    def start(self):
        logging.info(f"Starting TrajectoryGenerator with {self.num_workers} workers.")
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self.stop_event.set()
        # Drain the queue to unblock workers waiting on queue.put()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass
            
        # Optional: Wait for threads to finish their current loop
        for t in self.threads:
            t.join(timeout=1.0)

    def get_batch(self):
        return self.queue.get()

    def _worker(self, worker_id):
        seed_base = self.learner.config.seed + (worker_id * 10000)
        rng = np.random.RandomState(seed_base)
        
        while not self.stop_event.is_set():
            try:
                step_seed = rng.randint(0, 2**30)
                key = jax.random.PRNGKey(step_seed)
                
                batch = self.learner.generate_trajectories(key)
                
                self.queue.put(batch)
                
            except Exception as e:
                logging.error(f"Worker {worker_id} crashed: {e}")
                time.sleep(1.0)

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
            # Use deterministic seed sequence for evaluation based on global seed
            games = [PyGameState(d1_path, d2_path, config.seed + played + i) for i in range(current_batch)]
            
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
                    mask_p1 = np.zeros((len(p1_indices), learner.num_actions), dtype=np.float32)
                    for i, idx in enumerate(p1_indices):
                        mask_p1[i, games[idx].legal_actions()] = 1.0
                    logits_p1 = learner._inference_fn(current_params_p1, obs_p1, mask=mask_p1)
                    batch_logits[p1_indices] = np.array(logits_p1)
                
                if p2_indices:
                    obs_p2 = current_obs[p2_indices]
                    mask_p2 = np.zeros((len(p2_indices), learner.num_actions), dtype=np.float32)
                    for i, idx in enumerate(p2_indices):
                        mask_p2[i, games[idx].legal_actions()] = 1.0
                    logits_p2 = learner._inference_fn(baseline_params_p2, obs_p2, mask=mask_p2)
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

def train_loop(config: RNaDConfig, experiment_manager: Optional[Any] = None, checkpoint_dir: str = "checkpoints", resume_checkpoint: Optional[str] = None, trial: Optional[optuna.Trial] = None):
    learner = RNaDLearner("deckgym_ptcgp", config)
    learner.init(jax.random.PRNGKey(config.seed))

    start_step = 0

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            start_step, _ = learner.load_checkpoint(resume_checkpoint)
            start_step += 1 # Resume from next step
        else:
            logging.error(f"Checkpoint {resume_checkpoint} not found!")
            return
    else:
        # Check for latest checkpoint in directory
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            start_step, _ = learner.load_checkpoint(latest_checkpoint)
            start_step += 1
            logging.info(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")

    if experiment_manager:
        experiment_manager.log_params(config)

    logging.info(f"Starting training loop from step {start_step}...")

    # Save interval
    save_interval = config.save_interval if hasattr(config, 'save_interval') else 1000

    # Start Trajectory Generator
    generator = TrajectoryGenerator(learner, num_workers=config.num_workers, max_queue_size=20)
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
                try:
                    jax.profiler.stop_trace()
                except Exception:
                    pass

        metrics_accum = {}
        start_time = time.time()

        num_updates_done = 0
        for i_acc in range(config.accumulation_steps):
            # 1. Get batch from background generator
            batch = generator.get_batch()
            
            # 2. Update in micro-batches
            for i_micro in range(learner.num_micro_batches):
                start = i_micro * learner.sgd_batch_size
                end = start + learner.sgd_batch_size
                micro_batch = slice_batch(batch, start, end)

                metrics = learner.update(micro_batch, step)
                
                for k, v in metrics.items():
                    if isinstance(v, (int, float, jnp.ndarray)):
                        metrics_accum[k] = metrics_accum.get(k, 0) + v
                num_updates_done += 1
        
        num_updates = num_updates_done or 1

        end_time = time.time()

        # Average metrics
        metrics = {k: v / num_updates for k, v in metrics_accum.items()}

        # Calculate SPS
        # metrics['mean_episode_length'] is already averaged.
        # Total samples = mean_len * batch_size * accumulation_steps
        # batch_size here is the simulation batch size.
        total_samples = metrics.get('mean_episode_length', 0) * config.batch_size * config.accumulation_steps
        step_time = end_time - start_time
        sps = total_samples / (step_time + 1e-6)
        metrics['sps'] = sps
        
        # Log config values as metrics for tuning tracking
        metrics['batch_size'] = float(config.batch_size)
        metrics['accumulation_steps'] = float(config.accumulation_steps)
        metrics['update_batch_size'] = float(learner.sgd_batch_size)
        metrics['num_workers'] = float(config.num_workers)

        if step % config.log_interval == 0:
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
             metadata = {}
             if experiment_manager and hasattr(experiment_manager, 'run_id'):
                 metadata['mlflow_run_id'] = experiment_manager.run_id
             learner.save_checkpoint(ckpt_path, step, metadata=metadata)
        
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

                 # HPO: C. Pruning by Win Rate
                 if trial:
                     # Report intermediate objective value
                     if "test_winrate_mean" in eval_metrics:
                         trial.report(eval_metrics["test_winrate_mean"], step)
                         if trial.should_prune():
                             logging.info(f"Step {step}: Pruning trial due to poor win rate.")
                             raise optuna.TrialPruned()

        # HPO: A. Value Network Convergence Check (Step 2000-5000)
        if trial and 2000 <= step <= 5000:
            if 'explained_variance' in metrics and metrics['explained_variance'] < 0.05:
                logging.info(f"Step {step}: Pruning trial due to low explained_variance (< 0.05).")
                raise optuna.TrialPruned()
            
        # HPO: B. Entropy Collapse Check (Step > 5000)
        if trial and step > 5000:
            # 1. Mode Collapse check
            if 'mean_entropy' in metrics and metrics['mean_entropy'] < 0.1:
                logging.info(f"Step {step}: Pruning trial due to mode collapse (entropy < 0.1).")
                raise optuna.TrialPruned()
            
            # 2. Learning Stagnation check (Entropy not decreasing significantly from start)
            # config.entropy_schedule_start is the initial target entropy
            if 'mean_entropy' in metrics and metrics['mean_entropy'] > config.entropy_schedule_start * 0.95:
                 # If entropy hasn't dropped by at least 5% after 5000 steps, it might be stuck
                 logging.info(f"Step {step}: Pruning trial due to entropy stagnation.")
                 raise optuna.TrialPruned()

    # Save final checkpoint
    final_step = config.max_steps - 1
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{final_step}.pkl")
    metadata = {}
    if experiment_manager and hasattr(experiment_manager, 'run_id'):
        metadata['mlflow_run_id'] = experiment_manager.run_id
    learner.save_checkpoint(ckpt_path, final_step, metadata=metadata)
    
    # Properly stop background threads
    logging.info("Stopping trajectory generator...")
    generator.stop()
    logging.info("Training complete.")

if __name__ == "__main__":
    train_loop(RNaDConfig())
