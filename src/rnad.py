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
    win_reward: float = 1.0
    point_reward: float = 0.0
    damage_reward: float = 0.0

def v_trace(
    v_tm1: jnp.ndarray, # (T, B)
    r_t: jnp.ndarray,   # (T, B)
    rho_t: jnp.ndarray, # (T, B)
    gamma: float = 0.99,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes V-trace targets and advantages.
    Assumes v_tm1 contains values for steps 0 to T-1.
    """
    T = v_tm1.shape[0]

    # Clip importance weights
    rho_bar_t = jnp.minimum(rho_t, clip_rho_threshold)
    c_bar_t = jnp.minimum(rho_t, clip_pg_rho_threshold)

    # We need a bootstrap value for V_T. Assuming 0 for now as episodes end.
    v_all = jnp.concatenate([v_tm1, jnp.zeros((1, v_tm1.shape[1]))], axis=0) # (T+1, B)

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
        clip_pg_rho_threshold=config.clip_pg_rho_threshold
    )

    # Value Loss
    value_loss = 0.5 * jnp.mean((jax.lax.stop_gradient(vs) - values) ** 2)

    # Policy Loss
    policy_loss = -jnp.mean(log_pi_a * jax.lax.stop_gradient(pg_adv))

    total_loss = policy_loss + value_loss

    return total_loss, (policy_loss, value_loss)

class RNaDLearner:
    def __init__(self, game_name: str, config: RNaDConfig):
        self.game = pyspiel.load_game(
            game_name,
            {
                "deck_id_1": config.deck_id_1,
                "deck_id_2": config.deck_id_2,
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

        (total_loss, (p_loss, v_loss)), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, {'total': total_loss, 'policy': p_loss, 'value': v_loss}

    def update(self, batch, step: int):
        # Anneal alpha_rnad
        progress = min(1.0, step / self.config.max_steps)
        alpha = self.config.entropy_schedule_start + progress * (self.config.entropy_schedule_end - self.config.entropy_schedule_start)

        self.params, self.opt_state, metrics = self._update_fn(
            self.params, self.fixed_params, self.opt_state, batch, alpha_rnad=alpha
        )
        metrics['alpha'] = alpha
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
        initial_obs = self.batched_sim.reset(seed=int(jax.random.randint(key, (), 0, 1000000)))
        
        # Storage per environment
        env_trajs = [{'obs': [], 'act': [], 'rew': [], 'log_prob': []} for _ in range(self.config.batch_size)]
        
        active_mask = np.ones(self.config.batch_size, dtype=bool)
        current_obs = np.array(initial_obs)

        while active_mask.any():
            # 1. Get Logits for all environments (including inactive ones, masked in Rust)
            logits = self._inference_fn(self.params, current_obs)
            logits_np = np.array(logits)

            # 2. Sample and Step in Rust (Parallelized!)
            (next_obs, rewards, dones, _, valid_mask, actions, log_probs) = \
                self.batched_sim.sample_and_step(logits_np)
            
            # 3. Store transitions
            for i in range(self.config.batch_size):
                if active_mask[i] and valid_mask[i]:
                    env_trajs[i]['obs'].append(current_obs[i])
                    env_trajs[i]['act'].append(actions[i])
                    env_trajs[i]['rew'].append(rewards[i])
                    env_trajs[i]['log_prob'].append(log_probs[i])
                    
                    if dones[i]:
                        active_mask[i] = False
                elif active_mask[i] and not valid_mask[i]:
                    # This should not happen if logic in Rust is correct
                    active_mask[i] = False

            current_obs = np.array(next_obs)

        # Pad trajectories to max length
        max_len = max(len(t['obs']) for t in env_trajs)
        if max_len == 0:
             # Should not happen if game is valid
            return {
                'obs': jnp.zeros((max_len, self.config.batch_size, *self.obs_shape)),
                'act': jnp.zeros((max_len, self.config.batch_size), dtype=np.int32),
                'rew': jnp.zeros((max_len, self.config.batch_size)),
                'log_prob': jnp.zeros((max_len, self.config.batch_size))
            }

        def pad(data, shape, dtype=np.float32, value=0):
            res = np.zeros((max_len, self.config.batch_size, *shape), dtype=dtype) + value
            for i, t in enumerate(env_trajs):
                l = len(t[data])
                if l > 0:
                    res[:l, i] = np.array(t[data])
            return res

        batch = {
            'obs': jnp.array(pad('obs', self.obs_shape)),
            'act': jnp.array(pad('act', (), dtype=np.int32)),
            'rew': jnp.array(pad('rew', ())),
            'log_prob': jnp.array(pad('log_prob', ()))
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

    for step in range(start_step, config.max_steps):
        # 1. Get batch from background generator
        batch = generator.get_batch()

        # 2. Update
        metrics = learner.update(batch, step)

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

    # Save final checkpoint
    final_step = config.max_steps - 1
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{final_step}.pkl")
    learner.save_checkpoint(ckpt_path, final_step)
    logging.info("Training complete.")

if __name__ == "__main__":
    train_loop(RNaDConfig())
