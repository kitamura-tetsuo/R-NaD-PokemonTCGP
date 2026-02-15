import argparse
import sys
import os
import json
import time
import logging
import pickle
import random
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pyspiel
import deckgym
from threading import Thread, Event
from queue import Queue, Full, Empty
from typing import NamedTuple, List, Dict, Any, Optional, Tuple

# Import project modules
from src.rnad import RNaDConfig, RNaDLearner, find_latest_checkpoint, LeagueConfig
from src.tree_viz import get_fast_state_key, extract_state_info

# Try importing tensorflow for SavedModel loading
try:
    import tensorflow as tf
except ImportError:
    tf = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MinerConfig(NamedTuple):
    checkpoint_dir: str = "checkpoints"
    output_file: str = "mined_data.jsonl"
    deck_id_1: str = "deckgym-core/example_decks/mewtwoex.txt"
    deck_id_2: str = "deckgym-core/example_decks/mewtwoex.txt"
    league_decks_student: Optional[str] = None
    league_decks_teacher: Optional[str] = None
    diagnostic_games_per_checkpoint: int = 10
    max_depth: int = 5 # Oracle search depth
    prediction_error_threshold: float = 0.5 # Squared error threshold
    value_change_threshold: float = 0.4 # Absolute change threshold
    device: str = "gpu"
    batch_size: int = 1 # Inference batch size for self-play
    seed: int = 42

class OracleSolver:
    """
    Solves a game state using Expectiminimax / DFS up to max_depth.
    Returns the exact value (-1.0 to 1.0) if determined, or None if inconclusive (depth limit reached without result).
    """
    def __init__(self, game, max_depth, ai_player):
        self.game = game
        self.max_depth = max_depth
        self.ai_player = ai_player
        self.transposition_table = {}

    def solve(self, state) -> Optional[float]:
        self.transposition_table = {}
        return self._dfs(state, 0)

    def _dfs(self, state, depth) -> Optional[float]:
        # 1. Check Terminal
        if state.is_terminal():
            # Return returns for ai_player
            return state.returns()[self.ai_player]

        # 2. Check Depth Limit
        if depth >= self.max_depth:
            # Reached max depth without definitive result.
            # According to user instruction: "Max depth is depth FROM game over".
            # So if we haven't hit game over, we can't be sure.
            return None

        # 3. Transposition Table
        state_raw = state.rust_game.get_state()
        pending_chance = state._pending_stochastic_action if hasattr(state, "_pending_stochastic_action") else None
        state_key = get_fast_state_key(state_raw, pending_chance)
        
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        # 4. Expansion
        value = None

        if state.is_chance_node():
            # Chance node: Average of children
            outcomes = state.chance_outcomes()
            expected_value = 0.0
            total_prob = 0.0
            
            # Optimization: Try to solve all children. If ANY returns None, the Chance node is None (cannot be strictly determined)
            # Alternatively, if probabilities are small enough, maybe we could ignore? But strictly speaking no.
            
            can_solve = True
            for action, prob in outcomes:
                child = state.clone()
                child.apply_action(action)
                v = self._dfs(child, depth + 1)
                if v is None:
                    can_solve = False
                    break
                expected_value += v * prob
                total_prob += prob

            if can_solve:
                value = expected_value / total_prob # Should sum to 1, but safe-guard
            else:
                value = None

        else:
            # Decision node
            current_player = state.current_player()
            legal_actions = state.legal_actions()
            
            # Optimization: Heuristic pruning or move ordering could go here.
            # For now, simple strict minimax.
            
            best_val = -float('inf') if current_player == self.ai_player else float('inf')
            can_solve_any = False # If we can solve at least one move that leads to win, we are good?
            # Actually::
            # If MAX player: if there is a move that leads to +1.0 (proven), then Value is +1.0. We don't need to know others.
            # If MIN player: if there is a move that leads to -1.0 (proven for AI), then Value is -1.0 (optimal opponent kills us).
            
            # If we explore a move and it returns None (unknown), we treat it as ?
            # MAX node: max(v1, v2, ...). If v1=1.0, we don't care if v2 is None. Result is 1.0.
            # If v1=None, v2=-1.0. We don't know if v1 is better. Result is None (or at least -1.0?).
            # Let's keep it simple: If we can't resolve all relevant children, we might return None, 
            # UNLESS we find a winning line for the current player.
            
            has_unknown = False
            
            for action in legal_actions:
                child = state.clone()
                child.apply_action(action)
                v = self._dfs(child, depth + 1)
                
                if current_player == self.ai_player: # MAX
                    if v == 1.0:
                        value = 1.0
                        has_unknown = False # Found killer move
                        break
                    if v is None:
                        has_unknown = True
                    else:
                        best_val = max(best_val, v)
                else: # MIN (Opponent behavior)
                    # Use standard Minimax (assuming opponent plays optimally to beat us)
                    # Note: R-NaD usually assumes opponent plays by policy? 
                    # But Oracle usually implies perfect play. "Essential failure" implies we missed a winning line or walked into a losing one.
                    # We'll use Minimax for Oracle.
                    if v == -1.0: # Opponent can force us to lose
                        # (out of AI perspective, value is -1)
                        # Actually if opponent is p2, and AI is p1. p1 gets -1.
                        value = -1.0 # (Eval is always from AI perspective here? OpenSpiel returns[player])
                        # Wait, state.returns()[ai_player]. 
                        # If ai_player=0. P1 wins -> 1.0.
                        # If current_player=1 (opponent). They want to minimize P1 return.
                        # So yes, if they find a move giving P1 -1.0 (P2 win), they will take it.
                        best_val = -1.0
                        has_unknown = False
                        break
                    if v is None:
                        has_unknown = True
                    else:
                        best_val = min(best_val, v)
            
            if value is None: # Loop didn't break with definitive result
                if has_unknown:
                    value = None
                else:
                    value = best_val

        self.transposition_table[state_key] = value
        return value


class Miner:
    def __init__(self, config: MinerConfig):
        self.config = config
        self.running = True
        self.tf_module = None # For SavedModel
        self.predict_fn = None # Abstraction
        
        # JAX Config
        if config.device == 'cpu':
            jax.config.update("jax_platform_name", "cpu")
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        
        # Initialize Game (Default)
        self.gametype = "deckgym_ptcgp"
        self.game = pyspiel.load_game(
            self.gametype,
            {
                "deck_id_1": config.deck_id_1,
                "deck_id_2": config.deck_id_2,
                "seed": config.seed,
                "max_game_length": 200 # Default
            }
        )
        
        # League Config
        self.league = None
        if config.league_decks_student or config.league_decks_teacher:
            logging.info("Loading League Configuration...")
            # If one is missing, we use it for both? Or just use default for other.
            # LeagueConfig.from_csv handles None gracefully
            self.league = LeagueConfig.from_csv(config.league_decks_student, config.league_decks_teacher)
        
        # Learner Placeholder (JAX)
        # Check if checkpoint exists to load config
        initial_rnad_config = RNaDConfig(
            deck_id_1=config.deck_id_1,
            deck_id_2=config.deck_id_2,
            model_type="transformer", # Assumption
            transformer_embed_dim=64, # Default
            transformer_layers=2
        )

        # Checkpoint or SavedModel detection
        # If config.checkpoint_dir is actually a direct file path or SavedModel dir, handle it
        # But name is checkpoint_dir. 
        # Logic: If it looks like a SavedModel (contains saved_model.pb), we act differently.
        # BUT: checkpoint_dir is usually a directory watched for NEW checkpoints.
        # If user passes a specific SavedModel path via CLI (using --checkpoint arg in script?), 
        # let's assume if it points to a specific location we treat it as single-shot or watch?
        # The user shell script passed --checkpoint "saved_model/clob/600".
        # So we should probably treat that as the source.
        
        # NOTE: The argument in miner.py is --checkpoint_dir. 
        # If the user passes a specific path, we should probably check if it IS the checkpoint or a dir.
        
        latest = find_latest_checkpoint(config.checkpoint_dir)
        # If config.checkpoint_dir is a SavedModel, find_latest won't find .pkl
        
        is_saved_model = False
        if os.path.isdir(config.checkpoint_dir) and "saved_model.pb" in os.listdir(config.checkpoint_dir):
            is_saved_model = True
        
        # Try to load config from pickle if available
        if latest and not is_saved_model:
            try:
                logging.info(f"Peeking at checkpoint {latest} for config...")
                with open(latest, 'rb') as f:
                    ckpt_data = pickle.load(f)
                    if 'config' in ckpt_data:
                        loaded_cfg = ckpt_data['config']
                        logging.info(f"Examples of loaded config: embed_dim={loaded_cfg.transformer_embed_dim}")
                        
                        initial_rnad_config = initial_rnad_config._replace(
                            model_type=getattr(loaded_cfg, 'model_type', 'transformer'),
                            transformer_embed_dim=loaded_cfg.transformer_embed_dim,
                            transformer_layers=loaded_cfg.transformer_layers,
                            transformer_heads=loaded_cfg.transformer_heads,
                            transformer_seq_len=loaded_cfg.transformer_seq_len,
                            hidden_size=loaded_cfg.hidden_size,
                            num_blocks=loaded_cfg.num_blocks,
                            unroll_length=loaded_cfg.unroll_length
                        )
            except Exception as e:
                logging.warning(f"Failed to load config from checkpoint: {e}")

        self.learner = RNaDLearner("deckgym_ptcgp", initial_rnad_config)
        
        # Output File
        if not os.path.exists(config.output_file):
            with open(config.output_file, 'w') as f:
                pass # Create file
                
        self.known_checkpoints = set()

    def load_weights(self, path):
        logging.info(f"Loading weights from: {path}")
        
        # Check if SavedModel
        if os.path.isdir(path) and "saved_model.pb" in os.listdir(path):
            if tf is None:
                logging.error("TensorFlow not installed/importable, cannot load SavedModel.")
                return
                
            logging.info("Detected SavedModel format. Loading with TensorFlow...")
            try:
                loaded = tf.saved_model.load(path)
                self.tf_module = loaded
                
                # Define wrapper
                def tf_predict(obs_np):
                    # obs_np: (Batch, Dim)
                    # TF expects tensor
                    # The SavedModel likely has a 'predict' signature or __call__
                    
                    # Check signature
                    # The export script defines: predict(self, obs) returning {"policy": ..., "value": ...}
                    # Or __call__(self, obs) returning (logits, values)
                    
                    out = self.tf_module.predict(obs_np)
                    if isinstance(out, dict):
                        return out['policy'].numpy(), out['value'].numpy()
                    else:
                        # Fallback if signature differs
                        return out
                
                self.predict_fn = tf_predict
                logging.info("SavedModel loaded successfully.")
                
            except Exception as e:
                logging.error(f"Failed to load SavedModel: {e}")
                
        else:
            # Assume Pickle Checkpoint (JAX)
            try:
                self.learner.load_checkpoint(path)
                
                def jax_predict(obs_np):
                    # self.learner._inference_fn and _value_fn return jax arrays, need blocking/numpy conv
                    # obs_np: (Batch, Dim)
                    
                    logits = self.learner._inference_fn(self.learner.params, obs_np).block_until_ready()
                    value = self.learner._value_fn(self.learner.params, obs_np).block_until_ready()
                    return np.array(logits), np.array(value)

                self.predict_fn = jax_predict
                
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")

    def run_diagnostic_games(self, num_games):
        """
        Runs self-play games and returns trajectories of interests.
        Using self.predict_fn
        """
        rng = np.random.RandomState(self.config.seed + int(time.time()))
        
        interesting_data = []
        
        for i in range(num_games):
            
            # League Sampling
            d1, d2 = self.config.deck_id_1, self.config.deck_id_2
            if self.league:
                # Sample 1 pair
                l_d1, l_d2 = self.league.sample_decks(1)
                d1, d2 = l_d1[0], l_d2[0]
                
                # Reload game with new decks
                # This is a bit expensive to reload game every time?
                # PySpiel load_game is usually fast.
                self.game = pyspiel.load_game(
                    self.gametype,
                    {
                        "deck_id_1": d1,
                        "deck_id_2": d2,
                        "seed": self.config.seed + i,
                        "max_game_length": 200
                    }
                )

            # Play one game
            state = self.game.new_initial_state()
            trajectory = [] # List of (state, value_pred, action)
            
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes)
                    action = rng.choice(action_list, p=prob_list)
                    state.apply_action(action)
                else:
                    curr_p = state.current_player()
                    obs = state.observation_tensor(curr_p)
                    obs_np = np.array(obs)[None, ...] # Batch 1
                    
                    if self.predict_fn:
                        logits, value_arr = self.predict_fn(obs_np)
                        logits = logits[0]
                        value = float(value_arr[0, 0] if value_arr.ndim > 1 else value_arr[0])
                    else:
                        # Fallback / Error
                        logits = np.zeros(self.game.num_distinct_actions())
                        value = 0.0
                    
                    # Mask
                    legal_actions = state.legal_actions()
                    mask = np.zeros_like(logits, dtype=bool)
                    mask[legal_actions] = True
                    logits[~mask] = -1e9
                    
                    # Policy (Greedy-ish or Sample)
                    # Safe softmax
                    if tf is not None and isinstance(logits, tf.Tensor):
                         logits = logits.numpy()
                    
                    # Clip for numerical stability
                    logits = np.clip(logits, -1e9, 1e9)
                    
                    e_x = np.exp(logits - np.max(logits))
                    probs = e_x / e_x.sum(axis=0) # Simple softmax

                    # Renormalize on mask to be sure
                    probs = probs * mask
                    if probs.sum() == 0:
                        probs[mask] = 1.0
                    probs = probs / probs.sum()
                    
                    action = rng.choice(len(probs), p=probs)
                    
                    # Store info
                    trajectory.append({
                        "state": state.clone(),
                        "model_value": value,
                        "player": curr_p
                    })
                    
                    state.apply_action(action)
            
            # Game Over
            outcome = state.returns() # [r0, r1]
            
            # Analyze Trajectory
            for step_idx, step_data in enumerate(trajectory):
                s = step_data["state"]
                v_model = step_data["model_value"]
                p = step_data["player"]
                
                # Actual outcome for player p
                final_r = outcome[p] # 1.0, -1.0, or 0.0 (roughly)
                
                # Condition 1: Large Prediction Error
                sq_err = (v_model - final_r) ** 2
                
                # Condition 2: Value Change (Sudden Drop)
                val_change = 0.0
                if step_idx < len(trajectory) - 1:
                    next_step = trajectory[step_idx+1]
                    next_v = next_step["model_value"]
                    if next_step["player"] != p:
                        next_v = -next_v # Invert perspective
                    val_change = abs(v_model - next_v)

                is_interesting = False
                reason = []
                
                if sq_err > self.config.prediction_error_threshold:
                    is_interesting = True
                    reason.append("prediction_error")
                
                if val_change > self.config.value_change_threshold:
                    is_interesting = True
                    reason.append("value_swing")

                if is_interesting:
                    interesting_data.append({
                        "state": s,
                        "model_value": v_model,
                        "reason": reason,
                        "actual_outcome": final_r,
                        "turn": s.rust_game.get_state().turn_count
                    })
        
        return interesting_data

    def run_oracle(self, interesting_states):
        """
        Runs OracleSolver on the interesting states.
        """
        results = []
        logging.info(f"Running Oracle on {len(interesting_states)} states...")
        
        for item in interesting_states:
            state = item["state"]
            # Solve
            solver = OracleSolver(self.game, self.config.max_depth, state.current_player())
            oracle_val = solver.solve(state)
            
            if oracle_val is not None:
                # We found a definitive result!
                results.append({
                    "state_key": extract_state_info(state.rust_game.get_state()), # Human readable
                    "oracle_value": float(oracle_val),
                    "model_value": float(item["model_value"]),
                    "actual_outcome": float(item["actual_outcome"]),
                    "reasons": item["reason"]
                    # We might want to save features for training later, 
                    # but for now let's save metadata.
                })
        
        return results

    def save_results(self, results):
        if not results:
            return
            
        with open(self.config.output_file, 'a') as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        logging.info(f"Saved {len(results)} mined samples to {self.config.output_file}")

    def main_loop(self):
        logging.info(f"Miner started. Watching {self.config.checkpoint_dir}...")
        
        # Initialize
        # If config.checkpoint_dir is directly a SavedModel, we load it once.
        if os.path.exists(os.path.join(self.config.checkpoint_dir, "saved_model.pb")):
             self.load_weights(self.config.checkpoint_dir)
             
             # If it's a fixed path (not watching directory), we run once and exit?
             # For now let's Run once if it's not a standard checkpoint dir.
             logging.info("Loaded SavedModel (Single Run Mode).")
             candidates = self.run_diagnostic_games(self.config.diagnostic_games_per_checkpoint)
             logging.info(f"Found {len(candidates)} candidate states.")
             
             if candidates:
                 mined = self.run_oracle(candidates)
                 self.save_results(mined)
             
             return # Exit after single run

        # Standard Watch Mode
        latest = find_latest_checkpoint(self.config.checkpoint_dir)
        if latest:
            self.load_weights(latest)
            # self.known_checkpoints.add(latest) # Process latest


        while self.running:
            # 1. Check for new checkpoint
            latest = find_latest_checkpoint(self.config.checkpoint_dir)
            if latest and latest not in self.known_checkpoints:
                logging.info(f"New checkpoint found: {latest}")
                # Wait a bit to ensure write complete?
                time.sleep(1)
                self.load_weights(latest)
                self.known_checkpoints.add(latest)
                
                # 2. Run Diagnostic
                logging.info("Starting diagnostic games...")
                candidates = self.run_diagnostic_games(self.config.diagnostic_games_per_checkpoint)
                logging.info(f"Found {len(candidates)} candidate states.")
                
                # 3. Run Oracle
                if candidates:
                    mined = self.run_oracle(candidates)
                    
                    # 4. Save
                    self.save_results(mined)
            
            else:
                # No new checkpoint. Sleep.
                time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory or path to model")
    # Supports "checkpoint" alias if user uses it for single file? 
    # Use --checkpoint argument to map to checkpoint_dir for compatibility
    parser.add_argument("--checkpoint", type=str, default=None, help="Alias for checkpoint_dir")
    
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--league_decks_student", type=str, default=None)
    parser.add_argument("--league_decks_teacher", type=str, default=None)
    parser.add_argument("--diagnostic_games_per_checkpoint", type=int, default=10)
    
    args = parser.parse_args()
    
    ckpt = args.checkpoint if args.checkpoint else args.checkpoint_dir

    config = MinerConfig(
        checkpoint_dir=ckpt,
        device=args.device,
        max_depth=args.max_depth,
        league_decks_student=args.league_decks_student,
        league_decks_teacher=args.league_decks_teacher,
        diagnostic_games_per_checkpoint=args.diagnostic_games_per_checkpoint
    )
    
    miner = Miner(config)
    miner.main_loop()
