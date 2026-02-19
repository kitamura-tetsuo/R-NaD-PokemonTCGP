import argparse
import sys
import os
import json
import sqlite3
import re
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

place_pattern = re.compile(r"^Place\((.*), (\d+)\)$")
attach_pattern = re.compile(r"\((\d+), [a-zA-Z]+, (\d+)\)")

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
    find_depth: int = 5 # Oracle search depth
    mine_depth: int = 5 # Tree mining depth
    disable_retreat_depth: int = 3 # Stop exploring retreats beyond this depth
    disable_energy_attach_threshold: int = 100 # Stop exploring energy attachments if pokemon has >= this many energy
    prediction_error_threshold: float = 0.5 # Squared error threshold
    value_change_threshold: float = 0.4 # Absolute change threshold
    device: str = "gpu"
    batch_size: int = 1 # Inference batch size for self-play
    max_visualizations: int = 100 # Limit number of tree visualizations per batch
    min_turn: int = 0 # Minimum turn count for candidate states
    seed: int = 42

class TreeStorage:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup_db()
    
    def setup_db(self):
        self.cursor.execute("DROP TABLE IF EXISTS nodes")
        self.cursor.execute("DROP TABLE IF EXISTS edges")
        
        self.cursor.execute("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY,
                step INTEGER,
                turn INTEGER,
                acting_player INTEGER,
                is_terminal BOOLEAN,
                is_chance BOOLEAN,
                is_ai BOOLEAN,
                is_repeated BOOLEAN,
                repeated_node_id INTEGER,
                action_name TEXT,
                state_json TEXT,
                state_hash INTEGER,
                is_winning BOOLEAN
            )
        """)
        
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_state_hash ON nodes(state_hash)")
        
        self.cursor.execute("""
            CREATE TABLE edges (
                parent_id INTEGER,
                child_id INTEGER,
                action_name TEXT,
                UNIQUE(parent_id, child_id, action_name)
            )
        """)
        self.conn.commit()
    
    def add_node(self, node_data, state_hash=None):
        state_json = json.dumps(node_data.get("state")) if node_data.get("state") else None
        
        self.cursor.execute("""
            INSERT INTO nodes (
                id, step, turn, acting_player, 
                is_terminal, is_chance, is_ai, 
                is_repeated, repeated_node_id, action_name, state_json, state_hash, is_winning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node_data["id"],
            node_data.get("step"),
            node_data.get("state", {}).get("turn", 0) if node_data.get("state") else 0,
            node_data.get("acting_player"),
            node_data.get("is_terminal", False),
            node_data.get("is_chance", False),
            node_data.get("is_ai", False),
            node_data.get("is_repeated", False),
            node_data.get("repeated_node_id"),
            node_data.get("action_name"),
            state_json,
            state_hash,
            node_data.get("is_winning", False)
        ))
        return node_data["id"]

    def check_visited(self, state_hash):
        self.cursor.execute("SELECT id FROM nodes WHERE state_hash = ? AND is_repeated = 0 LIMIT 1", (state_hash,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def add_edge(self, parent_id, child_id, action_name):
        self.cursor.execute("INSERT OR IGNORE INTO edges (parent_id, child_id, action_name) VALUES (?, ?, ?)", (parent_id, child_id, action_name))

    def set_winning(self, node_id):
        self.cursor.execute("UPDATE nodes SET is_winning = 1 WHERE id = ?", (node_id,))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

def save_tree_to_sqlite(initial_state, db_path, mine_depth, disable_retreat_depth, disable_energy_attach_threshold, ai_player):
    db = TreeStorage(db_path)
    node_id_counter = 0
    
    # Stack items: (state, depth, step, action_name, parent_id, ancestors)
    # ancestors: List of (node_id, acting_player, is_chance)
    stack = [(initial_state, 0, 0, "Root", None, [])]
    
    while stack:
        state, depth, step, action_name, parent_id, ancestors = stack.pop()
        
        node_id_counter += 1
        current_node_id = node_id_counter
        
        # 1. Generate Fast Key
        state_raw = state.rust_game.get_state()
        pending_chance = state._pending_stochastic_action if hasattr(state, "_pending_stochastic_action") else None
        state_key = get_fast_state_key(state_raw, pending_chance)
        state_hash = hash(state_key)

        # 2. Check Repeated
        existing_id = db.check_visited(state_hash)
        
        is_repeated = False
        repeated_id = None
        
        if existing_id is not None:
            is_repeated = True
            repeated_id = existing_id
        
        # 3. Extract Info
        state_info = None
        if not is_repeated:
            state_info = extract_state_info(state_raw)
            if pending_chance is not None:
                state_info["_pending_chance"] = pending_chance

        node_data = {
            "id": current_node_id,
            "action_name": action_name + (" (Dup)" if is_repeated else ""),
            "step": step,
            "acting_player": state.current_player(),
            "is_terminal": state.is_terminal(),
            "is_repeated": is_repeated,
            "repeated_node_id": repeated_id,
            "state": state_info,
            "is_winning": False
        }
        
        if state.is_chance_node():
           node_data["is_chance"] = True

        if not state.is_chance_node() and not state.is_terminal() and state.current_player() == ai_player:
           node_data["is_ai"] = True

        db.add_node(node_data, state_hash=state_hash)
        
        if parent_id is not None:
            db.add_edge(parent_id, current_node_id, action_name)

        # Pruning Checks
        should_stop_branch = False

        # Win Pruning
        if state.is_terminal():
            returns = state.returns()
            # If AI wins (assuming ai_player is maximizing and win > 0)
            if returns[ai_player] > 0:
                # Mark current terminal node as winning
                db.set_winning(current_node_id)
                
                # Traverse up ancestors to prune siblings
                to_prune_parents = set()
                
                # Iterate backwards through ancestors
                curr_ancestors = list(ancestors)
                while curr_ancestors:
                    p_id, p_actor, p_chance = curr_ancestors.pop()
                    # Prune siblings if the parent was AI's choice (and not chance)
                    if p_actor == ai_player and not p_chance:
                        to_prune_parents.add(p_id)
                        db.set_winning(p_id) # Mark ancestor as winning
                    else:
                        # Stop if we hit an opponent node or chance node
                        break
                
                if to_prune_parents:
                    # Filter stack: Remove items whose parent_id is in to_prune_parents
                    # stack item index 4 is parent_id
                    stack = [item for item in stack if item[4] not in to_prune_parents]
            
            should_stop_branch = True

        if is_repeated or depth >= mine_depth:
            should_stop_branch = True

        if should_stop_branch:
            if node_id_counter % 1000 == 0: db.commit()
            continue

        # 4. Generate Children
        
        # Prepare ancestors for children
        # Add current node to ancestors
        # (id, acting_player, is_chance)
        current_ancestor_entry = (current_node_id, state.current_player(), state.is_chance_node())
        new_ancestors = ancestors + [current_ancestor_entry]

        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                stack.append((child, depth + 1, step + 1, f"Chance (p={prob:.2f})", current_node_id, new_ancestors))
        else:
            curr_p = state.current_player()
            actions_to_process = []
            bench_place_groups = {} 
            
            for action in state.legal_actions():
                action_str = state.action_to_string(curr_p, action)

                # Retreat Pruning
                if depth >= disable_retreat_depth and "Retreat" in action_str:
                    continue

                # Energy Pruning
                if "Attach" in action_str:
                    matches = attach_pattern.findall(action_str)
                    should_skip = False
                    rust_state = state.rust_game.get_state()
                    p_idx = state.current_player()
                    
                    for _, idx_str in matches:
                        idx = int(idx_str)
                        target_mon = None
                        if idx == 0:
                            target_mon = rust_state.get_active_pokemon(p_idx)
                        else:
                            bench = rust_state.get_bench_pokemon(p_idx)
                            # bench is a list of Options (None or Pokemon) usually, or list of Pokemon?
                            # tree_viz loop: for mon in rust_state.get_bench_pokemon(p):
                            # It seems it returns a list of Option<Pokemon>.
                            # Indices 1..N map to bench[0]..bench[N-1].
                            bench_idx = idx - 1
                            if bench_idx < len(bench):
                                target_mon = bench[bench_idx]
                        
                        if target_mon:
                            # attached_energy is a list
                            if len(target_mon.attached_energy) >= disable_energy_attach_threshold:
                                should_skip = True
                                break
                    
                    if should_skip:
                        continue

                match = place_pattern.match(action_str)
                if match:
                    card_name = match.group(1)
                    index = int(match.group(2))
                    if index >= 1: 
                        if card_name not in bench_place_groups:
                            bench_place_groups[card_name] = []
                        bench_place_groups[card_name].append((index, action, action_str))
                        continue
                
                actions_to_process.append((action, action_str))
            
            for card_name, group in bench_place_groups.items():
                group.sort(key=lambda x: x[0])
                best = group[0]
                actions_to_process.append((best[1], best[2]))
            
            for action, action_str in actions_to_process:
                child = state.clone()
                child.apply_action(action)
                stack.append((child, depth + 1, step + 1, action_str, current_node_id, new_ancestors))

        if node_id_counter % 1000 == 0:
            db.commit()

    db.commit()
    db.close()

class OracleSolver:
    """
    Solves a game state using Expectiminimax / DFS up to max_depth.
    Returns the exact value (-1.0 to 1.0) if determined, or None if inconclusive (depth limit reached without result).
    """
    def __init__(self, game, find_depth, ai_player):
        self.game = game
        self.find_depth = find_depth
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
        if depth >= self.find_depth:
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
            
        self.current_checkpoint_path = "unknown"
        
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
        self.current_checkpoint_path = path
        
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
                    prob_arr = np.array(prob_list)
                    prob_arr /= prob_arr.sum()
                    action = rng.choice(action_list, p=prob_arr)
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
                
                # Turn Filter
                turn_count = s.rust_game.get_state().turn_count
                if turn_count < self.config.min_turn:
                    is_interesting = False

                if is_interesting:
                    interesting_data.append({
                        "state": s,
                        "model_value": v_model,
                        "reason": reason,
                        "actual_outcome": final_r,
                        "turn": s.rust_game.get_state().turn_count,
                        "deck_id_1": d1,
                        "deck_id_2": d2,
                        "seed": self.config.seed + i
                    })
        
        return interesting_data

    def run_oracle(self, interesting_states):
        """
        Runs OracleSolver on the interesting states.
        """
        results = []
        logging.info(f"Running Oracle on {len(interesting_states)} states...")
        visualizations_count = 0
        
        for item in interesting_states:
            state = item["state"]
            # Solve
            solver = OracleSolver(self.game, self.config.find_depth, state.current_player())
            oracle_val = solver.solve(state)
            
            if oracle_val is not None:
                # We found a definitive result!
                res = {
                    "state_key": extract_state_info(state.rust_game.get_state()), # Human readable
                    "oracle_value": float(oracle_val),
                    "model_value": float(item["model_value"]),
                    "actual_outcome": float(item["actual_outcome"]),
                    "reasons": item["reason"],
                    "state_blob": state.observation_tensor(state.current_player()),
                    "history": state.history(),
                    "deck_id_1": item["deck_id_1"],
                    "deck_id_2": item["deck_id_2"],
                    "seed": item["seed"]
                }
                results.append(res)
                self.save_results([res])
                
                # Create detailed tree visualization for this solved state
                # Create detailed tree visualization for this solved state
                if visualizations_count >= self.config.max_visualizations:
                    continue

                try:
                    checkpoint_name = os.path.basename(self.current_checkpoint_path.rstrip(os.sep))
                    if not checkpoint_name: checkpoint_name = "unknown"
                    
                    mined_dir = os.path.join("data", "mined", checkpoint_name)
                    os.makedirs(mined_dir, exist_ok=True)
                    
                    # Use a unique index or seed for filename
                    # We don't have a global index here easily, using seed + result count
                    # Or use 'index' assuming results are appended?
                    # The user said "[index].sqlite". Maybe I should maintain a counter or use seed.
                    # Using global timestamp + unique id might be safer, but user asked for [index].
                    # I'll use list index 'len(results)' (which represents count in this batch) + timestamp to be safe?
                    # Or better: "data/mined/[checkpoint_name]/" is a dir.
                    # I'll use the seed as index since it's unique per run usually? No.
                    # I'll just use a sanitized timestamp + random string or similar?
                    # "data/mined/[chechpoint_name] へ [index].sqlite"
                    # I will use the position in results list as index effectively, 
                    # but since we append to results in the loop, len(results) is 1, 2, ...
                    
                    # Since run_oracle is called once per batch, and we might run multiple times?
                    # I'll use a simple counter based on existing files or random?
                    # Let's simple check number of files?
                    # No, that's race-y.
                    # Let's just use the seed from the item? User requested "Index".
                    # I will interpret index as the index in the current batch processing or loop.
                    
                    sqlite_filename = f"{len(results)}.sqlite"
                    sqlite_path = os.path.join(mined_dir, sqlite_filename)
                    if os.path.exists(sqlite_path):
                         # Avoid overwrite if possible
                         sqlite_filename = f"{len(results)}_{int(time.time())}.sqlite"
                         sqlite_path = os.path.join(mined_dir, sqlite_filename)
                    
                    logging.info(f"Saving tree visualization to {sqlite_path}")
                    save_tree_to_sqlite(state, sqlite_path, self.config.mine_depth, self.config.disable_retreat_depth, self.config.disable_energy_attach_threshold, state.current_player())
                    visualizations_count += 1
                    
                except Exception as e:
                    logging.error(f"Failed to save tree visualization: {e}")
        
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
                 # self.save_results(mined) # Handled incrementally in run_oracle
             
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
                    # self.save_results(mined) # Handled incrementally in run_oracle
            
            else:
                # No new checkpoint. Sleep.
                exit()
                time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory or path to model")
    # Supports "checkpoint" alias if user uses it for single file? 
    # Use --checkpoint argument to map to checkpoint_dir for compatibility
    parser.add_argument("--checkpoint", type=str, default=None, help="Alias for checkpoint_dir")
    
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--find_depth", type=int, default=5)
    parser.add_argument("--mine_depth", type=int, default=5)
    parser.add_argument("--disable_retreat_depth", type=int, default=3)
    parser.add_argument("--disable_energy_attach_threshold", type=int, default=100)
    parser.add_argument("--league_decks_student", type=str, default=None)
    parser.add_argument("--league_decks_teacher", type=str, default=None)
    parser.add_argument("--diagnostic_games_per_checkpoint", type=int, default=10)
    parser.add_argument("--max_visualizations", type=int, default=100, help="Max number of tree visualizations to save per batch")
    parser.add_argument("--min_turn", type=int, default=0, help="Minimum turn count for candidate states")
    
    args = parser.parse_args()
    
    ckpt = args.checkpoint if args.checkpoint else args.checkpoint_dir

    config = MinerConfig(
        checkpoint_dir=ckpt,
        device=args.device,
        find_depth=args.find_depth,
        mine_depth=args.mine_depth,
        disable_retreat_depth=args.disable_retreat_depth,
        disable_energy_attach_threshold=args.disable_energy_attach_threshold,
        league_decks_student=args.league_decks_student,
        league_decks_teacher=args.league_decks_teacher,
        diagnostic_games_per_checkpoint=args.diagnostic_games_per_checkpoint,
        max_visualizations=args.max_visualizations,
        min_turn=args.min_turn
    )
    
    miner = Miner(config)
    miner.main_loop()
