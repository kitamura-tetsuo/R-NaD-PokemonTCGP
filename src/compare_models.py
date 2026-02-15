import argparse
import sys
import os
import json
import logging
import random
import time
import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
import pickle
import math
import sqlite3
import hashlib
from typing import Optional
import optax
    
# Compatibility fix for loading checkpoints from newer optax versions
if 'optax.transforms' not in sys.modules:
    sys.modules['optax.transforms'] = optax
if 'optax.transforms._accumulation' not in sys.modules:
    sys.modules['optax.transforms._accumulation'] = optax

# Setup Path to include src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pyspiel
from src.rnad import RNaDConfig, find_latest_checkpoint, LeagueConfig, merge_recursive
from src.models import DeckGymNet, CardTransformerNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try importing tensorflow for SavedModel loading
try:
    import tensorflow as tf
except ImportError:
    tf = None

def compute_hash(numpy_array):
    """Compute SHA256 hash of a numpy array."""
    return hashlib.sha256(numpy_array.tobytes()).hexdigest()

class ModelWrapper:
    def __init__(self, path: str, device: str = 'gpu', default_config: RNaDConfig = None, embedding_matrix=None, num_actions=None, obs_shape=None):
        self.path = path
        self.device = device
        self.predict_fn = None
        self.is_saved_model = False
        self.config = default_config
        self.params = None
        self.rng = jax.random.PRNGKey(42)  # Initial seed
        self.embedding_matrix = embedding_matrix
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.model_hash = None
        
        if os.path.exists(os.path.join(path, "saved_model.pb")):
            self.is_saved_model = True
        elif os.path.isdir(path):
            latest = find_latest_checkpoint(path)
            if latest:
                self.path = latest
                self.is_saved_model = False
            else:
                 pass

        self._load()

    def _load(self):
        if self.is_saved_model:
            logging.info(f"Loading SavedModel from {self.path}")
            if tf is None:
                raise ImportError("TensorFlow is required to load SavedModels.")
            
            try:
                if self.device == 'cpu':
                    tf.config.set_visible_devices([], 'GPU')
            except Exception as e:
                logging.warning(f"Failed to set TF visible devices: {e}")

            self.tf_module = tf.saved_model.load(self.path)
            
            # Compute Hash from first variable
            try:
                if hasattr(self.tf_module, 'variables') and len(self.tf_module.variables) > 0:
                    # Use the first variable (usually normalized or kernel)
                    var = self.tf_module.variables[0].numpy()
                    self.model_hash = compute_hash(var)
                else:
                    # Fallback to path hash if no variables found (unlikely)
                    self.model_hash = hashlib.md5(self.path.encode()).hexdigest()
            except Exception as e:
                logging.warning(f"Failed to compute hash for SavedModel: {e}")
                self.model_hash = "unknown_saved_model"

            def tf_predict(obs_np):
                out = self.tf_module.predict(obs_np)
                if isinstance(out, dict):
                    return out['policy'].numpy(), out['value'].numpy()
                else:
                    return out[0].numpy(), out[1].numpy()
            
            self.predict_fn = tf_predict
        
        else:
            logging.info(f"Loading JAX Checkpoint from {self.path}")
            try:
                with open(self.path, 'rb') as f:
                    data = pickle.load(f)
                
                loaded_params = data['params']
                
                # Compute Hash from first leaf of params
                try:
                    leaves = jax.tree_util.tree_leaves(loaded_params)
                    if leaves:
                        self.model_hash = compute_hash(np.array(leaves[0]))
                    else:
                        self.model_hash = "empty_params"
                except Exception as e:
                    logging.warning(f"Failed to compute hash for JAX model: {e}")
                    self.model_hash = "unknown_jax"

                if 'config' in data:
                    ckpt_config = data['config']
                    if self.config:
                        self.config = self.config._replace(
                            model_type=getattr(ckpt_config, 'model_type', 'transformer'),
                            transformer_embed_dim=getattr(ckpt_config, 'transformer_embed_dim', 64),
                            transformer_layers=getattr(ckpt_config, 'transformer_layers', 2),
                            transformer_heads=getattr(ckpt_config, 'transformer_heads', 4),
                            hidden_size=getattr(ckpt_config, 'hidden_size', 256),
                            num_blocks=getattr(ckpt_config, 'num_blocks', 2)
                        )
                    else:
                        self.config = ckpt_config
            except Exception as e:
                logging.error(f"Failed to load checkpoint pickle: {e}")
                raise e

            def forward(x):
                if self.config.model_type == "transformer":
                    net = CardTransformerNet(
                        num_actions=self.num_actions,
                        embedding_matrix=self.embedding_matrix,
                        hidden_size=self.config.transformer_embed_dim,
                        num_blocks=self.config.transformer_layers,
                        num_heads=self.config.transformer_heads,
                    )
                else:
                    net = DeckGymNet(
                        num_actions=self.num_actions,
                        hidden_size=self.config.hidden_size,
                        num_blocks=self.config.num_blocks
                    )
                return net(x)

            network = hk.transform(forward)
            jit_apply = jax.jit(network.apply)
            
            # Init fresh params for current code
            if self.obs_shape:
                 dummy_obs = jnp.zeros((1, *self.obs_shape))
                 fresh_params = network.init(self.rng, dummy_obs)
                 
                 # Merge loaded params into fresh params
                 logging.info("Merging checkpoint params into fresh model structure...")
                 self.params = merge_recursive(fresh_params, loaded_params)
            else:
                 logging.warning("obs_shape not provided, skipping param init & merge. Using loaded params directly (risk of shape mismatch).")
                 self.params = loaded_params
            
            def jax_predict(obs_np):
                logits, values = jit_apply(self.params, self.rng, obs_np)
                return np.array(logits), np.array(values)
                
            self.predict_fn = jax_predict
            
        logging.info(f"Model ID (Hash): {self.model_hash}")

    def get_action(self, state, temperature=1.0):
        curr_p = state.current_player()
        obs = state.observation_tensor(curr_p)
        obs_np = np.array(obs)[None, ...]

        logits, _ = self.predict_fn(obs_np)
        logits = logits[0] 

        legal_actions = state.legal_actions()
        mask = np.zeros_like(logits, dtype=bool)
        mask[legal_actions] = True
        
        logits[~mask] = -1e9
        logits = logits - np.max(logits)
        
        if temperature == 0:
            best_idx = np.argmax(logits)
            return best_idx
        else:
            logits = logits / temperature
            probs = np.exp(logits)
            probs = probs * mask 
            probs_sum = probs.sum()
            if probs_sum == 0:
                 probs[mask] = 1.0
                 probs_sum = probs.sum()
            probs /= probs_sum
            
            action = np.random.choice(len(probs), p=probs)
            return action

def init_db(db_path="matches.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS matches (
                    model1_hash TEXT,
                    model2_hash TEXT,
                    deck1 TEXT,
                    deck2 TEXT,
                    seed INTEGER,
                    p1_is_model1 INTEGER,
                    winner INTEGER,
                    timestamp REAL,
                    PRIMARY KEY (model1_hash, model2_hash, deck1, deck2, seed, p1_is_model1)
                )''')
    conn.commit()
    return conn

def get_cached_result(conn, model1_hash, model2_hash, deck1, deck2, seed, p1_is_model1):
    c = conn.cursor()
    c.execute("SELECT winner FROM matches WHERE model1_hash=? AND model2_hash=? AND deck1=? AND deck2=? AND seed=? AND p1_is_model1=?",
              (model1_hash, model2_hash, deck1, deck2, seed, 1 if p1_is_model1 else 0))
    row = c.fetchone()
    if row:
        return row[0]
    return None

def cache_result(conn, model1_hash, model2_hash, deck1, deck2, seed, p1_is_model1, winner):
    c = conn.cursor()
    try:
        c.execute("INSERT OR REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (model1_hash, model2_hash, deck1, deck2, seed, 1 if p1_is_model1 else 0, winner, time.time()))
        conn.commit()
    except Exception as e:
        logging.warning(f"Failed to cache result: {e}")

def run_game(game, model1, model2, p1_is_model1=True):
    state = game.new_initial_state()
    p0_model = model1 if p1_is_model1 else model2
    p1_model = model2 if p1_is_model1 else model1
    
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            current_player = state.current_player()
            model = p0_model if current_player == 0 else p1_model
            action = model.get_action(state, temperature=1.0)
            state.apply_action(action)
    
    returns = state.returns()
    return returns 

def main():
    parser = argparse.ArgumentParser(description="Compare two R-NaD models.")
    parser.add_argument("--model1", type=str, required=True, help="Path to first checkpoint or SavedModel")
    parser.add_argument("--model2", type=str, required=True, help="Path to second checkpoint or SavedModel")
    parser.add_argument("--n_games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--deck_id", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Deck ID for P1")
    parser.add_argument("--deck2_id", type=str, default=None, help="Optional Deck ID for P2")
    parser.add_argument("--league_decks", type=str, default=None, help="Path to league decks CSV")
    parser.add_argument("--device", type=str, default="gpu", choices=['cpu', 'gpu'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db_path", type=str, default="matches.db", help="Path to SQLite DB for caching")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # DB Init
    conn = init_db(args.db_path)
    
    if args.device == 'cpu':
        jax.config.update("jax_platform_name", "cpu")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # League or Fixed Decks
    league = None
    if args.league_decks:
        logging.info(f"Loading League from {args.league_decks}")
        # Treat as student_csv to load into decks list
        league = LeagueConfig.from_csv(args.league_decks, None)
    
    # Initial dummy game for shapes
    dummy_game = pyspiel.load_game("deckgym_ptcgp", {"deck_id_1": args.deck_id, "deck_id_2": args.deck_id})
    num_actions = dummy_game.num_distinct_actions()
    
    emb_path = "card_embeddings.npy"
    if os.path.exists(emb_path):
        embedding_matrix = jnp.array(np.load(emb_path))
    else:
        embedding_matrix = jnp.zeros((10000, 26))
        
    default_config = RNaDConfig(deck_id_1=args.deck_id, deck_id_2=args.deck_id)

    logging.info("Loading Model 1...")
    m1 = ModelWrapper(args.model1, args.device, default_config, embedding_matrix, num_actions, obs_shape=dummy_game.observation_tensor_shape())
    logging.info("Loading Model 2...")
    m2 = ModelWrapper(args.model2, args.device, default_config, embedding_matrix, num_actions, obs_shape=dummy_game.observation_tensor_shape())
    
    logging.info(f"Starting Matchup: {args.n_games} games per matchup...")
    
    results = {
        "m1_wins": 0,
        "m2_wins": 0,
        "draws": 0,
        "m1_p0_wins": 0,
        "m1_p1_wins": 0,
        "m2_p0_wins": 0,
        "m2_p1_wins": 0
    }
    
    # 1. Generate Matchups
    matchups = []
    if league:
        # Full round-robin (or rather, full product including mirrors)
        for d1 in league.decks:
            for d2 in league.decks:
                matchups.append((d1, d2))
        logging.info(f"League Mode: {len(league.decks)} decks -> {len(matchups)} matchups.")
    else:
        d1 = args.deck_id
        d2 = args.deck2_id if args.deck2_id else args.deck_id
        matchups.append((d1, d2))
        
    total_games = len(matchups) * args.n_games
    logging.info(f"Total Games to Run: {total_games}")
    
    start_time = time.time()
    global_game_idx = 0
    
    # Constant to separate seeds between matchups
    # Assumes n_games < 10000
    MATCHUP_SEED_OFFSET = 10000
    
    for matchup_idx, (d1, d2) in enumerate(matchups):
        # logging.info(f"Matchup: {os.path.basename(d1)} vs {os.path.basename(d2)}")
        
        for i in range(args.n_games):
            global_game_idx += 1
            
            # 2. Determine Setup
            p1_is_model1 = (i % 2 == 0)
            
            # Unique seed for this specific game instance
            # We use matchup_idx to separate seeds, and i for offset within matchup.
            # This ensures that changing n_games doesn't shift seeds for earlier games in subsequent matchups.
            game_seed = args.seed + (matchup_idx * MATCHUP_SEED_OFFSET) + i
            
            # 3. Check Cache
            cached_winner = get_cached_result(
                conn, m1.model_hash, m2.model_hash, d1, d2, game_seed, p1_is_model1
            )
            
            if cached_winner is not None:
                winner = cached_winner
            else:
                # 4. Run Game
                try:
                    game = pyspiel.load_game(
                        "deckgym_ptcgp",
                        {
                            "deck_id_1": d1,
                            "deck_id_2": d2,
                            "seed": game_seed,
                            "max_game_length": 400
                        }
                    )
                    
                    outcome = run_game(game, m1, m2, p1_is_model1)
                    r0, r1 = outcome
                    
                    if r0 > r1: winner = 0
                    elif r1 > r0: winner = 1
                    else: winner = -1
                except Exception as e:
                    logging.error(f"Game failed: {e}. Treating as draw.")
                    winner = -1
                
                # Cache it
                cache_result(conn, m1.model_hash, m2.model_hash, d1, d2, game_seed, p1_is_model1, winner)

            # 5. Record Stats
            if p1_is_model1:
                if winner == 0:
                    results["m1_wins"] += 1
                    results["m1_p0_wins"] += 1
                elif winner == 1:
                    results["m2_wins"] += 1
                    results["m2_p1_wins"] += 1
                else:
                    results["draws"] += 1
            else:
                if winner == 0:
                    results["m2_wins"] += 1
                    results["m2_p0_wins"] += 1
                elif winner == 1:
                    results["m1_wins"] += 1
                    results["m1_p1_wins"] += 1
                else:
                    results["draws"] += 1
                    
            if global_game_idx % 10 == 0 or global_game_idx == total_games:
                elapsed = time.time() - start_time
                print(f"Game {global_game_idx}/{total_games} completed. M1 Wins: {results['m1_wins']}, M2 Wins: {results['m2_wins']}, Draws: {results['draws']} ({elapsed:.1f}s)")

    conn.close()

    m1_win_rate = results["m1_wins"] / total_games
    m2_win_rate = results["m2_wins"] / total_games
    draw_rate = results["draws"] / total_games
    
    p = m1_win_rate
    n = total_games
    z = 1.96
    
    if n > 0:
        conf_interval = z * math.sqrt((p * (1 - p)) / n)
        lower_bound = max(0.0, p - conf_interval)
        upper_bound = min(1.0, p + conf_interval)
    else:
        lower_bound = 0
        upper_bound = 0

    print("\n" + "="*40)
    print("LEAGUE RESULTS")
    print("="*40)
    print(f"Model 1: {args.model1} (Hash: {m1.model_hash[:8]}...)")
    print(f"Model 2: {args.model2} (Hash: {m2.model_hash[:8]}...)")
    print(f"Total Games: {total_games}")
    print("-" * 40)
    print(f"Model 1 Wins: {results['m1_wins']} ({m1_win_rate*100:.1f}%)")
    print(f"Model 2 Wins: {results['m2_wins']} ({m2_win_rate*100:.1f}%)")
    print(f"Draws:       {results['draws']} ({draw_rate*100:.1f}%)")
    print("-" * 40)
    print(f"M1 Win Rate 95% CI: [{lower_bound*100:.1f}%, {upper_bound*100:.1f}%]")
    
    effective_wins = results["m1_wins"] + 0.5 * results["draws"]
    p_hat = effective_wins / n
    std_error = math.sqrt(0.25 / n)
    z_score = (p_hat - 0.5) / std_error
    
    print(f"Z-Score (H0: p=0.5): {z_score:.2f}")
    if abs(z_score) > 1.96:
        print("Result: STATISTICALLY SIGNIFICANT difference (p < 0.05)")
    else:
        print("Result: NO statistically significant difference")
        
    print("\nMatrix:")
    print(f"         | M1 (P0) | M2 (P0)")
    print(f"---------+---------+--------")
    print(f"M1 (P1)  |   ---   | W: {results['m1_p1_wins']} L: {results['m2_p0_wins']}")
    print(f"M2 (P1)  | W: {results['m2_p1_wins']} L: {results['m1_p0_wins']} |   ---")
    
    report = {
        "model1": args.model1,
        "model1_hash": m1.model_hash,
        "model2": args.model2,
        "model2_hash": m2.model_hash,
        "n_games": args.n_games,
        "total_games": total_games,
        "results": results,
        "stats": {
            "m1_win_rate": m1_win_rate,
            "ci_lower": lower_bound,
            "ci_upper": upper_bound,
            "z_score": z_score
        }
    }
    
    with open("league_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("\nReport saved to league_report.json")

if __name__ == "__main__":
    main()
