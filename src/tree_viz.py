import argparse
import sys
import os

# Configure JAX platform before importing jax (if --device cpu is used)
if "--device" in sys.argv:
    try:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            # Prevent JAX from discovering CUDA plugins to avoid segfaults on CPU-only machines
            # with jax-cuda installed
            sys.modules["jax_plugins.xla_cuda12"] = None
            sys.modules["jax_plugins.xla_cuda13"] = None
    except ValueError:
        pass

import json
import sqlite3
import re
import random
import datetime
import logging
import jax
import haiku as hk
import numpy as np
import pyspiel
import deckgym
import deckgym_openspiel
import pickle
import jax.numpy as jnp
from src.rnad import RNaDConfig, find_latest_checkpoint
from src.models import DeckGymNet, TransformerNet, CardTransformerNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Compatibility fix for loading checkpoints from newer optax versions
try:
    import optax
except ImportError:
    optax = None

if optax is not None:
    if 'optax.transforms' not in sys.modules:
        sys.modules['optax.transforms'] = optax
    if 'optax.transforms._accumulation' not in sys.modules:
        sys.modules['optax.transforms._accumulation'] = optax

place_pattern = re.compile(r"^Place\((.*), (\d+)\)$")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the game tree with one AI side.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file.")
    parser.add_argument("--deck_id_1", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck 1 file.")
    parser.add_argument("--deck_id_2", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck 2 file.")
    parser.add_argument("--output", type=str, default="tree.sqlite", help="Path to output HTML file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for chance nodes (if not exploring all) and initial split.")
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='gpu', help="Device for inference.")
    parser.add_argument("--disable_jit", action="store_true", help="Disable JIT compilation.")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth of the tree.")
    parser.add_argument("--ai_player", type=int, default=0, help="Which player is controlled by AI (0 or 1).")
    parser.add_argument("--explore_all_chance", action="store_true", help="Whether to explore all chance outcomes (can explode).")
    parser.add_argument("--mined_source", type=str, default=None, help="Path to mined_data.jsonl to load state from.")
    parser.add_argument("--mined_index", type=int, default=-1, help="Index of the mined state to load (default: last one).")
    parser.add_argument("--explore_ai_moves", action="store_true", help="If true, expand all legal moves for AI instead of just argmax.")
    return parser.parse_args()

def get_card_image_url(card_id):
    parts = card_id.split()
    if len(parts) != 2: return ""
    set_id, card_num = parts[0], parts[1]
    return f"https://limitlesstcg.nyc3.cdn.digitaloceanspaces.com/pocket/{set_id}/{set_id}_{card_num}_EN_SM.webp"

def extract_state_info(rust_state):
    info = {
        "turn": rust_state.turn_count,
        "current_player": rust_state.current_player,
        "points": rust_state.points,
        "winner": None,
        "players": []
    }
    if rust_state.is_game_over():
        outcome = rust_state.winner
        info["winner"] = {"winner": outcome.winner, "is_tie": outcome.is_tie}

    for p in [0, 1]:
        p_info = {
            "hand": [], "active": None, "bench": [],
            "discard_pile_size": rust_state.get_discard_pile_size(p),
            "deck_size": rust_state.get_deck_size(p)
        }
        for card in rust_state.get_hand(p):
            p_info["hand"].append({"id": card.id, "name": card.name, "url": get_card_image_url(card.id)})
        
        active = rust_state.get_active_pokemon(p)
        if active:
            tool_info = None
            if hasattr(active, "attached_tool") and active.attached_tool:
                tool = active.attached_tool
                tool_info = {"id": tool.id, "name": tool.name, "url": get_card_image_url(tool.id)}
            p_info["active"] = {
                "id": active.card.id, "name": active.name, "url": get_card_image_url(active.card.id),
                "hp": active.remaining_hp, "max_hp": active.total_hp,
                "energy": [e.name for e in active.attached_energy], "tool": tool_info, "status": []
            }
            if active.poisoned: p_info["active"]["status"].append("Poisoned")
            if active.asleep: p_info["active"]["status"].append("Asleep")
            if active.paralyzed: p_info["active"]["status"].append("Paralyzed")

        for mon in rust_state.get_bench_pokemon(p):
            if not mon: continue
            tool_info = None
            if hasattr(mon, "attached_tool") and mon.attached_tool:
                tool = mon.attached_tool
                tool_info = {"id": tool.id, "name": tool.name, "url": get_card_image_url(tool.id)}
            p_info["bench"].append({
                "id": mon.card.id, "name": mon.name, "url": get_card_image_url(mon.card.id),
                "hp": mon.remaining_hp, "max_hp": mon.total_hp,
                "energy": [e.name for e in mon.attached_energy], "tool": tool_info, "status": []
            })
        info["players"].append(p_info)
    return info

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
                state_hash INTEGER
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
                is_repeated, repeated_node_id, action_name, state_json, state_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            state_hash
        ))
        return node_data["id"]

    def check_visited(self, state_hash):
        self.cursor.execute("SELECT id FROM nodes WHERE state_hash = ? AND is_repeated = 0 LIMIT 1", (state_hash,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def add_edge(self, parent_id, child_id, action_name):
        self.cursor.execute("INSERT OR IGNORE INTO edges (parent_id, child_id, action_name) VALUES (?, ?, ?)", (parent_id, child_id, action_name))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


# Optimized key generation using tuples instead of JSON
def get_fast_state_key(state_raw, pending_chance):
    if pending_chance is not None and isinstance(pending_chance, tuple) and len(pending_chance) > 1:
        # Ensure probabilities list is a tuple for hashing
        if isinstance(pending_chance[1], list):
            pending_chance = (pending_chance[0], tuple(pending_chance[1]))

    players_info = []
    for p in [0, 1]:
        # Hand (sorted ids for canonicalization)
        hand_ids = tuple(sorted([c.id for c in state_raw.get_hand(p)]))
        
        # Active
        active = state_raw.get_active_pokemon(p)
        active_info = None
        if active:
            # active properties
            energy_names = tuple(sorted([e.name for e in active.attached_energy]))
            status = []
            if active.poisoned: status.append(1)
            if active.asleep: status.append(2)
            if active.paralyzed: status.append(3)
            tool_id = active.attached_tool.id if hasattr(active, "attached_tool") and active.attached_tool else ""
            active_info = (active.card.id, active.remaining_hp, energy_names, tuple(status), tool_id)
        
        # Bench (sorted for canonicalization)
        bench_list = []
        for mon in state_raw.get_bench_pokemon(p):
            if mon:
                energy_names = tuple(sorted([e.name for e in mon.attached_energy]))
                tool_id = mon.attached_tool.id if hasattr(mon, "attached_tool") and mon.attached_tool else ""
                bench_list.append((mon.card.id, mon.remaining_hp, energy_names, tool_id))
        bench_info = tuple(sorted(bench_list))
        
        # Trash (sorted ids for canonicalization)
        # Use get_discard_pile if available, otherwise just size (fallback)
        if hasattr(state_raw, "get_discard_pile"):
            try:
                trash_ids = tuple(sorted([c.id for c in state_raw.get_discard_pile(p)]))
            except Exception:
                    # Fallback if method exists but fails (e.g. wrong binding sig)
                trash_ids = (state_raw.get_discard_pile_size(p),)
        else:
            trash_ids = (state_raw.get_discard_pile_size(p),)

        players_info.append((
            hand_ids,
            active_info,
            bench_info,
            state_raw.get_deck_size(p),
            trash_ids,
            tuple(state_raw.points) if hasattr(state_raw.points, '__iter__') else state_raw.points
        ))

    return (
        state_raw.turn_count,
        state_raw.current_player,
        tuple(players_info),
        pending_chance
    )

def main():
    args = parse_args()
    if args.device == 'cpu':
        jax.config.update("jax_platform_name", "cpu")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.disable_jit: jax.config.update("jax_disable_jit", True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    emb_path = "card_embeddings.npy"
    embedding_matrix = jnp.array(np.load(emb_path)) if os.path.exists(emb_path) else jnp.zeros((10000, 26))

    # Load Mined Data if specified
    mined_history = None
    if args.mined_source:
        if not os.path.exists(args.mined_source):
            logging.error(f"Mined source file not found: {args.mined_source}")
            return

        with open(args.mined_source, 'r') as f:
            lines = f.readlines()
            if not lines:
                logging.error("Mined source file is empty.")
                return
            
            try:
                if args.mined_index >= len(lines):
                     logging.warning(f"Index {args.mined_index} out of range (len={len(lines)}), using last entry.")
                     idx = -1
                else:
                     idx = args.mined_index

                data = json.loads(lines[idx])
                args.deck_id_1 = data.get("deck_id_1", args.deck_id_1)
                args.deck_id_2 = data.get("deck_id_2", args.deck_id_2)
                args.seed = data.get("seed", args.seed)
                mined_history = data.get("history", [])
                
                logging.info(f"Loaded mined state (idx={idx}): seed={args.seed}, history_len={len(mined_history)}")
            except json.JSONDecodeError:
                logging.error("Failed to decode JSON from mined source.")
                return

    config = RNaDConfig(deck_id_1=args.deck_id_1, deck_id_2=args.deck_id_2)
    game = pyspiel.load_game("deckgym_ptcgp", {
        "deck_id_1": config.deck_id_1, "deck_id_2": config.deck_id_2,
        "seed": args.seed, "max_game_length": config.unroll_length
    })
    num_actions, obs_shape = game.num_distinct_actions(), game.observation_tensor_shape()

    checkpoint_path = args.checkpoint
    if checkpoint_path and os.path.isdir(checkpoint_path):
        latest = find_latest_checkpoint(checkpoint_path)
        if latest: checkpoint_path = latest

    # Model Setup
    def forward_factory(c):
        def forward(x):
            net = CardTransformerNet(
                num_actions=num_actions,
                embedding_matrix=embedding_matrix,
                hidden_size=c.transformer_embed_dim,
                num_blocks=c.transformer_layers,
                num_heads=c.transformer_heads
            )
            return net(x)
        return forward

    checkpoint_data = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            if 'config' in checkpoint_data:
                config = checkpoint_data['config']
                logging.info(f"Loaded config from checkpoint: dim={config.transformer_embed_dim}, layers={config.transformer_layers}")

    network = hk.transform(forward_factory(config))
    params = network.init(rng, jnp.zeros((1, *obs_shape)))
    
    if checkpoint_data:
        params = checkpoint_data['params']
        logging.info("Checkpoint parameters loaded.")
        del checkpoint_data # Free memory
    
    jit_apply = jax.jit(network.apply) if not args.disable_jit else network.apply

    def predict(state):
        obs = np.array(state.observation_tensor(state.current_player()))[None, ...]
        logits, _ = jit_apply(params, rng, obs)
        logits = np.array(logits[0])
        mask = np.zeros_like(logits, dtype=bool)
        mask[state.legal_actions()] = True
        logits[~mask] = -1e9
        return logits

    # Tree Search
    node_id_counter = 0
    # visited_states = {} # Removed for memory optimization

    db = TreeStorage(args.output)
    logging.info(f"Initialized SQLite database at {args.output}")

    def explore_iterative(initial_state, max_depth):
        nonlocal node_id_counter
        unique_states_counter = 0
        # Stack items: (state, depth, step, action_name, parent_id)
        # parent_id is None for Root
        stack = [(initial_state, 0, 0, "Root", None)]
        
        while stack:
            state, depth, step, action_name, parent_id = stack.pop()
            
            node_id_counter += 1
            current_node_id = node_id_counter
            
            # 1. Generate Fast Key
            state_raw = state.rust_game.get_state()
            pending_chance = None
            if hasattr(state, "_pending_stochastic_action") and state._pending_stochastic_action is not None:
                pending_chance = state._pending_stochastic_action
                
            state_key = get_fast_state_key(state_raw, pending_chance)
            state_hash = hash(state_key)

            # 2. Check Repeated (Query DB)
            existing_id = db.check_visited(state_hash)
            
            is_repeated = False
            repeated_id = None
            
            if existing_id is not None:
                is_repeated = True
                repeated_id = existing_id
                # Even if repeated, we add a node to represent this instance in the tree (as a leaf/dup)
                # But we DON'T add to stack (prune)
            else:
                unique_states_counter += 1
            
            # 3. Extract Info (only if not repeated, or we want to show it? Current logic showed it as Dup node)
            # To save memory/time, maybe don't extract full state for Dup?
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
                "state": state_info
            }
            
            # Chance node check
            if state.is_chance_node():
               node_data["is_chance"] = True

            # AI check
            if not state.is_chance_node() and not state.is_terminal() and state.current_player() == args.ai_player:
               node_data["is_ai"] = True

            # Insert Node
            db.add_node(node_data, state_hash=state_hash)
            
            # Insert Edge (if parent exists)
            if parent_id is not None:
                db.add_edge(parent_id, current_node_id, action_name)

            # Pruning conditions
            if is_repeated or state.is_terminal() or depth >= max_depth:
                if node_id_counter % 1000 == 0: db.commit()
                continue

            # 4. Generate Children
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                # Determine children to push
                # Note: Stack is LIFO, so if we want consistent order, push in reverse?
                # Doesn't matter much for functionality
                
                if args.explore_all_chance:
                    for action, prob in outcomes:
                        child_state = state.clone()
                        child_state.apply_action(action)
                        stack.append((child_state, depth + 1, step + 1, f"Chance (p={prob:.2f})", current_node_id))
                else:
                    action_list, prob_list = zip(*outcomes)
                    action = np.random.choice(action_list, p=prob_list)
                    child_state = state.clone()
                    child_state.apply_action(action)
                    stack.append((child_state, depth + 1, step + 1, "Chance (Sampled)", current_node_id))

            else:
                curr_p = state.current_player()
                if curr_p == args.ai_player and not args.explore_ai_moves:
                    # AI Trace (Greedy / Argmax)
                    logits = predict(state)
                    action = int(np.argmax(logits))
                    child_state = state.clone()
                    child_state.apply_action(action)
                    action_str = state.action_to_string(curr_p, action)
                    stack.append((child_state, depth + 1, step + 1, action_str, current_node_id))
                else:
                    # Opponent OR AI Exploration (Explore all legal actions)
                    legal_actions = state.legal_actions()
                    actions_to_process = []
                    bench_place_groups = {} 

                    for action in legal_actions:
                        action_str = state.action_to_string(curr_p, action)
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
                        child_state = state.clone()
                        child_state.apply_action(action)
                        stack.append((child_state, depth + 1, step + 1, action_str, current_node_id))


            if node_id_counter % 1000 == 0:
                print(f"Nodes processed: {node_id_counter} (Unique States: {unique_states_counter}), Stack size: {len(stack)}")
                db.commit()

    logging.info("Starting tree exploration...")
    initial_state = game.new_initial_state()
    if mined_history:
        logging.info("Replaying history...")
        for action in mined_history:
            initial_state.apply_action(action)
        logging.info(f"Replay complete, current turn: {initial_state.rust_game.get_state().turn_count}")

    explore_iterative(initial_state, args.max_depth)
    
    db.commit()
    db.close()
    logging.info(f"Tree exploration finished. Data saved to {args.output}")

if __name__ == "__main__":
    main()
