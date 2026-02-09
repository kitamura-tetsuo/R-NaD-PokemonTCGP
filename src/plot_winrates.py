import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import os
import sys
import argparse
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Dict, Any

# Add src to python path if not already there
sys.path.append(os.getcwd())

from src.models import DeckGymNet

# Try importing deckgym
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
    print("Error: Could not import deckgym. Make sure it is installed.")
    sys.exit(1)

def load_checkpoint(filepath: str) -> Tuple[Any, Any, int]:
    """Loads a checkpoint from the given path."""
    print(f"Loading checkpoint: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data['config'], data['step']

def get_deck_name(deck_path: str) -> str:
    """Extracts a simple name from the deck path."""
    filename = os.path.basename(deck_path)
    name, _ = os.path.splitext(filename)
    return name

def evaluate_pair(p1_params, p2_params, config, deck1_path: str, deck2_path: str, num_games: int, batch_size: int) -> Dict[str, int]:
    """
    Evaluates a pair of decks using the given params for P1 and P2.
    Returns: {'p1_wins': int, 'p2_wins': int, 'ties': int, 'total': int}
    """

    # Determine config values (Assumes both players use same architecture config)
    if isinstance(config, dict):
        hidden_size = config.get('hidden_size', 256)
        num_blocks = config.get('num_blocks', 4)
    else:
        hidden_size = getattr(config, 'hidden_size', 256)
        num_blocks = getattr(config, 'num_blocks', 4)

    # Determine num_actions
    try:
        num_actions = PyGameState.get_action_space_size()
    except Exception:
        num_actions = 38907

    def forward(obs):
        net = DeckGymNet(
            num_actions=num_actions,
            hidden_size=hidden_size,
            num_blocks=num_blocks
        )
        return net(obs)

    network = hk.without_apply_rng(hk.transform(forward))

    results = {'p1_wins': 0, 'p2_wins': 0, 'ties': 0, 'total': 0}

    games_remaining = num_games

    while games_remaining > 0:
        current_batch_size = min(batch_size, games_remaining)

        # Initialize batch of games
        games = []
        for _ in range(current_batch_size):
            # Using random seed (None)
            games.append(PyGameState(deck1_path, deck2_path, None))

        # Initial observations
        current_obs = []
        for g in games:
            current_obs.append(g.encode_observation())
        current_obs = np.array(current_obs, dtype=np.float32)

        active_mask = np.ones(current_batch_size, dtype=bool)

        while active_mask.any():
            # Identify active games and current player
            active_indices = np.where(active_mask)[0]

            p1_indices = []
            p2_indices = []

            for idx in active_indices:
                # Get current player (0 or 1)
                state = games[idx].get_state()
                cp = state.current_player
                if cp == 0:
                    p1_indices.append(idx)
                else:
                    p2_indices.append(idx)

            # Prepare logits array
            batch_logits = np.zeros((current_batch_size, num_actions), dtype=np.float32)

            # Inference for P1
            if p1_indices:
                obs_p1 = current_obs[p1_indices]
                logits_p1, _ = network.apply(p1_params, obs_p1)
                batch_logits[p1_indices] = np.array(logits_p1)

            # Inference for P2
            if p2_indices:
                obs_p2 = current_obs[p2_indices]
                logits_p2, _ = network.apply(p2_params, obs_p2)
                batch_logits[p2_indices] = np.array(logits_p2)

            # Mask illegal actions & Sample
            actions = [0] * current_batch_size # Initialize list

            for i in active_indices:
                legal = games[i].legal_actions()
                if not legal:
                    active_mask[i] = False
                    continue

                logits = batch_logits[i]
                mask = np.ones(logits.shape[-1], dtype=bool)
                mask[legal] = False
                logits[mask] = -1e9

                # Softmax with stability
                l_max = np.max(logits)
                exp_l = np.exp(logits - l_max)
                probs = exp_l / np.sum(exp_l)

                a = np.random.choice(len(probs), p=probs)
                actions[i] = a

            # Step environments
            next_obs_list = [None] * current_batch_size

            # Pre-fill inactive with zeros (to maintain array shape/type)
            for i in range(current_batch_size):
                if not active_mask[i]:
                    next_obs_list[i] = np.zeros_like(current_obs[0])

            for i in active_indices:
                if active_mask[i]: # Check again as it might have been disabled if no legal actions
                    try:
                        done, p0_won = games[i].step_with_id(actions[i])

                        if done:
                            active_mask[i] = False
                            outcome = games[i].get_state().winner
                            if outcome is None:
                                results['ties'] += 1
                            elif outcome.is_tie:
                                results['ties'] += 1
                            elif outcome.winner == 0:
                                results['p1_wins'] += 1
                            elif outcome.winner == 1:
                                results['p2_wins'] += 1

                            next_obs_list[i] = np.zeros_like(current_obs[0])
                        else:
                            next_obs_list[i] = games[i].encode_observation()
                    except Exception as e:
                        print(f"Error stepping game {i}: {e}")
                        active_mask[i] = False
                        next_obs_list[i] = np.zeros_like(current_obs[0])

            current_obs = np.array(next_obs_list, dtype=np.float32)

        games_remaining -= current_batch_size
        results['total'] += current_batch_size

        print(f"Progress: {results['total']}/{num_games} games played. Wins: {results['p1_wins']}-{results['p2_wins']} (Ties: {results['ties']})")

    return results

def main():
    parser = argparse.ArgumentParser(description="Plot win rates for DeckGym checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoints.")
    parser.add_argument("--control_checkpoint", type=str, required=True, help="Path to the control checkpoint for Player 2.")
    parser.add_argument("--decks", type=str, nargs='+', required=True, help="List of deck files/IDs.")
    parser.add_argument("--output", type=str, default="winrates.png", help="Output plot filename.")
    parser.add_argument("--num_games", type=int, default=8, help="Number of games per deck pair.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for simulation.")
    parser.add_argument("--cache_file", type=str, default="winrates.json", help="Cache file for win rates.")

    args = parser.parse_args()

    # Load Control Checkpoint
    try:
        control_params, control_config, control_step = load_checkpoint(args.control_checkpoint)
        print(f"Loaded control checkpoint from step {control_step}")
    except Exception as e:
        print(f"Failed to load control checkpoint: {e}")
        sys.exit(1)

    # Load Cache
    cache = {}
    if os.path.exists(args.cache_file):
        with open(args.cache_file, 'r') as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Could not decode cache file. Starting fresh.")

    # Find checkpoints
    files = os.listdir(args.checkpoint_dir)
    checkpoints = []
    for f in files:
        match = re.search(r"checkpoint_(\d+).pkl", f)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, os.path.join(args.checkpoint_dir, f)))

    checkpoints.sort(key=lambda x: x[0])
    print(f"Found {len(checkpoints)} checkpoints.")

    # Prepare Plotting
    plot_data = {} # Key: "DeckA vs DeckB", Value: (steps, win_rates)

    # Generate Pairs
    deck_pairs = list(itertools.product(args.decks, args.decks))

    for step, cp_path in checkpoints:
        print(f"\nProcessing Checkpoint Step {step}...")

        try:
            params, config, loaded_step = load_checkpoint(cp_path)

            for d1, d2 in deck_pairs:
                d1_name = get_deck_name(d1)
                d2_name = get_deck_name(d2)

                pair_key = f"{d1_name}_vs_{d2_name}"
                # Include control step in cache key to invalidate if control changes
                cache_key = f"{step}_{pair_key}_vs_control_{control_step}"

                if cache_key in cache:
                    print(f"  [Cached] {pair_key}")
                    res = cache[cache_key]
                else:
                    print(f"  Evaluating {pair_key}...")
                    # P1 = Target (params), P2 = Control (control_params)
                    res = evaluate_pair(params, control_params, config, d1, d2, args.num_games, args.batch_size)
                    cache[cache_key] = res

                    # Update cache file immediately
                    with open(args.cache_file, 'w') as f:
                        json.dump(cache, f, indent=2)

                # Calculate win rate for P1
                total = res['total']
                if total > 0:
                    win_rate = res['p1_wins'] / total
                else:
                    win_rate = 0.0

                if pair_key not in plot_data:
                    plot_data[pair_key] = {'steps': [], 'win_rates': []}

                plot_data[pair_key]['steps'].append(step)
                plot_data[pair_key]['win_rates'].append(win_rate)

        except Exception as e:
            print(f"Error processing checkpoint {cp_path}: {e}")
            import traceback
            traceback.print_exc()

    # Plot
    plt.figure(figsize=(12, 8))
    for label, data in plot_data.items():
        plt.plot(data['steps'], data['win_rates'], marker='o', label=label)

    plt.xlabel('Checkpoint Step')
    plt.ylabel('Win Rate (Target P1 vs Control P2)')
    plt.title(f'Win Rates vs Control (Step {control_step})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)

    print(f"Saving plot to {args.output}")
    plt.savefig(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
