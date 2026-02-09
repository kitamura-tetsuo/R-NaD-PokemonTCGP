import argparse
import sys
import os
import json
import random
import logging
import jax
import haiku as hk
import numpy as np
import pyspiel
import deckgym
import deckgym_openspiel
from src.rnad import RNaDLearner, RNaDConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run a battle between two agents.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (optional). If not provided, uses random weights.")
    parser.add_argument("--deck_id_1", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck 1 file.")
    parser.add_argument("--deck_id_2", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck 2 file.")
    parser.add_argument("--output", type=str, default="battle.html", help="Path to output HTML file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def get_card_image_url(card_id):
    """
    Constructs the Limitless TCG image URL from the card ID.
    Example ID: "A1 129" -> "https://limitlesstcg.nyc3.cdn.digitaloceanspaces.com/pocket/A1/A1_129_EN_SM.webp"
    Example ID: "P-A 001" -> "https://limitlesstcg.nyc3.cdn.digitaloceanspaces.com/pocket/P-A/P-A_001_EN_SM.webp"
    """
    parts = card_id.split()
    if len(parts) != 2:
        return "" # Or placeholder

    set_id = parts[0]
    card_num = parts[1]

    # URL Format: {Set}/{Set}_{ID}_EN_SM.webp
    # Note: URL uses underscore, ID uses space.
    url = f"https://limitlesstcg.nyc3.cdn.digitaloceanspaces.com/pocket/{set_id}/{set_id}_{card_num}_EN_SM.webp"
    return url

def extract_state_info(rust_state):
    """
    Extracts structured information from the Rust Game State object.
    """
    info = {
        "turn": rust_state.turn_count,
        "current_player": rust_state.current_player,
        "points": rust_state.points,
        "winner": rust_state.winner if rust_state.is_game_over else None,
        "players": []
    }

    for p in [0, 1]:
        p_info = {
            "hand": [],
            "active": None,
            "bench": [],
            "discard_pile_size": rust_state.get_discard_pile_size(p),
            "deck_size": rust_state.get_deck_size(p),
            "energy_attached": 0 # Placeholder if needed, individual cards have energy info
        }

        # Hand
        hand = rust_state.get_hand(p)
        for card in hand:
            p_info["hand"].append({
                "id": card.id,
                "name": card.name,
                "url": get_card_image_url(card.id)
            })

        # Active
        active = rust_state.get_active_pokemon(p)
        if active:
            p_info["active"] = {
                "id": active.card.id,
                "name": active.name,
                "url": get_card_image_url(active.card.id),
                "hp": active.remaining_hp,
                "max_hp": active.total_hp,
                "energy": str(active.attached_energy),
                "status": []
            }
            if active.poisoned: p_info["active"]["status"].append("Poisoned")
            if active.asleep: p_info["active"]["status"].append("Asleep")
            if active.paralyzed: p_info["active"]["status"].append("Paralyzed")
            # Check energy types? active.attached_energy might be simple count or detailed.
            # Rust binding shows 'attached_energy' property. Assuming int for now based on typical output.
            # Actually, attached_energy usually returns a map or list of types.
            # Let's inspect it later if needed. For now, just display it.

        # Bench
        bench = rust_state.get_bench_pokemon(p)
        for mon in bench:
            if not mon:
                continue
            p_info["bench"].append({
                "id": mon.card.id,
                "name": mon.name,
                "url": get_card_image_url(mon.card.id),
                "hp": mon.remaining_hp,
                "max_hp": mon.total_hp,
                "energy": str(mon.attached_energy),
                "status": [] # Bench usually has no status
            })

        info["players"].append(p_info)

    return info

def generate_html(history, output_path):
    """
    Generates an interactive HTML file to visualize the battle history.
    """

    # Serialize history to JSON for embedding
    history_json = json.dumps(history)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeckGym Battle Visualization</title>
    <style>
        body {{ font-family: sans-serif; background-color: #f0f0f0; margin: 0; padding: 20px; }}
        .container {{ max_width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .controls {{ text-align: center; margin-bottom: 20px; }}
        button {{ padding: 10px 20px; font-size: 16px; cursor: pointer; }}
        .status-bar {{ display: flex; justify-content: space-between; margin-bottom: 10px; font-weight: bold; }}

        .board {{ display: flex; flex-direction: column; gap: 20px; }}
        .player-area {{ border: 1px solid #ccc; padding: 10px; border-radius: 5px; }}
        .player-area.current {{ border-color: #007bff; border-width: 2px; }}

        .area-title {{ font-weight: bold; margin-bottom: 5px; }}

        .zone {{ display: flex; gap: 10px; margin-bottom: 10px; align-items: flex-start; min-height: 150px; }}
        .zone-title {{ width: 60px; font-size: 14px; color: #666; }}

        .card-container {{ position: relative; width: 100px; }}
        .card-img {{ width: 100%; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); }}
        .card-stats {{ position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); color: white; font-size: 10px; padding: 2px; text-align: center; }}
        .status-icon {{ position: absolute; top: 0; right: 0; background: red; color: white; border-radius: 50%; width: 15px; height: 15px; font-size: 10px; display: flex; align-items: center; justify-content: center; }}

        .log {{ margin-top: 20px; max-height: 150px; overflow-y: auto; background: #eee; padding: 10px; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DeckGym Battle Visualization</h1>

        <div class="controls">
            <button onclick="prevTurn()">Previous</button>
            <span id="turn-display">Turn: 0</span>
            <button onclick="nextTurn()">Next</button>
        </div>

        <div id="board" class="board">
            <!-- Board content rendered here -->
        </div>

        <div class="log" id="log-display"></div>
    </div>

    <script>
        const history = {history_json};
        let currentStep = 0;

        function renderCard(card, type="hand") {{
            if (!card) return '<div class="card-container" style="border: 1px dashed #ccc; height: 140px;"></div>';

            let statsHtml = '';
            if (type === 'active' || type === 'bench') {{
                statsHtml = `<div class="card-stats">HP: ${{card.hp}}/${{card.max_hp}}<br>E: ${{card.energy}}</div>`;
            }}

            let statusHtml = '';
            if (card.status && card.status.length > 0) {{
                statusHtml = `<div class="status-icon" title="${{card.status.join(', ')}}">!</div>`;
            }}

            return `
                <div class="card-container">
                    <img src="${{card.url}}" alt="${{card.name}}" class="card-img" onerror="this.src='https://via.placeholder.com/100x140?text=${{encodeURIComponent(card.name)}}'">
                    ${{statsHtml}}
                    ${{statusHtml}}
                </div>
            `;
        }}

        function renderPlayer(pIndex, state) {{
            const p = state.players[pIndex];
            const isCurrent = state.current_player === pIndex;

            let activeHtml = renderCard(p.active, 'active');

            let benchHtml = p.bench.map(c => renderCard(c, 'bench')).join('');

            let handHtml = p.hand.map(c => renderCard(c, 'hand')).join('');

            return `
                <div class="player-area ${{isCurrent ? 'current' : ''}}">
                    <div class="area-title">Player ${{pIndex + 1}} (Points: ${{state.points[pIndex]}}) - Deck: ${{p.deck_size}} - Discard: ${{p.discard_pile_size}}</div>

                    <div class="zone">
                        <div class="zone-title">Active</div>
                        ${{activeHtml}}
                    </div>

                    <div class="zone">
                        <div class="zone-title">Bench</div>
                        ${{benchHtml}}
                    </div>

                    <div class="zone">
                        <div class="zone-title">Hand</div>
                        ${{handHtml}}
                    </div>
                </div>
            `;
        }}

        function render() {{
            const state = history[currentStep];
            if (!state) return;

            document.getElementById('turn-display').innerText = `Step: ${{currentStep}} / ${{history.length - 1}} (Turn: ${{state.turn}})`;

            // Render P2 (top) then P1 (bottom) usually, or just 0 and 1
            const p1Html = renderPlayer(0, state);
            const p2Html = renderPlayer(1, state);

            document.getElementById('board').innerHTML = p2Html + p1Html; // P2 on top? Or strictly P1, P2. Let's do P2 then P1 so P1 is at bottom.

            // Update Log (optional, simple step info)
            document.getElementById('log-display').innerText = `Action: ${{state.action_name || "Start"}}`;
        }}

        function nextTurn() {{
            if (currentStep < history.length - 1) {{
                currentStep++;
                render();
            }}
        }}

        function prevTurn() {{
            if (currentStep > 0) {{
                currentStep--;
                render();
            }}
        }}

        // Initial render
        render();
    </script>
</body>
</html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)
    logging.info(f"HTML visualization saved to {output_path}")

def main():
    args = parse_args()

    # Initialize Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # Config
    config = RNaDConfig(
        deck_id_1=args.deck_id_1,
        deck_id_2=args.deck_id_2
    )

    logging.info(f"Initializing Learner with deck1={args.deck_id_1}, deck2={args.deck_id_2}")
    learner = RNaDLearner("deckgym_ptcgp", config)
    learner.init(rng)

    if args.checkpoint and os.path.exists(args.checkpoint):
        logging.info(f"Loading checkpoint: {args.checkpoint}")
        learner.load_checkpoint(args.checkpoint)
    else:
        logging.info("Using random weights (no checkpoint provided or found).")

    # Start Game
    game = learner.game
    state = game.new_initial_state()

    history = []

    # Record initial state
    initial_info = extract_state_info(state.rust_game.get_state())
    initial_info["action_name"] = "Game Start"
    history.append(initial_info)

    step_count = 0
    max_steps = 200 # Safety limit

    while not state.is_terminal() and step_count < max_steps:
        step_count += 1

        current_player = state.current_player()

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            action_name = f"Chance: {state.action_to_string(current_player, action)}"
            state.apply_action(action)
        else:
            # Agent Action
            obs = state.observation_tensor(current_player)
            obs = np.array(obs) # Convert to numpy for haiku

            # Use learner network
            # obs shape is (dim,), add batch dim (1, dim)
            obs_batched = obs[None, ...]

            logits, _ = learner.network.apply(learner.params, rng, obs_batched)
            logits = np.array(logits[0]) # Remove batch dim

            legal_mask = np.zeros_like(logits, dtype=bool)
            legal_actions = state.legal_actions()
            legal_mask[legal_actions] = True

            # Mask illegal actions
            logits[~legal_mask] = -1e9

            # Simple greedy or sample? Let's sample to be interesting
            probs = jax.nn.softmax(logits)
            probs = np.array(probs)
            probs = probs / probs.sum() # Renormalize just in case

            action = np.random.choice(len(probs), p=probs)
            action_name = state.action_to_string(current_player, action)

            state.apply_action(action)

        # Record State
        info = extract_state_info(state.rust_game.get_state())
        info["action_name"] = action_name
        history.append(info)

    logging.info(f"Game over. Winner: {state.rust_game.get_state().winner}")

    # Generate HTML
    generate_html(history, args.output)

if __name__ == "__main__":
    main()
