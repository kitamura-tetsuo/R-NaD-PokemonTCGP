import argparse
import sys
import os
import json
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run a battle between two agents.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (optional). If not provided, uses random weights.")
    parser.add_argument("--deck_id_1", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck 1 file.")
    parser.add_argument("--deck_id_2", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck 2 file.")
    parser.add_argument("--output", type=str, default="battle.html", help="Path to output HTML file.")
    parser.add_argument("--seed", type=int, default=int(datetime.datetime.now().timestamp()), help="Random seed.")
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='gpu', help="Device to run inference on (cpu or gpu).")
    parser.add_argument("--disable_jit", action="store_true", help="Disable JIT compilation to save memory.")
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
        "winner": None,
        "players": []
    }

    if rust_state.is_game_over():
        outcome = rust_state.winner
        info["winner"] = {
            "winner": outcome.winner,
            "is_tie": outcome.is_tie
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
            tool_info = None
            if hasattr(active, "attached_tool") and active.attached_tool:
                tool = active.attached_tool
                tool_info = {
                    "id": tool.id,
                    "name": tool.name,
                    "url": get_card_image_url(tool.id)
                }

            p_info["active"] = {
                "id": active.card.id,
                "name": active.name,
                "url": get_card_image_url(active.card.id),
                "hp": active.remaining_hp,
                "max_hp": active.total_hp,
                "energy": str(active.attached_energy),
                "tool": tool_info,
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

            tool_info = None
            if hasattr(mon, "attached_tool") and mon.attached_tool:
                tool = mon.attached_tool
                tool_info = {
                    "id": tool.id,
                    "name": tool.name,
                    "url": get_card_image_url(tool.id)
                }

            p_info["bench"].append({
                "id": mon.card.id,
                "name": mon.name,
                "url": get_card_image_url(mon.card.id),
                "hp": mon.remaining_hp,
                "max_hp": mon.total_hp,
                "energy": str(mon.attached_energy),
                "tool": tool_info,
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

        .board {{ display: flex; flex-direction: row; gap: 20px; align-items: flex-start; }}
        .player-area {{ border: 1px solid #ccc; padding: 10px; border-radius: 5px; flex: 1; min-width: 0; }}
        .player-area.current {{ border-color: #007bff; border-width: 2px; background-color: #f8f9ff; }}

        .area-title {{ font-weight: bold; margin-bottom: 5px; }}

        .zone {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 5px; align-items: flex-start; min-height: 80px; }}
        .zone-title {{ width: 50px; font-size: 12px; color: #666; font-weight: bold; }}

        .card-container {{ position: relative; width: 50px; }}
        .card-img {{ width: 100%; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); }}
        .card-stats {{ position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); color: white; font-size: 10px; padding: 2px; text-align: center; }}
        .status-icon {{ position: absolute; top: 0; right: 0; background: red; color: white; border-radius: 50%; width: 15px; height: 15px; font-size: 10px; display: flex; align-items: center; justify-content: center; }}
        .tool-icon {{ position: absolute; top: 20px; right: -10px; width: 25px; height: 35px; z-index: 10; }}
        .tool-img {{ width: 100%; height: 100%; border-radius: 3px; box-shadow: 1px 1px 3px rgba(0,0,0,0.3); border: 1px solid white; }}

        .log {{ margin-top: 20px; max-height: 150px; overflow-y: auto; background: #eee; padding: 10px; font-family: monospace; }}
        .chart-container {{ margin-top: 20px; height: 300px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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

        <div class="chart-container">
            <canvas id="evalChart"></canvas>
        </div>

        <div class="log" id="log-display"></div>
        <div class="log" id="full-stats-display" style="max-height: 300px;"></div>
    </div>

    <script>
        const history = {history_json};
        let currentStep = 0;

        function renderCard(card, type="hand") {{
            if (!card) return '<div class="card-container" style="border: 1px dashed #ccc; height: 70px;"></div>';

            let statsHtml = '';
            if (type === 'active' || type === 'bench') {{
                statsHtml = `<div class="card-stats">HP: ${{card.hp}}/${{card.max_hp}}<br>E: ${{card.energy}}</div>`;
            }}

            let statusHtml = '';
            if (card.status && card.status.length > 0) {{
                statusHtml = `<div class="status-icon" title="${{card.status.join(', ')}}">!</div>`;
            }}

            let toolHtml = '';
            if (card.tool) {{
                toolHtml = `<div class="tool-icon" title="${{card.tool.name}}"><img src="${{card.tool.url}}" class="tool-img" onerror="this.src='https://via.placeholder.com/25x35?text=T'"></div>`;
            }}

            return `
                <div class="card-container">
                    <img src="${{card.url}}" alt="${{card.name}}" class="card-img" onerror="this.src='https://via.placeholder.com/50x70?text=${{encodeURIComponent(card.name)}}'">
                    ${{toolHtml}}
                    ${{statsHtml}}
                    ${{statusHtml}}
                </div>
            `;
        }}

        function renderPlayer(pIndex, state) {{
            const p = state.players[pIndex];
            const isCurrent = state.acting_player === pIndex;

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

            document.getElementById('board').innerHTML = p1Html + p2Html; 

            // Update Log
            let logText = `Action: ${{state.action_name || "Start"}}`;
            if (state.top_candidates && state.top_candidates.length > 0) {{
                logText += '\\n\\nTop alternatives:';
                state.top_candidates.forEach(cand => {{
                    logText += `\\n- ${{cand.name}}: ${{ (cand.prob * 100).toFixed(1) }}%`;
                }});
            }}
            document.getElementById('log-display').innerText = logText;

            // Generate Full Stats Table
            let tableHtml = '<table style="width:100%; border-collapse: collapse;"><thead><tr><th style="border-bottom: 1px solid #ddd; text-align: left;">Action</th><th style="border-bottom: 1px solid #ddd; text-align: right;">Probability</th></tr></thead><tbody>';
            if (state.all_candidates && state.all_candidates.length > 0) {{
                state.all_candidates.forEach(cand => {{
                    const isSelected = (cand.name === state.action_name);
                    const bg = isSelected ? 'background-color: #bbdefb;' : '';
                    const weight = isSelected ? 'font-weight: bold;' : '';
                    tableHtml += `<tr style="${{bg}}${{weight}}"><td style="padding: 4px; border-bottom: 1px solid #eee;">${{cand.name}}</td><td style="padding: 4px; border-bottom: 1px solid #eee; text-align: right;">${{(cand.prob * 100).toFixed(2)}}%</td></tr>`;
                }});
            }} else {{
                 tableHtml += '<tr><td colspan="2">No probability data available</td></tr>';
            }}
            tableHtml += '</tbody></table>';
            document.getElementById('full-stats-display').innerHTML = tableHtml;

            updateChart();
        }}

        let chart;
        function initChart() {{
            const ctx = document.getElementById('evalChart').getContext('2d');
            const labels = history.map((_, i) => i);
            const evalP1 = history.map(h => h.eval_0 || 0);
            const evalP2 = history.map(h => h.eval_1 || 0);

            chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Player 1 Eval',
                            data: evalP1,
                            borderColor: 'blue',
                            fill: false,
                            tension: 0.1
                        }},
                        {{
                            label: 'Player 2 Eval',
                            data: evalP2,
                            borderColor: 'red',
                            fill: false,
                            tension: 0.1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: false,
                            title: {{ display: true, text: 'V-Value' }}
                        }},
                        x: {{
                            title: {{ display: true, text: 'Step' }}
                        }}
                    }},
                    plugins: {{
                        annotation: {{
                            annotations: {{
                                line1: {{
                                    type: 'line',
                                    xMin: currentStep,
                                    xMax: currentStep,
                                    borderColor: 'black',
                                    borderWidth: 2,
                                    label: {{
                                        content: 'Current',
                                        enabled: true,
                                        position: 'top'
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}

        // We need the annotation plugin for the vertical line
        // But for simplicity, we can just use a custom plugin or just redraw the line.
        // Let's use a simpler approach: vertical line plugin
        const verticalLinePlugin = {{
            id: 'verticalLine',
            afterDraw: (chart) => {{
                const ctx = chart.ctx;
                const xAxis = chart.scales.x;
                const yAxis = chart.scales.y;
                const x = xAxis.getPixelForValue(currentStep);

                ctx.save();
                ctx.beginPath();
                ctx.moveTo(x, yAxis.top);
                ctx.lineTo(x, yAxis.bottom);
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
                ctx.stroke();
                ctx.restore();
            }}
        }};
        Chart.register(verticalLinePlugin);

        function updateChart() {{
            if (chart) {{
                // Update the vertical line position
                if (chart.options.plugins.annotation && chart.options.plugins.annotation.annotations.line1) {{
                    chart.options.plugins.annotation.annotations.line1.xMin = currentStep;
                    chart.options.plugins.annotation.annotations.line1.xMax = currentStep;
                }}
                chart.update();
            }}
        }}

        initChart();

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

        document.addEventListener('keydown', (e) => {{
            if (e.key === "ArrowLeft") prevTurn();
            if (e.key === "ArrowRight") nextTurn();
        }});

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

    # JAX Configuration
    if args.device == 'cpu':
        jax.config.update("jax_platform_name", "cpu")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        logging.info("Forced use of CPU for inference.")
    elif args.device == 'gpu':
        # Even if GPU, we might want to be careful with memory if the user is concerned
        # But let's stick to default behavior unless they asked for CPU, 
        # or maybe set preallocate=false just in case if they didn't specify but we are in this context.
        # User asked: "GPUメモリやメインメモリにモデルが収まらない場合でも" -> implying CPU fallback or careful GPU usage.
        # Let's set preallocate=false by default for this script as it's not training.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        logging.info("JIT compilation disabled.")

    # Initialize Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # Load embedding matrix
    emb_path = "card_embeddings.npy"
    if os.path.exists(emb_path):
        embedding_matrix = jnp.array(np.load(emb_path))
        logging.info(f"Loaded embedding matrix from {emb_path}, shape: {embedding_matrix.shape}")
    else:
        logging.warning(f"Embedding matrix not found at {emb_path}. Using zero matrix.")
        embedding_matrix = jnp.zeros((10000, 26))

    # Config
    config = RNaDConfig(
        deck_id_1=args.deck_id_1,
        deck_id_2=args.deck_id_2
    )

    logging.info(f"Initializing Game with deck1={args.deck_id_1}, deck2={args.deck_id_2}")
    game = pyspiel.load_game(
        "deckgym_ptcgp",
        {
            "deck_id_1": config.deck_id_1,
            "deck_id_2": config.deck_id_2,
            "seed": args.seed,
            "max_game_length": config.unroll_length
        }
    )
    num_actions = game.num_distinct_actions()
    obs_shape = game.observation_tensor_shape()

    predict_fn = None
    params = None
    jit_apply = None

    checkpoint_path = args.checkpoint
    is_saved_model = False

    if checkpoint_path and os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
        # Check if directory contains JAX checkpoints
        latest_ckpt = find_latest_checkpoint(checkpoint_path)
        if latest_ckpt:
            logging.info(f"Found latest JAX checkpoint in directory: {latest_ckpt}")
            checkpoint_path = latest_ckpt
        else:
            is_saved_model = True

    if is_saved_model:
        # SavedModel Path
        logging.info(f"Loading SavedModel from: {args.checkpoint}")
        import tensorflow as tf
        if args.device == 'cpu':
            tf.config.set_visible_devices([], 'GPU')
            logging.info("Forced use of CPU for TensorFlow SavedModel.")
        loaded_model = tf.saved_model.load(args.checkpoint)

        def tf_predict(obs):
            # obs: (Batch, Dim)
            # The SavedModel export defines a 'predict' function returning a dict
            if hasattr(loaded_model, 'predict'):
                out = loaded_model.predict(obs)
            else:
                # Fallback if just __call__
                out = loaded_model(obs)
            
            # We expect out to be dict {'policy': logits, 'value': value}
            if isinstance(out, dict):
                return out['policy'].numpy(), out['value'].numpy()
            else:
                # Assume tuple (logits, value)
                return out[0].numpy(), out[1].numpy()

        predict_fn = tf_predict
    else:
        # JAX Checkpoint or Random
        def forward(x):
            # We need to decide model type. RNaDConfig defaults to "transformer".
            # Checkpoints might contain config, but let's assume default or what user passed if we added args for it (we didn't).
            # We will try to load config from checkpoint if available.
            if config.model_type == "transformer":
                net = CardTransformerNet(
                    num_actions=num_actions,
                    embedding_matrix=embedding_matrix,
                    hidden_size=config.transformer_embed_dim,
                    num_blocks=config.transformer_layers,
                    num_heads=config.transformer_heads,
                )
            else:
                net = DeckGymNet(
                    num_actions=num_actions,
                    hidden_size=config.hidden_size,
                    num_blocks=config.num_blocks
                )
            return net(x)

        network = hk.transform(forward)

        # Initialize Params
        dummy_obs = jnp.zeros((1, *obs_shape))
        params = network.init(rng, dummy_obs)

        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            try:
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Load params
                params = data['params']
                
                # Update config from checkpoint if present
                if 'config' in data:
                    loaded_config = data['config']
                    # Use loaded config logic:
                    ckpt_config = data['config']
                    logging.info(f"Checkpoint config found. Model type: {ckpt_config.model_type}")

                    def forward_ckpt(x):
                        if ckpt_config.model_type == "transformer":
                            net = CardTransformerNet(
                                num_actions=num_actions,
                                embedding_matrix=embedding_matrix,
                                hidden_size=ckpt_config.transformer_embed_dim,
                                num_blocks=ckpt_config.transformer_layers,
                                num_heads=ckpt_config.transformer_heads,
                            )
                        else:
                            net = DeckGymNet(
                                num_actions=num_actions,
                                hidden_size=ckpt_config.hidden_size,
                                num_blocks=ckpt_config.num_blocks
                            )
                        return net(x)

                    network = hk.transform(forward_ckpt)

            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")
                logging.info("Using random weights instead.")
        else:
            logging.info("Using random weights (no checkpoint provided or found).")

        jit_apply_fn = jax.jit(network.apply) if not args.disable_jit else network.apply
        jit_apply = jit_apply_fn # Keep ref

        def jax_predict(obs):
            return jit_apply_fn(params, rng, obs)

        predict_fn = jax_predict

    # Start Game
    # game = learner.game # Error here previously
    # Game is already initialized as `game`
    state = game.new_initial_state()

    history = []

    # Record initial state
    initial_info = extract_state_info(state.rust_game.get_state())
    initial_info["action_name"] = "Game Start"
    initial_info["acting_player"] = state.current_player()
    
    # Initial evaluations
    # Initial evaluations

    for p in [0, 1]:
        obs_p = state.observation_tensor(p)
        obs_p_batched = np.array(obs_p)[None, ...]
        _, val_p = predict_fn(obs_p_batched)
        initial_info[f"eval_{p}"] = float(val_p[0, 0])
        
    history.append(initial_info)

    step_count = 0
    max_steps = 200 # Safety limit

    while not state.is_terminal() and step_count < max_steps:
        step_count += 1

        current_player = state.current_player()
        top_candidates = []
        all_candidates = []

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

            logits, _ = predict_fn(obs_batched)
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

            # Get top 3 alternative candidates
            sorted_indices = np.argsort(probs)[::-1]
            for idx in sorted_indices:
                if probs[idx] <= 0:
                    continue
                cand_name = state.action_to_string(current_player, idx)

                # Add to all candidates
                all_candidates.append({"name": cand_name, "prob": float(probs[idx])})

                # Add to top candidates (skipping selected action)
                if idx != action and len(top_candidates) < 3:
                    top_candidates.append({"name": cand_name, "prob": float(probs[idx])})

            state.apply_action(action)

        # Record State
        info = extract_state_info(state.rust_game.get_state())
        info["action_name"] = action_name
        info["acting_player"] = current_player
        info["top_candidates"] = top_candidates
        info["all_candidates"] = all_candidates
        
        # Evaluations
        for p in [0, 1]:
            obs_p = state.observation_tensor(p)
            obs_p_batched = np.array(obs_p)[None, ...]
            _, val_p = predict_fn(obs_p_batched)
            info[f"eval_{p}"] = float(val_p[0, 0])
            
        history.append(info)

    logging.info(f"Game over. Winner: {state.rust_game.get_state().winner}")

    # Generate HTML
    generate_html(history, args.output)

if __name__ == "__main__":
    main()
