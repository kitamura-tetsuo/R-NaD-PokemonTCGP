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

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the game tree with one AI side.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file.")
    parser.add_argument("--deck_id_1", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck 1 file.")
    parser.add_argument("--deck_id_2", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck 2 file.")
    parser.add_argument("--output", type=str, default="tree.html", help="Path to output HTML file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for chance nodes (if not exploring all) and initial split.")
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='gpu', help="Device for inference.")
    parser.add_argument("--disable_jit", action="store_true", help="Disable JIT compilation.")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth of the tree.")
    parser.add_argument("--ai_player", type=int, default=0, help="Which player is controlled by AI (0 or 1).")
    parser.add_argument("--explore_all_chance", action="store_true", help="Whether to explore all chance outcomes (can explode).")
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

def generate_html(tree_data, output_path):
    tree_json = json.dumps(tree_data)
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DeckGym Tree Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: sans-serif; background-color: #f8f9fa; margin: 0; display: flex; height: 100vh; overflow: hidden; }}
        #tree-container {{ flex: 1; border-right: 1px solid #ddd; position: relative; cursor: grab; }}
        #tree-container:active {{ cursor: grabbing; }}
        #detail-panel {{ width: 400px; padding: 20px; overflow-y: auto; background: white; box-shadow: -2px 0 5px rgba(0,0,0,0.1); }}
        .node circle {{ fill: #fff; stroke: steelblue; stroke-width: 3px; cursor: pointer; }}
        .node text {{ font: 12px sans-serif; }}
        .link {{ fill: none; stroke: #ccc; stroke-width: 2px; }}
        .node--ai circle {{ stroke: #f03030; }}
        .node--chance circle {{ stroke: #30f030; }}
        .node--terminal circle {{ fill: #555; }}
        
        /* Card styles from battle.py */
        .ptcg-symbol {{ display: inline-block; width: 1.2em; text-align: center; border-radius: 50%; font-weight: bold; color: white; text-shadow: 1px 1px 1px black; margin-right: 2px; }}
        .type-grass {{ background-color: #78C850; }} .type-fire {{ background-color: #F08030; }} .type-water {{ background-color: #6890F0; }}
        .type-lightning {{ background-color: #F8D030; color: black; text-shadow: none; }} .type-psychic {{ background-color: #F85888; }}
        .type-fighting {{ background-color: #C03028; }} .type-darkness {{ background-color: #705848; }}
        .type-metal {{ background-color: #B8B8D0; color: black; text-shadow: none; }} .type-colorless {{ background-color: #A8A878; color: black; text-shadow: none; }}

        .board {{ display: flex; flex-direction: column; gap: 10px; }}
        .player-area {{ border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #fefefe; }}
        .zone {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 5px; align-items: flex-start; }}
        .card-container {{ position: relative; width: 40px; }}
        .card-img {{ width: 100%; border-radius: 3px; }}
        .card-stats {{ position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); color: white; font-size: 8px; padding: 1px; text-align: center; }}
        .node-label {{ background: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; font-size: 10px; }}
    </style>
</head>
<body>
    <div id="tree-container"></div>
    <div id="detail-panel">
        <h3>Node Details</h3>
        <p>Click a node to see the game state here.</p>
        <div id="state-display"></div>
    </div>

    <script>
        const treeData = {tree_json};
        
        const width = document.getElementById('tree-container').clientWidth;
        const height = document.getElementById('tree-container').clientHeight;
        const margin = {{ top: 20, right: 90, bottom: 30, left: 90 }};

        const svg = d3.select("#tree-container").append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .call(d3.zoom().on("zoom", (e) => g.attr("transform", e.transform)))
            .append("g");

        const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);

        const tree = d3.tree().nodeSize([40, 200]);
        let root = d3.hierarchy(treeData, d => d.children);

        update(root);

        function update(source) {{
            const nodes = root.descendants();
            const links = root.links();

            tree(root);

            const node = g.selectAll(".node")
                .data(nodes, d => d.data.id || (d.data.id = Math.random()));

            const nodeEnter = node.enter().append("g")
                .attr("class", d => `node ${{d.data.is_ai ? 'node--ai' : ''}} ${{d.data.is_chance ? 'node--chance' : ''}} ${{d.data.is_terminal ? 'node--terminal' : ''}}`)
                .attr("transform", d => `translate(${{d.y}},${{d.x}})`)
                .on("click", (e, d) => showDetails(d.data));

            nodeEnter.append("circle").attr("r", 10);

            nodeEnter.append("text")
                .attr("dy", ".35em")
                .attr("x", d => d.children ? -13 : 13)
                .attr("text-anchor", d => d.children ? "end" : "start")
                .text(d => d.data.action_name);

            const nodeUpdate = nodeEnter.merge(node);
            nodeUpdate.transition().duration(200).attr("transform", d => `translate(${{d.y}},${{d.x}})`);

            const link = g.selectAll(".link")
                .data(links, d => d.target.data.id);

            link.enter().insert("path", "g")
                .attr("class", "link")
                .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

            link.transition().duration(200).attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));
        }}

        function showDetails(data) {{
            const display = document.getElementById('state-display');
            if (!data.state) {{
                display.innerHTML = "<p>No state data for this node.</p>";
                return;
            }}
            
            const state = data.state;
            let html = `<div><strong>Step:</strong> ${{data.step}} | <strong>Turn:</strong> ${{state.turn}}</div>`;
            html += `<div><strong>Acting Player:</strong> Player ${{data.acting_player + 1}}</div>`;
            if (data.is_ai) html += `<div style="color:red">AI Decision Node</div>`;
            
            html += '<div class="board">';
            state.players.forEach((p, i) => {{
                html += `<div class="player-area">
                    <strong>Player ${{i+1}} (Points: ${{state.points[i]}})</strong><br>
                    Deck: ${{p.deck_size}} | Discard: ${{p.discard_pile_size}}<br>
                    <div class="zone">Active: ${{renderCard(p.active)}}</div>
                    <div class="zone">Bench: ${{p.bench.map(renderCard).join('')}}</div>
                    <div class="zone">Hand (${{p.hand.length}}): ${{p.hand.map(renderCard).join('')}}</div>
                </div>`;
            }});
            html += '</div>';
            display.innerHTML = html;
        }}

        function renderCard(card) {{
            if (!card) return '<div class="card-container" style="border:1px dashed #ccc;height:56px;"></div>';
            return `<div class="card-container">
                <img src="${{card.url}}" class="card-img" title="${{card.name}}">
                ${{card.hp ? `<div class="card-stats">HP:${{card.hp}}</div>` : ''}}
            </div>`;
        }}
    </script>
</body>
</html>
    """
    with open(output_path, "w") as f: f.write(html_content)

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

    def explore(state, depth, step, action_name):
        nonlocal node_id_counter
        node_id_counter += 1
        node = {
            "id": node_id_counter,
            "action_name": action_name,
            "step": step,
            "acting_player": state.current_player(),
            "children": [],
            "is_terminal": state.is_terminal(),
            "state": extract_state_info(state.rust_game.get_state())
        }

        if state.is_terminal() or depth >= args.max_depth:
            return node

        if state.is_chance_node():
            node["is_chance"] = True
            outcomes = state.chance_outcomes()
            if args.explore_all_chance:
                for action, prob in outcomes:
                    child_state = state.clone()
                    child_state.apply_action(action)
                    node["children"].append(explore(child_state, depth + 1, step + 1, f"Chance (p={prob:.2f})"))
            else:
                # Sample one chance outcome to avoid explosion
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
                child_state = state.clone()
                child_state.apply_action(action)
                node["children"].append(explore(child_state, depth + 1, step + 1, "Chance (Sampled)"))
        else:
            curr_p = state.current_player()
            if curr_p == args.ai_player:
                node["is_ai"] = True
                logits = predict(state)
                action = int(np.argmax(logits))
                child_state = state.clone()
                child_state.apply_action(action)
                node["children"].append(explore(child_state, depth + 1, step + 1, state.action_to_string(curr_p, action)))
            else:
                for action in state.legal_actions():
                    child_state = state.clone()
                    child_state.apply_action(action)
                    node["children"].append(explore(child_state, depth + 1, step + 1, state.action_to_string(curr_p, action)))
        
        return node

    logging.info("Starting tree exploration...")
    initial_state = game.new_initial_state()
    root_node = explore(initial_state, 0, 0, "Root")
    
    logging.info(f"Generating HTML: {args.output}")
    generate_html(root_node, args.output)

if __name__ == "__main__":
    main()
