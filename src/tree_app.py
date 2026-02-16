import streamlit as st
import sqlite3
import json
import argparse
import os
import glob
import graphviz
from streamlit_agraph import agraph, Node, Edge, Config

# Set page config to use wide mode
st.set_page_config(layout="wide", page_title="DeckGym Tree Viz")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".", help="Root directory to search for SQLite databases")
    parser.add_argument("--db_path", type=str, default=None, help="Path to specific SQLite database (optional override)")
    return parser.parse_args()

def find_db_files(root_dir):
    """Recursively find all .sqlite and .db files in the given directory."""
    db_files = []
    # extensions = ['*.sqlite', '*.db', '*.sqlite3']
    # But os.walk is better for recursion without complex glob patterns
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.sqlite', '.db', '.sqlite3')):
                db_files.append(os.path.join(root, file))
    return sorted(db_files)


# @st.cache_resource
def get_db_connection(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_node(conn, node_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
    return cursor.fetchone()

def get_children(conn, node_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM edges WHERE parent_id = ?", (int(node_id),))
    res = cursor.fetchall()
    return res

def render_card(card, width=100):
    if not card:
        st.write("Empty")
        return
    
    st.image(card['url'], width=width, caption=card['name'])
    if 'hp' in card:
        st.caption(f"HP: {card['hp']}/{card['max_hp']}")
    if 'energy' in card and card['energy']:
        st.caption(f"Energy: {', '.join(card['energy'])}")
    if 'status' in card and card['status']:
        st.caption(f"Status: {', '.join(card['status'])}")

def render_player(player_info, player_idx, points):
    st.subheader(f"Player {player_idx + 1} (Points: {points})")
    st.write(f"Deck: {player_info['deck_size']} | Discard: {player_info['discard_pile_size']}")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.write("Active:")
        if player_info['active']:
            render_card(player_info['active'])
        else:
            st.write("None")
    
    with col2:
        st.write("Bench:")
        if player_info['bench']:
            cols = st.columns(len(player_info['bench']))
            for i, mon in enumerate(player_info['bench']):
                with cols[i]:
                    render_card(mon)
        else:
            st.write("Empty")
    
    st.write(f"Hand ({len(player_info['hand'])}):")
    if player_info['hand']:
        cols = st.columns(min(len(player_info['hand']), 10))
        for i, card in enumerate(player_info['hand']):
            if i < 10:
                with cols[i]:
                    render_card(card, width=80)
            else:
                 # overflow handling essentially
                 pass

def main():
    try:
        # Streamlit doesn't handle argparse well when run with `streamlit run`
        # We'll rely on a default or specific env var, or just argument parsing if run as `python -m streamlit run`
        # But standard way is `streamlit run app.py -- --db_path foo.db`
        args = parse_args()
    except SystemExit:
        os._exit(0)

    # Sidebar: DB Selection
    st.sidebar.header("Database Selection")
    
    # Find DBs
    db_files = find_db_files(args.dir)
    
    # Add args.db_path if provided and valid
    if args.db_path and os.path.exists(args.db_path) and args.db_path not in db_files:
        db_files.insert(0, args.db_path)
        
    if not db_files:
        st.error(f"No SQLite databases found in {args.dir}. Please run tree_viz.py first or specify --db_path.")
        # Fallback if specific path given but not found in dir (already handled above if exists)
        # If args.db_path was invalid, we are here.
        return

    # Determine default index
    default_idx = 0
    if args.db_path in db_files:
        default_idx = db_files.index(args.db_path)
        
    selected_db = st.sidebar.selectbox(
        "Select Database", 
        db_files, 
        index=default_idx,
        key="selected_db_path"
    )
    
    # Detect change to reset state
    if "last_db_path" not in st.session_state:
        st.session_state.last_db_path = selected_db
    elif st.session_state.last_db_path != selected_db:
         # Changed!
         st.session_state.last_db_path = selected_db
         # Reset node ID so we don't try to look up a node that doesn't exist or is wrong
         if 'current_node_id' in st.session_state:
             del st.session_state.current_node_id
         st.rerun()

    db_path = selected_db
    
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}.")
        return

    conn = get_db_connection(db_path)

    # Session State for navigation
    if 'current_node_id' not in st.session_state:
        st.session_state.current_node_id = 1 # Root usually 1 if initialized with counter=0 then +1
        # Check if node 1 exists, otherwise find min id
        cursor = conn.cursor()
        cursor.execute("SELECT min(id) as min_id FROM nodes")
        min_id = cursor.fetchone()['min_id']
        if min_id:
            st.session_state.current_node_id = min_id


    # Sidebar for navigation controls
    with st.sidebar:
        st.header("Navigation")
        
        # Debug info
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM nodes")
        n_nodes = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM edges")
        n_edges = cursor.fetchone()[0]
        st.info(f"DB Stats: {n_nodes} nodes, {n_edges} edges")
        
        node_id_input = st.number_input("Go to Node ID", value=int(st.session_state.current_node_id), step=1)
        if st.button("Go"):
            st.session_state.current_node_id = node_id_input
            st.rerun()

    # Main Content
    node = get_node(conn, st.session_state.current_node_id)
    
    if not node:
        st.error(f"Node {st.session_state.current_node_id} not found.")
        return

    # Basic Node Info
    st.title(f"Node {node['id']}")
    
    cols = st.columns(4)
    cols[0].metric("Step", node['step'])
    cols[1].metric("Turn", node['turn'])
    cols[2].metric("Acting Player", f"Player {node['acting_player'] + 1}")
    cols[3].metric("Action to reach", node['action_name'] or "Root")

    if node['is_repeated']:
        st.warning(f"This is a REPEATED state. See Node {node['repeated_node_id']}")
        if st.button(f"Jump to Node {node['repeated_node_id']}"):
            st.session_state.current_node_id = node['repeated_node_id']
            st.rerun()

    # State Visualization
    if node['state_json']:
        state = json.loads(node['state_json'])
        
        st.markdown("---")
        p1_col, p2_col = st.columns(2)
        
        with p1_col:
            render_player(state['players'][0], 0, state['points'][0])
        
        with p2_col:
            render_player(state['players'][1], 1, state['points'][1])
            
        if '_pending_chance' in state:
            st.info(f"Pending Chance: {state['_pending_chance']}")


    # Tree Navigation - Agraph
    st.markdown("---")
    st.header("Tree Navigation (Interactive)")

    # Data structures for Agraph
    nodes = []
    edges = []
    
    # Track existing nodes to avoid duplicates
    existing_node_ids = set()

    def add_agraph_node(n, level, is_current=False, is_ancestor=False):
        node_id = str(n['id'])
        if node_id in existing_node_ids:
             return
        
        # Filter Repeated - actually handled by caller mostly, but double check
        if n['is_repeated']:
             return

        player = n['acting_player'] + 1
        turn = n['turn']
        
        # Label
        label = f"Node {n['id']}\nP{player} Turn {turn}"
        
        color = "#FFFFFF" # Default white
        
        if is_current:
             label += "\n(Current)"
             color = "#FFD700" # Gold
        elif is_ancestor:
             color = "#90EE90" # LightGreen
        elif n['is_terminal']:
             # Determine Winner Color
             label += "\n(Game Over)"
             color = "#808080" # Default grey if unknown
             if n['state_json']:
                 try:
                     state = json.loads(n['state_json'])
                     if 'winner' in state and state['winner']:
                         winner_idx = state['winner'].get('winner')
                         if winner_idx == 0:
                             color = "#FF6B6B" # Red (P1)
                             label += "\nP1 Wins"
                         elif winner_idx == 1:
                             color = "#4D96FF" # Blue (P2)
                             label += "\nP2 Wins"
                 except:
                     pass
        
        # Add node
        nodes.append(Node(id=node_id, 
                          label=label, 
                          size=25, 
                          shape="box",
                          color=color,
                          level=level, 
                          font={'color': 'black'}))
        existing_node_ids.add(node_id)

    # 1. Ancestors
    ancestors = []
    curr_id = node['id']
    for _ in range(5): # Limit 5
        c = conn.cursor()
        c.execute("SELECT parent_id, action_name FROM edges WHERE child_id = ?", (curr_id,))
        parent_row = c.fetchone()
        if not parent_row:
            break
        
        parent_node = get_node(conn, parent_row['parent_id'])
        ancestors.insert(0, {
            'node': parent_node,
            'action': parent_row['action_name'],
            'child_id': curr_id
        })
        curr_id = parent_row['parent_id']

    # Ancestors start at level 0
    current_level_offset = len(ancestors)

    # Add ancestor nodes and edges
    for i, item in enumerate(ancestors):
        anc_node = item['node']
        # Ancestors are never repeated in the active path context
        add_agraph_node(anc_node, level=i, is_ancestor=True)
        
        source = str(anc_node['id'])
        target = str(item['child_id'])
        # Only add edge if both nodes exist (which they should)
        if source in existing_node_ids:
             # Target might not be in yet if it's the next node in list, but we add nodes in order
             # actually target is the child.
             # Note: We add nodes first, so edge targets must exist? 
             # Agraph doesn't strictly require order but it helps. 
             # The target for ancestors[i] is ancestors[i+1] (or current). 
             # We haven't added ancestors[i+1] yet in this loop? 
             # No, we iterate 0..N. 
             # Actually, target is `item['child_id']`. 
             # If `i` is the last ancestor, `child_id` is `node['id']`.
             # `node` is added AFTER this loop.
             # So we should add edges AFTER adding all nodes? 
             # Or just trust agraph handles it (it usually does).
             pass

        label = item['action']
        edges.append(Edge(source=source, 
                          label=label, 
                          target=target,
                          color="#000000"))

    # 2. Current Node
    add_agraph_node(node, level=current_level_offset, is_current=True)
    
    # 3. Descendants
    
    visited_edges = set() 
    
    def add_descendants(parent_id, current_base_level, current_depth, max_depth):
        if current_depth >= max_depth:
            return

        children = get_children(conn, parent_id)
        if not children:
            return

        for child in children:
            child_id = child['child_id']
            action = child['action_name']
            
            child_node = get_node(conn, child_id)
            if not child_node: 
                continue

            # SKIP REPEATED NODES
            if child_node['is_repeated']:
                continue

            child_level = current_base_level + 1
            add_agraph_node(child_node, level=child_level)
            
            edge_key = (parent_id, child_id, action)
            if edge_key not in visited_edges:
                edges.append(Edge(source=str(parent_id), 
                                  target=str(child_id), 
                                  label=action,
                                  color="#000000"))
                visited_edges.add(edge_key)
            
            # Recurse (already implicit by skipping repeated above)
            add_descendants(child_id, child_level, current_depth + 1, max_depth)

    add_descendants(node['id'], current_level_offset, 0, 3)

    # Agraph Config
    config = Config(width="100%", 
                    height=1200, 
                    directed=True, 
                    physics=False, 
                    hierarchical=True,
                    direction="LR",  # Add direction
                    nodeHighlightBehavior=True, 
                    highlightColor="#F7A7A6",
                    collapsible=False)

    # Render Agraph
    # It returns the id of the selected node
    selected_node_id = agraph(nodes=nodes, 
                              edges=edges, 
                              config=config)
    
    # Handle Navigation
    if selected_node_id:
        try:
            target_id = int(selected_node_id)
            if target_id != st.session_state.current_node_id:
                st.session_state.current_node_id = target_id
                st.rerun()
        except ValueError:
            pass

    st.caption("Click on nodes to navigate. Drag to move, scroll to zoom.")

if __name__ == "__main__":
    main()
