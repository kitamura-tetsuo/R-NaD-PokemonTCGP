import streamlit as st
import sqlite3
import json
import argparse
import os

# Set page config to use wide mode
st.set_page_config(layout="wide", page_title="DeckGym Tree Viz")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="tree.db", help="Path to SQLite database")
    return parser.parse_args()


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
    # Debug print
    st.write(f"DEBUG: Querying children for parent_id={node_id} (type: {type(node_id)})")
    cursor.execute("SELECT * FROM edges WHERE parent_id = ?", (int(node_id),))
    res = cursor.fetchall()
    st.write(f"DEBUG: Found {len(res)} children")
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

    db_path = args.db_path
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}. Please run tree_viz.py first.")
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
        return

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


    # Children / Actions
    st.markdown("---")
    st.header("Next Actions")
    
    # Debug: Check node ID type
    st.write(f"DEBUG: node['id'] = {node['id']} type={type(node['id'])}")
    
    children = get_children(conn, node['id'])
    
    if not children:
        st.write("Terminal Node or Leaf")
        # st.write("DEBUG: No children returned from DB")
    else:
        for child in children:
            col1, col2 = st.columns([3, 1])
            col1.write(f"**{child['action_name']}** -> Node {child['child_id']}")
            if col2.button("Explore", key=f"btn_{child['child_id']}"):
                st.session_state.current_node_id = child['child_id']
                st.rerun()

if __name__ == "__main__":
    main()
