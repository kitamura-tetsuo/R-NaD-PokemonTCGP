import sys
import os
import collections

# Add deckgym to path
sys.path.append(os.path.join(os.getcwd(), "deckgym-core/python"))
sys.path.append(os.getcwd())

import deckgym
import deckgym.deckgym

# Manually register PyBatchedSimulator if needed (copied from rnad.py logic)
if not hasattr(deckgym, 'PyBatchedSimulator'):
    if hasattr(deckgym, 'deckgym') and hasattr(deckgym.deckgym, 'PyBatchedSimulator'):
         deckgym.PyBatchedSimulator = deckgym.deckgym.PyBatchedSimulator
    else:
         try:
             from deckgym.deckgym import PyBatchedSimulator
             deckgym.PyBatchedSimulator = PyBatchedSimulator
         except ImportError:
             pass

def test_turn_order():
    deck1 = "deckgym-core/example_decks/mewtwoex.txt"
    deck2 = "deckgym-core/example_decks/blastoiseex.txt"
    
    # Initialize simulator with small batch
    batch_size = 100
    sim = deckgym.PyBatchedSimulator(
        deck1, deck2, batch_size, 1.0, 0.0, 0.0
    )
    
    # Reset
    # deck_ids_1 and deck_ids_2 are optional in reset if provided in init, 
    # but we can pass them explicitly to be safe, or just rely on init.
    # rnad.py generates them via league_config or passes them.
    # Let's pass them explicitly effectively simulating what rnad.py does.
    
    deck_ids_1 = [deck1] * batch_size
    deck_ids_2 = [deck2] * batch_size
    
    # We run multiple resets to get better statistics
    total_games = 0
    deck1_first_count = 0
    
    for _ in range(10):
        obs = sim.reset(seed=None, deck_ids_1=deck_ids_1, deck_ids_2=deck_ids_2)
        
        # obs structure: Vec<Vec<f32>>
        # We need to decode the observation to know who is the current player?
        # Or we can rely on `sim.games[i].state().current_player` if exposed?
        # PyBatchedSimulator doesn't expose the games directly in Python.
        # But `reset` returns observations for the *current player*.
        
        # Wait, I cannot easily tell who is current player from the observation vector alone without knowing the encoding scheme deeply.
        # BUT, PyBatchedSimulator checks `game.state().current_player`.
        
        # I might need to add a method to PyBatchedSimulator to debug this, OR rely on `PyGame`.
        pass
        
    print("Cannot easily verify with PyBatchedSimulator directly without encoding knowledge.")
    print("Switching to valid PyGame verification.")

    from deckgym.deckgym import PyGame
    
    deck1_first = 0
    deck2_first = 0
    
    for i in range(1000):
        # Create a single game
        game = PyGame(deck1, deck2, None, None) # seed=None -> random
        state = game.get_state()
        
        # current_player is the player index who acts next.
        # Initially, current_player is the STARTING player.
        # Player 0 is constructed with deck1.
        # Player 1 is constructed with deck2.
        
        if state.current_player == 0:
            deck1_first += 1
        else:
            deck2_first += 1
            
    print(f"Total Games: 1000")
    print(f"Deck 1 (Player 0) Goes First: {deck1_first}")
    print(f"Deck 2 (Player 1) Goes First: {deck2_first}")
    
    if 450 <= deck1_first <= 550:
        print("PASS: Turn order appears random (close to 50/50).")
    else:
        print("FAIL: Turn order does not appear random.")

if __name__ == "__main__":
    test_turn_order()
