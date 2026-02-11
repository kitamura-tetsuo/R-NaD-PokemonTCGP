import numpy as np
import sys

# Try to import deckgym robustly
try:
    import deckgym.deckgym as deckgym
except ImportError:
    try:
        import deckgym
    except ImportError:
        print("Could not import deckgym. Skipping test.")
        sys.exit(0)

def test_deckgym_bindings():
    if hasattr(deckgym, '__file__'):
        print(f"DeckGym file: {deckgym.__file__}")
    print("Testing DeckGym bindings...")
    
    # Use two different example decks
    deck_path_1 = "deckgym-core/example_decks/mewtwoex.txt"
    deck_path_2 = "deckgym-core/example_decks/blastoiseex.txt"

    batch_size = 2
    
    # Check if deck files exist, otherwise skip (CI might not have them checked out fully if submodules fail)
    import os
    if not os.path.exists(deck_path_1) or not os.path.exists(deck_path_2):
        print("Deck files not found. Skipping test.")
        return

    sim = deckgym.PyBatchedSimulator(
        deck_path_1,
        deck_path_2,
        batch_size,
        1.0, 0.0, 0.0
    )
    
    print("Sim initialized.")
    
    # Test reset
    res = sim.reset(seed=42)
    print(f"Reset returned type: {type(res)}")
    
    if isinstance(res, tuple):
        obs, current_players = res
        print(f"Obs shape: {len(obs)} x {len(obs[0])}")
        print(f"Current players: {current_players}")
        assert len(current_players) == batch_size
    else:
        print("Reset did NOT return a tuple! FAILED.")
        return

    # Test step
    # Pass numpy array, not list
    logits = np.random.randn(batch_size, 100).astype(np.float32)
    
    step_res = sim.sample_and_step(logits)
    print(f"Step returned tuple length: {len(step_res)}")
    
    if len(step_res) == 8:
        print("Step returned 8 elements. SUCCESS.")
        next_obs, rewards, dones, timed_out, valid_mask, actions, log_probs, next_cur_players = step_res
        print(f"Next Current players: {next_cur_players}")
        assert len(next_cur_players) == batch_size
    else:
        print(f"Step returned {len(step_res)} elements. FAILED.")

if __name__ == "__main__":
    test_deckgym_bindings()
