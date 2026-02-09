import deckgym.deckgym as deckgym
import numpy as np

def test_deckgym_bindings():
    print(f"DeckGym file: {deckgym.__file__}")
    print("Testing DeckGym bindings...")
    
    deck_path_1 = "deckgym-core/example_decks/mewtwoex.txt"
    deck_path_2 = "deckgym-core/example_decks/mewtwoex.txt" # Using same deck is fine if we provide explicit IDs in reset, OR if we don't rely on cache size check.
    # But wait, the issue is that cache size < 2 triggers error in reset() if NO IDs provided.
    # So we must provide IDs or have >=2 distinct paths in cache.
    
    # Let's use two differents paths if possible, or just mock it by copying a file?
    # Or just pass IDs to reset as intended in usage.
    
    # Better: just use two different example decks
    deck_path_1 = "deckgym-core/example_decks/mewtwoex.txt"
    deck_path_2 = "deckgym-core/example_decks/blastoiseex.txt"

    batch_size = 2
    
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
    logits = np.random.randn(batch_size, 100).astype(np.float32).tolist()
    
    step_res = sim.sample_and_step(logits)
    print(f"Step returned tuple length: {len(step_res)}")
    
    # Expecting 8 elements now? 
    # (next_obs, rewards, dones, timed_out, valid_mask, actions, log_probs, next_current_players)
    
    if len(step_res) == 8:
        print("Step returned 8 elements. SUCCESS.")
        next_obs, rewards, dones, timed_out, valid_mask, actions, log_probs, next_cur_players = step_res
        print(f"Next Current players: {next_cur_players}")
        assert len(next_cur_players) == batch_size
    else:
        print(f"Step returned {len(step_res)} elements. FAILED.")

if __name__ == "__main__":
    test_deckgym_bindings()
