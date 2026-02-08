import pytest
import pyspiel

def test_game_load():
    """
    Test that the deckgym_ptcgp game can be loaded successfully via pyspiel.
    This verifies that deckgym_openspiel is correctly installed and registered.
    """
    try:
        import deckgym_openspiel
    except ImportError:
        pytest.fail("deckgym_openspiel is not installed or not in the python path.")

    try:
        game = pyspiel.load_game("deckgym_ptcgp")
    except Exception as e:
        pytest.fail(f"Could not load game 'deckgym_ptcgp'. Error: {e}")

    assert game is not None
    assert game.num_distinct_actions() > 0
