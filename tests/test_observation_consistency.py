import numpy as np
import random
import pyspiel
from deckgym_openspiel.game import DeckGymGame
from deckgym_openspiel.state import DeckGymState

DECK_A = "deckgym-core/example_decks/mewtwoex.txt"
DECK_B = "deckgym-core/example_decks/mewtwoex.txt"

def get_random_state(seed=42):
    game = DeckGymGame({"deck_id_1": DECK_A, "deck_id_2": DECK_B, "seed": seed})
    state = game.new_initial_state()
    
    rng = random.Random(seed)
    # 少し長めに進めて、場面やトラッシュにカードが行くようにする
    for _ in range(50):
        if state.is_terminal():
            break
        
        player = state.current_player()
        if player == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            action = outcomes[rng.randint(0, len(outcomes)-1)][0]
        else:
            legal_actions = state.legal_actions(player)
            if not legal_actions:
                break
            action = rng.choice(legal_actions)
        
        state.apply_action(action)
    return state

def card_to_id_f32(card):
    # カードIDを数値に変換するロジック（本来はRust側と合わせる必要があるが、
    # ここでは便宜上、CardId::from_card_idの結果を模倣するか、
    # 実際の結果を単に比較するに留める。）
    # Rust側 (encoding.rs) では CardId enum の値 (usize) を f32 に直している。
    # PyCard.id -> CardId enum への変換が必要。
    # ここでは PyGameState.get_card_id() があればそれを使いたいが、
    # 存在しない場合は、encoding.rsの結果を信じて、
    # 「複数の異なるカードが有効なIDを持っている」事だけを一旦前提とする。
    pass

def check_pokemon_slot(obs_part, rust_pokemon, p_idx, debug_prefix=""):
    # Index within slot (size 24):
    # 0: HP
    # 1-10: Energy
    # 11: CardID
    # 12-16: Status (poison, asleep, paralyzed, confused, burned)
    # 17: PlayedThisTurn
    # 18: AbilityUsed
    # 19: ToolID
    # 20-23: Key Effects
    
    if rust_pokemon is None:
        assert obs_part[0] == 0.0, f"{debug_prefix}: HP should be 0 for empty slot"
        assert np.all(obs_part[1:11] == 0.0), f"{debug_prefix}: Energy should be 0"
        assert obs_part[11] == -1.0, f"{debug_prefix}: CardID should be -1"
        return

    if not np.isclose(obs_part[0], rust_pokemon.remaining_hp / 300.0, atol=1e-6):
        print(f"{debug_prefix}: HP mismatch! obs={obs_part[0]}, actual={rust_pokemon.remaining_hp / 300.0} (raw hp={rust_pokemon.remaining_hp})")
        assert False, f"{debug_prefix}: HP mismatch"
    
    # Energy check
    # Note: encoding.rs uses energy_type_to_index which maps Grass=0, Fire=1, ..., Colorless=9
    actual_energies = np.zeros(10)
    energy_map = {"Grass":0, "Fire":1, "Water":2, "Lightning":3, "Psychic":4, 
                  "Fighting":5, "Darkness":6, "Metal":7, "Dragon":8, "Colorless":9}
    for e in rust_pokemon.attached_energy:
        actual_energies[energy_map[e.name]] += 1.0
    assert np.array_equal(obs_part[1:11], actual_energies), f"{debug_prefix}: Energy mismatch"

    # Card ID - We assume numeric comparison is subtle, but it should be consistent
    assert obs_part[11] >= 0, f"{debug_prefix}: CardID should be >= 0"

    # Status
    assert obs_part[12] == (1.0 if rust_pokemon.poisoned else 0.0)
    assert obs_part[13] == (1.0 if rust_pokemon.asleep else 0.0)
    assert obs_part[14] == (1.0 if rust_pokemon.paralyzed else 0.0)
    
    # PlayedThisTurn / AbilityUsed
    assert obs_part[17] == (1.0 if rust_pokemon.played_this_turn else 0.0)
    assert obs_part[18] == (1.0 if rust_pokemon.ability_used else 0.0)

def test_full_observation_consistency():
    for seed in range(100, 110):
        print(f"Testing full consistency with seed {seed}")
        state = get_random_state(seed=seed)
        p = state.current_player()
        if p < 0: continue
        
        obs = np.array(state.observation_tensor(p))
        rust_state = state.rust_game.get_state()
        
        # --- Section 1: Turn Info (0-6) ---
        assert obs[0] == rust_state.turn_count
        assert obs[1] == rust_state.points[p]
        assert obs[2] == rust_state.points[1-p]
        assert obs[3] == (1.0 if rust_state.current_player == p else 0.0)
        assert obs[4] == (1.0 if rust_state.has_played_support else 0.0)
        assert obs[5] == (1.0 if rust_state.has_retreated else 0.0)

        # --- Section 1.2: Current Energy (7-16) ---
        energy_map = {"Grass":0, "Fire":1, "Water":2, "Lightning":3, "Psychic":4, 
                      "Fighting":5, "Darkness":6, "Metal":7, "Dragon":8, "Colorless":9}
        curr_energy = rust_state.current_energy
        energy_vec = np.zeros(10)
        if curr_energy:
            energy_vec[energy_map[curr_energy.name]] = 1.0
        assert np.array_equal(obs[7:17], energy_vec), "Current energy mismatch"

        # --- Section 2: Hand Counts (37-38) ---
        my_hand_size = rust_state.get_hand_size(p)
        opp_hand_size = rust_state.get_hand_size(1-p)
        assert obs[37] == my_hand_size
        assert obs[38] == opp_hand_size

        # --- Section 3: Pokemon Slots (39-230) ---
        # My Active/Bench
        for i in range(4):
            check_pokemon_slot(obs[39 + i*24 : 39 + (i+1)*24], 
                              rust_state.get_pokemon_at_position(p, i), 
                              p, f"My Slot {i}")
        
        # Opp Active/Bench
        for i in range(4):
            check_pokemon_slot(obs[39 + (4+i)*24 : 39 + (5+i)*24], 
                              rust_state.get_pokemon_at_position(1-p, i), 
                              1-p, f"Opp Slot {i}")

        # --- Section 4: Hand (231-240) ---
        hand_obs = obs[231:241]
        active_hand_cards = np.sum(hand_obs != -1.0)
        assert active_hand_cards == my_hand_size, "Hand card count mismatch"

        # --- Section 5: Deck (241-260 in OLD, 251-271 in NEW) ---
        # NOTE: encoding.rs layout:
        # ...
        # Hand (Self): 10 slots (231-241)
        # Hand (Opponent Known): 10 slots (241-251)  <-- MISSED IN OLD TEST
        # Deck (Self): 20 slots (251-271)
        # Opponent Deck Count: 1 slot (271)
        # Opponent Deck Known: 20 slots (272-292)
        # Discard (Self): 10 slots (292-302)
        # Discard (Opponent): 10 slots (302-312)

        deck_obs = obs[251:271]
        actual_deck_size = rust_state.get_deck_size(p)
        # encoding.rsではソートして20枚まで。
        valid_deck_slots = np.sum(deck_obs != -1.0)
        assert valid_deck_slots == min(20, actual_deck_size), f"Deck size in obs mismatch: {valid_deck_slots} != min(20, {actual_deck_size})"
        
        # 昇順ソートされている事を確認
        sorted_deck_obs = deck_obs[deck_obs != -1.0]
        assert np.all(np.diff(sorted_deck_obs) >= 0), "Deck slots should be sorted"

        # --- Section 6: Opponent Deck Count (271) ---
        assert obs[271] == rust_state.get_deck_size(1-p)

        # --- Section 7: Discard Piles (292-312) ---
        # Skip Opponent Deck Known (272-292)

        my_discard_size = rust_state.get_discard_pile_size(p)
        opp_discard_size = rust_state.get_discard_pile_size(1-p)
        
        my_discard_obs = obs[292:302]
        opp_discard_obs = obs[302:312]
        
        assert np.sum(my_discard_obs != -1.0) == min(10, my_discard_size)
        assert np.sum(opp_discard_obs != -1.0) == min(10, opp_discard_size)

    print("All full consistency checks passed!")

if __name__ == "__main__":
    try:
        test_full_observation_consistency()
    except AssertionError as e:
        print(f"Full consistency check FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
