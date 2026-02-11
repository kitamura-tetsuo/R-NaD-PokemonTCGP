import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    import deckgym
    print("Successfully imported deckgym")
except ImportError as e:
    print(f"Error importing deckgym: {e}")
    sys.exit(1)

from src.models.encoder import AbilityVectorEncoder

def test_ability_encoding():
    all_cards = deckgym.get_all_cards()
    
    # Find a card with an ability
    ability_cards = [c for c in all_cards if c.ability is not None]
    if not ability_cards:
        print("No cards with abilities found!")
        return

    print(f"Found {len(ability_cards)} cards with abilities.")
    
    encoder = AbilityVectorEncoder()
    
    for card in ability_cards[:5]:
        print(f"\nCard: {card.name} ({card.id})")
        print(f"Ability Title: {card.ability.title}")
        print(f"Ability Effect: {card.ability.effect}")
        
        mechanic_info = card.ability_mechanic_info
        print(f"Mechanic Info: {mechanic_info}")
        
        if mechanic_info:
            vec = encoder.encode(mechanic_info)
            print(f"Encoded Vector (first 10): {vec[:10]}")
            print(f"Vector Sum: {vec.sum()}")
            assert vec.sum() > 0, "Vector should not be empty for a known mechanic"
        else:
            print("No structured mechanic info available for this ability.")

if __name__ == "__main__":
    test_ability_encoding()
