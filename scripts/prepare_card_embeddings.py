import json
import numpy as np
import os
import sys

# Add current directory to path so we can import deckgym and our encoders
sys.path.insert(0, os.getcwd())

try:
    import deckgym
    print("Successfully imported deckgym")
except ImportError as e:
    print(f"Error: Could not import deckgym. Make sure deckgym.so is in the current directory. {e}")
    sys.exit(1)

from src.models.encoder import LogicVectorEncoder, TrainerVectorEncoder

def prepare_card_embeddings(output_path):
    # Using deckgym.get_all_cards() instead of reading JSON manually
    # to ensure consistency with the Rust core.
    all_cards = deckgym.get_all_cards()
    num_cards = len(all_cards)

    logic_encoder = LogicVectorEncoder()
    trainer_encoder = TrainerVectorEncoder()
    
    # Base Numerical: HP (1), Stage (1), Retreat Cost (1), NumAttacks (1), MaxDamage (1) = 5
    # Categorical (One-Hot): Type (10), Weakness (11 - includes None) = 21
    # Base Dim = 26
    base_dim = 26
    attack_dim = logic_encoder.vector_size
    trainer_dim = trainer_encoder.vector_size
    feature_dim = base_dim + (2 * attack_dim) + trainer_dim
    
    embeddings = np.zeros((num_cards, feature_dim), dtype=np.float32)

    energy_types = [
        "Grass", "Fire", "Water", "Lightning", "Psychic", 
        "Fighting", "Darkness", "Metal", "Dragon", "Colorless"
    ]
    type_to_idx = {t: i for i, t in enumerate(energy_types)}

    print(f"Generating embeddings for {num_cards} cards. Feature dimension: {feature_dim}")

    for i, card in enumerate(all_cards):
        # --- Base Features (Compatible with previous version) ---
        if card.is_pokemon:
            # Numerical
            embeddings[i, 0] = card.hp / 300.0  # Normalize HP
            embeddings[i, 1] = float(getattr(card, 'stage', 0)) / 2.0  # Stage 0, 1, 2
            embeddings[i, 2] = float(card.retreat_cost) / 4.0
            
            attacks = card.attacks
            embeddings[i, 3] = len(attacks) / 3.0
            
            max_dmg = 0
            for atk in attacks:
                max_dmg = max(max_dmg, atk.fixed_damage)
            embeddings[i, 4] = max_dmg / 200.0

            # Type One-Hot (offset 5)
            # card.energy_type might be a PyEnergyType object
            e_type = str(card.energy_type) if card.energy_type else None
            if e_type in type_to_idx:
                embeddings[i, 5 + type_to_idx[e_type]] = 1.0

            # Weakness One-Hot (offset 15)
            weakness = str(card.weakness) if card.weakness else None
            if weakness in type_to_idx:
                embeddings[i, 15 + type_to_idx[weakness]] = 1.0
            else:
                embeddings[i, 15 + 10] = 1.0 # None/Other

            # --- New Logic Features (Pokemon) ---
            # Attack 1
            if len(attacks) >= 1:
                atk1_vec = logic_encoder.encode(attacks[0].mechanic_info)
                embeddings[i, base_dim : base_dim + attack_dim] = atk1_vec
            
            # Attack 2
            if len(attacks) >= 2:
                atk2_vec = logic_encoder.encode(attacks[1].mechanic_info)
                embeddings[i, base_dim + attack_dim : base_dim + 2 * attack_dim] = atk2_vec

        elif card.is_trainer:
            # Numerical flag for trainers
            embeddings[i, 1] = -1.0 
            
            # --- New Logic Features (Trainer) ---
            t_vec = trainer_encoder.encode(card.trainer_mechanic_info)
            embeddings[i, base_dim + 2 * attack_dim : base_dim + 2 * attack_dim + trainer_dim] = t_vec

    np.save(output_path, embeddings)
    print(f"Saved {num_cards} card embeddings with dimension {feature_dim} to {output_path}")

if __name__ == "__main__":
    out_path = "card_embeddings.npy"
    prepare_card_embeddings(out_path)
