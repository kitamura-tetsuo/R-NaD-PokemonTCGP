import json
import numpy as np
import os

def prepare_card_embeddings(database_path, output_path):
    with open(database_path, 'r', encoding='utf-8') as f:
        db = json.load(f)

    # Define feature dimensions
    # Numerical: HP (1), Stage (1), Retreat Cost (1), NumAttacks (1), MaxDamage (1)
    # Categorical (One-Hot): Type (10), Weakness (11 - includes None)
    # Total Dim = 5 + 10 + 11 = 26 (Simplified version)

    energy_types = [
        "Grass", "Fire", "Water", "Lightning", "Psychic", 
        "Fighting", "Darkness", "Metal", "Dragon", "Colorless"
    ]
    type_to_idx = {t: i for i, t in enumerate(energy_types)}

    num_cards = len(db)
    feature_dim = 26
    embeddings = np.zeros((num_cards, feature_dim), dtype=np.float32)

    for i, item in enumerate(db):
        # Determine if it's a Pokemon, Trainer, or Energy (though DB mostly has Pokemon and Trainers)
        if "Pokemon" in item:
            card = item["Pokemon"]
            # Numerical
            embeddings[i, 0] = card.get("hp", 0) / 300.0  # Normalize HP
            embeddings[i, 1] = card.get("stage", 0) / 2.0  # Stage 0, 1, 2
            embeddings[i, 2] = len(card.get("retreat_cost", [])) / 4.0
            
            attacks = card.get("attacks", [])
            embeddings[i, 3] = len(attacks) / 3.0
            
            max_dmg = 0
            for atk in attacks:
                max_dmg = max(max_dmg, atk.get("fixed_damage", 0))
            embeddings[i, 4] = max_dmg / 200.0

            # Type One-Hot (offset 5)
            card_type = card.get("energy_type")
            if card_type in type_to_idx:
                embeddings[i, 5 + type_to_idx[card_type]] = 1.0

            # Weakness One-Hot (offset 15)
            weakness = card.get("weakness")
            if weakness in type_to_idx:
                embeddings[i, 15 + type_to_idx[weakness]] = 1.0
            else:
                embeddings[i, 15 + 10] = 1.0 # None/Other

        elif "Trainer" in item:
            card = item["Trainer"]
            # Trainers have different stats, but for now we'll just flag them
            # Maybe add a "is_trainer" flag if we had more dims
            embeddings[i, 1] = -1.0 # Unique flag for trainers in "stage" slot for now
            # We could add more features for trainers later (e.g., card type: Supporter, Item, Tool)
            pass
        
        # Note: In a real implementation, we would also use a text encoder for attacks and effects.
        # For this first version, we focus on the structure.

    np.save(output_path, embeddings)
    print(f"Saved {num_cards} card embeddings with dimension {feature_dim} to {output_path}")

if __name__ == "__main__":
    db_path = "deckgym-core/database.json"
    out_path = "card_embeddings.npy"
    if os.path.exists(db_path):
        prepare_card_embeddings(db_path, out_path)
    else:
        print(f"Error: {db_path} not found.")
