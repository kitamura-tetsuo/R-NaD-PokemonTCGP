import deckgym
import numpy as np
import jax

sim = deckgym.PyBatchedSimulator(
    "deckgym-core/example_decks/mewtwoex.txt",
    "deckgym-core/example_decks/mewtwoex.txt",
    1, 1.0, 0.0, 0.0
)
res = sim.reset(seed=42, deck_ids_1=["deckgym-core/example_decks/mewtwoex.txt"], deck_ids_2=["deckgym-core/example_decks/mewtwoex.txt"])
print(f"Reset returned {len(res)} items")
for i, x in enumerate(res):
    print(f"Item {i} type: {type(x)}")
