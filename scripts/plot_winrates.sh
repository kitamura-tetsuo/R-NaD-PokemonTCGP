#!/bin/bash
uv run src/plot_winrates.py \
    --checkpoint_dir "./checkpoints" \
    --control_checkpoint "./checkpoints/checkpoint_0.pkl" \
    --decks "./deckgym-core/example_decks/mewtwoex.txt" "./deckgym-core/example_decks/blastoiseex.txt"
