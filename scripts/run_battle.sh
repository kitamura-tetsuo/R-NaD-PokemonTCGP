#!/bin/bash
export PYTHONPATH=$(pwd)
uv run scripts/battle.py --checkpoint /home/ubuntu/src/R-NaD-PokemonTCGP/checkpoints/checkpoint_244.pkl --deck_id_1 "deckgym-core/example_decks/mewtwoex.txt" --deck_id_2 "deckgym-core/example_decks/blastoiseex.txt" "$@"
