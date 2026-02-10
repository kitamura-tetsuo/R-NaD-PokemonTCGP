#!/bin/bash
export PYTHONPATH=$(pwd)
uv run src/battle.py \
    --checkpoint "checkpoints_sample/checkpoint_30.pkl" \
    --deck_id_1 "deckgym-core/example_decks/mewtwoex.txt" \
    --deck_id_2 "deckgym-core/example_decks/blastoiseex.txt" \
    "$@"
