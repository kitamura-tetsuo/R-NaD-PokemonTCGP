#!/bin/bash
set -e

bash ./scripts/setup.sh

export PYTHONPATH=$PYTHONPATH:.:deckgym-core/python
uv run python train.py --batch_size 1 --deck_id_1 "deckgym-core/example_decks/mewtwoex.txt" --deck_id_2 "deckgym-core/example_decks/blastoiseex.txt" "$@"
