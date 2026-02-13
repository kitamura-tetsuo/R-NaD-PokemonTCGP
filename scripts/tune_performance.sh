#!/bin/bash
set -e

bash ./scripts/setup.sh

export RAYON_NUM_THREADS=8
export PYTHONPATH=$PYTHONPATH:.:deckgym-core/python

uv run src/tune_performance.py  --transformer_layers 1 --transformer_heads 1 --transformer_embed_dim 1 --transformer_seq_len 1
