#!/bin/bash
export PYTHONPATH=$(pwd)
uv run src/battle.py \
    --checkpoint "checkpoints/checkpoint_40.pkl" \
    --deck_id_1 "train_data/8acd216f.txt" \
    --deck_id_2 "train_data/ab2bf611.txt" \
    --device "cpu"
    "$@"
