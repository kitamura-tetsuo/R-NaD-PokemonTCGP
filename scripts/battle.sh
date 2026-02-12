#!/bin/bash
export PYTHONPATH=$(pwd)
uv run src/battle.py \
    --checkpoint "checkpoints/checkpoint_1290.pkl" \
    --deck_id_1 "train_data/8acd216f.txt" \
    --deck_id_2 "train_data/ab2bf611.txt" \
    --device "cpu"
    "$@"

-#     --checkpoint "saved_model/74e1ccf674b2b3504a0112d84b4601bb741968be/199" \