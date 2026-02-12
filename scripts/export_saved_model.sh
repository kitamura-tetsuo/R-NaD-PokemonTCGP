#!/bin/bash
export PYTHONPATH=$(pwd)
uv pip install tensorflow
uv run scripts/export_saved_model.py \
    --checkpoint_dir "checkpoints" \
    "$@"
