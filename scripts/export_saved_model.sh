#!/bin/bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu
uv pip install tensorflow
uv run scripts/export_saved_model.py \
    --checkpoint_dir "/content/drive/MyDrive/R-NaD-PokemonTCGP_experiments/checkpoints" \
    --output_dir  "/content/drive/MyDrive/R-NaD-PokemonTCGP_experiments/saved_model" \
    "$@"
