#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:.
uv run python train.py "$@"
