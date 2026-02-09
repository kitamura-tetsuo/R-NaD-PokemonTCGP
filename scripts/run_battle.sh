#!/bin/bash
export PYTHONPATH=$(pwd)
uv run scripts/battle.py "$@"
