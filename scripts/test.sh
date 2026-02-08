#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:.
uv run pytest tests/
