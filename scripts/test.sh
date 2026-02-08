#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:.:deckgym-core/python
uv run pytest tests/
