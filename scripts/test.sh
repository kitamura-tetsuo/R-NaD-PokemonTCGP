#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
