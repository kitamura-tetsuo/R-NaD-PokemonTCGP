#!/bin/bash
set -e

# Cleanup on exit
trap "rm -f requirements_temp.txt" EXIT

# Install build dependencies
pip install maturin setuptools pytest

# Install dependencies from requirements.txt excluding deckgym-core
grep -v "deckgym-core" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt

# Try to install deckgym-core if it exists and is not empty
if [ -d "deckgym-core" ] && [ "$(ls -A deckgym-core)" ]; then
    echo "Installing deckgym-core..."
    pip install ./deckgym-core
else
    echo "deckgym-core directory is empty or missing. Skipping installation."
fi
