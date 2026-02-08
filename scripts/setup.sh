#!/bin/bash
set -e

# Cleanup on exit
trap "rm -f requirements_temp.txt" EXIT

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    pip install uv
fi

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install build dependencies
echo "Installing build dependencies..."
uv pip install maturin setuptools pytest

# Install dependencies from requirements.txt excluding deckgym-core
grep -v "deckgym-core" requirements.txt > requirements_temp.txt
echo "Installing requirements..."
uv pip install -r requirements_temp.txt

# Try to install deckgym-core if it exists and is not empty
if [ -d "deckgym-core" ] && [ "$(ls -A deckgym-core)" ]; then
    echo "Installing deckgym-core..."
    uv pip install ./deckgym-core
else
    echo "deckgym-core directory is empty or missing. Skipping installation."
fi

echo "Setup complete. Activate environment with: source .venv/bin/activate"
