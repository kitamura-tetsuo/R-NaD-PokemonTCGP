#!/bin/bash
set -e

# Cleanup on exit
trap "rm -f requirements_temp.txt" EXIT

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
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
    (cd deckgym-core && maturin develop --release)
    
    # Copy the built shared object to the root for local usage
    # Maturin develop installs it to the venv, but the project seems to expect it in root
    FIND_SO=$(find deckgym-core/target/release -maxdepth 1 -name "libdeckgym.so" -o -name "deckgym.so" | head -n 1)
    if [ -f "$FIND_SO" ]; then
        echo "Updating deckgym.so in root..."
        cp "$FIND_SO" deckgym.so
    fi
    
    # Copy deckgym_openspiel for local usage
    if [ -d "deckgym-core/python/deckgym_openspiel" ]; then
        echo "Copying deckgym_openspiel to root..."
        rm -rf deckgym_openspiel
        cp -r deckgym-core/python/deckgym_openspiel .
    fi
else
    echo "deckgym-core directory is empty or missing. Skipping installation."
fi

uv run python scripts/prepare_card_embeddings.py

echo "Setup complete. Activate environment with: source .venv/bin/activate"
