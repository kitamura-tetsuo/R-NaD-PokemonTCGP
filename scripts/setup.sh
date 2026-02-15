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
    echo "Creating virtual environment .venv..."
    uv venv .venv --python 3.11
fi

# Activate virtual environment
source .venv/bin/activate

# Install build dependencies
echo "Installing build dependencies..."
uv pip install maturin setuptools pytest

# Install dependencies from requirements.txt
echo "Installing requirements..."
uv pip install -r requirements.txt

# --- Fix for libnvrtc.so missing ---
# nvidia-cuda-nvrtc-cu11 installs libnvrtc.so.11.2 but some libraries look for libnvrtc.so
NVRTC_LIB_DIR=$(python -c "import nvidia.cuda_nvrtc; import os; print(os.path.join(nvidia.cuda_nvrtc.__path__[0], 'lib'))")
if [ -d "$NVRTC_LIB_DIR" ]; then
    if [ ! -f "$NVRTC_LIB_DIR/libnvrtc.so" ]; then
        echo "Creating symlink for libnvrtc.so..."
        ln -sf "$NVRTC_LIB_DIR/libnvrtc.so.11.2" "$NVRTC_LIB_DIR/libnvrtc.so"
    fi
fi

# --- Fix for libcufft.so missing ---
# nvidia-cufft-cu11 installs libcufft.so.10 but some libraries look for libcufft.so
CUFFT_LIB_DIR=$(python -c "import nvidia.cufft; import os; print(os.path.join(nvidia.cufft.__path__[0], 'lib'))")
if [ -d "$CUFFT_LIB_DIR" ]; then
    if [ ! -f "$CUFFT_LIB_DIR/libcufft.so" ]; then
        echo "Creating symlink for libcufft.so..."
        ln -sf "$CUFFT_LIB_DIR/libcufft.so.10" "$CUFFT_LIB_DIR/libcufft.so"
    fi
fi

# --- Fix for libcusolver.so missing ---
# nvidia-cusolver-cu11 installs libcusolver.so.11 but some libraries look for libcusolver.so
CUSOLVER_LIB_DIR=$(python -c "import nvidia.cusolver; import os; print(os.path.join(nvidia.cusolver.__path__[0], 'lib'))")
if [ -d "$CUSOLVER_LIB_DIR" ]; then
    if [ ! -f "$CUSOLVER_LIB_DIR/libcusolver.so" ]; then
        echo "Creating symlink for libcusolver.so..."
        ln -sf "$CUSOLVER_LIB_DIR/libcusolver.so.11" "$CUSOLVER_LIB_DIR/libcusolver.so"
    fi
fi

# --- Fix for libcupti.so missing ---
# nvidia-cuda-cupti-cu11 installs libcupti.so.11.8 but some libraries look for libcupti.so
CUPTI_LIB_DIR=$(python -c "import nvidia.cuda_cupti; import os; print(os.path.join(nvidia.cuda_cupti.__path__[0], 'lib'))")
if [ -d "$CUPTI_LIB_DIR" ]; then
    if [ ! -f "$CUPTI_LIB_DIR/libcupti.so" ]; then
        echo "Creating symlink for libcupti.so..."
        ln -sf "$CUPTI_LIB_DIR/libcupti.so.11.8" "$CUPTI_LIB_DIR/libcupti.so"
    fi
fi

# --- Fix for libnccl.so missing ---
# nvidia-nccl-cu11 installs libnccl.so.2 but some libraries look for libnccl.so
NCCL_LIB_DIR=$(python -c "import nvidia.nccl; import os; print(os.path.join(nvidia.nccl.__path__[0], 'lib'))")
if [ -d "$NCCL_LIB_DIR" ]; then
    if [ ! -f "$NCCL_LIB_DIR/libnccl.so" ]; then
        echo "Creating symlink for libnccl.so..."
        ln -sf "$NCCL_LIB_DIR/libnccl.so.2" "$NCCL_LIB_DIR/libnccl.so"
    fi
fi

# --- Fix for libcusparse.so missing ---
# nvidia-cusparse-cu11 installs libcusparse.so.11 but some libraries look for libcusparse.so
CUSPARSE_LIB_DIR=$(python -c "import nvidia.cusparse; import os; print(os.path.join(nvidia.cusparse.__path__[0], 'lib'))")
if [ -d "$CUSPARSE_LIB_DIR" ]; then
    if [ ! -f "$CUSPARSE_LIB_DIR/libcusparse.so" ]; then
        echo "Creating symlink for libcusparse.so..."
        ln -sf "$CUSPARSE_LIB_DIR/libcusparse.so.11" "$CUSPARSE_LIB_DIR/libcusparse.so"
    fi
fi

# Try to install deckgym-core if it exists and is not empty
if [ -d "deckgym-core" ] && [ "$(ls -A deckgym-core)" ]; then
    echo "Installing deckgym-core..."
    (cd deckgym-core && maturin develop)
    # (cd deckgym-core && maturin develop --release)
    
    # Copy the built shared object to the root for local usage
    # Maturin develop installs it to the venv, but the project seems to expect it in root
    FIND_SO=$(find deckgym-core/target/debug -maxdepth 1 -name "libdeckgym.so" -o -name "deckgym.so" | head -n 1)
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

# --- Fix for libcublas.so missing ---
# nvidia-cublas-cu11 installs libcublas.so.11 but some libraries look for libcublas.so
CUBLAS_LIB_DIR=$(python -c "import nvidia.cublas; import os; print(os.path.join(nvidia.cublas.__path__[0], 'lib'))")
if [ -d "$CUBLAS_LIB_DIR" ]; then
    if [ ! -f "$CUBLAS_LIB_DIR/libcublas.so" ]; then
        echo "Creating symlink for libcublas.so..."
        ln -sf "$CUBLAS_LIB_DIR/libcublas.so.11" "$CUBLAS_LIB_DIR/libcublas.so"
    fi
fi

# --- Fix for libcudart.so missing ---
# nvidia-cuda-runtime-cu11 installs libcudart.so.11.0 but some libraries look for libcudart.so
CUDART_LIB_DIR=$(python -c "import nvidia.cuda_runtime; import os; print(os.path.join(nvidia.cuda_runtime.__path__[0], 'lib'))")
if [ -d "$CUDART_LIB_DIR" ]; then
    if [ ! -f "$CUDART_LIB_DIR/libcudart.so" ]; then
        echo "Creating symlink for libcudart.so..."
        ln -sf "$CUDART_LIB_DIR/libcudart.so.11.0" "$CUDART_LIB_DIR/libcudart.so"
    fi
fi

uv run python scripts/prepare_card_embeddings.py

echo "Setup complete. Activate environment with: source .venv/bin/activate"
