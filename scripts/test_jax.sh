#!/bin/bash
source .venv/bin/activate

# Determine library paths for LD_LIBRARY_PATH (Copied from train.sh)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cublas; print(nvidia.cublas.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_nvrtc; print(nvidia.cuda_nvrtc.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cufft; print(nvidia.cufft.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cusolver; print(nvidia.cusolver.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_cupti; print(nvidia.cuda_cupti.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")/lib

# Memory allocation settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python -X faulthandler test_jax_gpu.py
