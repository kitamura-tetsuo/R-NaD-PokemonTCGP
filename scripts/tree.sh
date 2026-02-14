#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)

# Determine library paths for LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cublas; print(nvidia.cublas.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_nvrtc; print(nvidia.cuda_nvrtc.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cufft; print(nvidia.cufft.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cusolver; print(nvidia.cusolver.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cuda_cupti; print(nvidia.cuda_cupti.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cusparse; print(nvidia.cusparse.__path__[0])")/lib

python src/tree_viz.py \
    --checkpoint "checkpoints_sample/checkpoint_transformer_2.pkl" \
    --deck_id_1 "train_data/8acd216f.txt" \
    --deck_id_2 "train_data/ab2bf611.txt" \
    --device "gpu" \
    --max_depth 200 \
    "$@"

