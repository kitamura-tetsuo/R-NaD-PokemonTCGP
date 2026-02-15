#!/bin/bash
set -e

source .venv/bin/activate

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


# Memory allocation settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Other environment variables
export RAYON_NUM_THREADS=8
export PYTHONPATH=$PYTHONPATH:.:deckgym-core/python
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434

# Check if python command is available (via venv activation)
if ! command -v python &> /dev/null; then
    echo "Python not found. Please ensure .venv is activated."
    exit 1
fi

# python train.py \
#   --batch_size 2 \
#   --accumulation_steps 128 \
#   --max_steps 20000 \
#   --save_interval 10 \
#   --log_interval 1 \
#   --test_interval -1 \
#   --test_games 4 \
#   --win_reward 1.0 \
#   --point_reward 0.1 \
#   --damage_reward 0.01 \
#   --timeout_reward 0.0 \
#   --league_decks_student "train_data/teacher.csv" \
#   --league_decks_teacher "train_data/teacher.csv" \
#   --enable_profiler \
#   --transformer_layers 6 \
#   --transformer_heads 4 \
#   --transformer_embed_dim 128 \
#   --transformer_seq_len 16 \
#   "$@"

python train.py \
  --batch_size 1 \
  --accumulation_steps 1024 \
  --update_batch_size 1 \
  --num_workers 8 \
  --max_steps 20000 \
  --save_interval 10 \
  --log_interval 1 \
  --test_interval -1 \
  --test_games 4 \
  --win_reward 1.0 \
  --point_reward 0.1 \
  --damage_reward 0.01 \
  --timeout_reward 0.0 \
  --league_decks_student "train_data/teacher.csv" \
  --league_decks_teacher "train_data/teacher.csv" \
  "$@"