#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)

# Determine library paths for LD_LIBRARY_PATH (same as train.sh)
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

echo "Starting miner"
python src/miner.py \
    --checkpoint "checkpoints/" \
    --league_decks_student "train_data/teacher.csv" \
    --league_decks_teacher "train_data/teacher.csv" \
    --device "gpu" \
    --find_depth 7 \
    --mine_depth 20 \
    --disable_retreat_depth 3 \
    --max_visualizations 1 \
    "$@"

# echo "Starting tree viz"
# python src/tree_viz.py \
#     --mined_source mined_data.jsonl \
#     --mined_index -1 \
#     --max_depth 10 \
#     --device "cpu" \
#     --output analysis.sqlite

streamlit run src/tree_app.py -- --dir "data/mined/checkpoint_50.pkl"

# python src/distill.py \
#     --checkpoint_dir "checkpoints" \
#     --data_file "mined_data.jsonl" \
#     --device "cpu" \
#     "$@"
