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


# echo "Starting miner"
# python src/miner.py \
#     --checkpoint "checkpoints/" \
#     --league_decks_student "train_data/teacher.csv" \
#     --league_decks_teacher "train_data/teacher.csv" \
#     --device "gpu" \
#     --min_turn 10 \
#     --find_depth 4 \
#     --mine_depth 20 \
#     --disable_retreat_depth 3 \
#     --disable_energy_attach_threshold 4 \
#     --diagnostic_games_per_checkpoint 100 \
#     --max_visualizations 0 \
#     "$@"

# echo "Starting tree viz"
# python src/tree_viz.py \
#     --mined_source mined_data.jsonl \
#     --mined_index -1 \
#     --max_depth 10 \
#     --device "cpu" \
#     --output analysis.sqlite

# streamlit run src/tree_app.py -- --dir "data/mined/checkpoint_50.pkl"

# 1. Capture the latest checkpoint (before distill)
PREV_MODEL=$(ls -t checkpoints/checkpoint_*.pkl 2>/dev/null | head -n 1)
if [ -z "$PREV_MODEL" ]; then
    echo "No existing checkpoint found."
else
    echo "Previous model: $PREV_MODEL"
fi

echo "Starting distillation"
python src/distill.py \
    --checkpoint_dir "checkpoints" \
    --data_file "mined_data.jsonl" \
    --device "gpu" \
    --transformer_embed_dim 256 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --accumulation_steps 8 \
    --update_batch_size 64 \
    "$@"

# 1e-3
# ----------------------------------------
# Model 1 Wins: 83 (51.9%)
# Model 2 Wins: 72 (45.0%)
# Draws:       5 (3.1%)
# ----------------------------------------
# M1 Win Rate 95% CI: [44.1%, 59.6%]
# Z-Score (H0: p=0.5): 0.87
# Result: NO statistically significant difference

# 1e-4
# ----------------------------------------
# Model 1 Wins: 78 (48.8%)
# Model 2 Wins: 76 (47.5%)
# Draws:       6 (3.8%)
# ----------------------------------------
# M1 Win Rate 95% CI: [41.0%, 56.5%]
# Z-Score (H0: p=0.5): 0.16
# Result: NO statistically significant difference
# 1e-5
# ----------------------------------------
# Model 1 Wins: 76 (47.5%)
# Model 2 Wins: 80 (50.0%)
# Draws:       4 (2.5%)
# ----------------------------------------
# M1 Win Rate 95% CI: [39.8%, 55.2%]
# Z-Score (H0: p=0.5): -0.32
# Result: NO statistically significant difference

# 1e-6
# ----------------------------------------
# Model 1 Wins: 77 (48.1%)
# Model 2 Wins: 78 (48.8%)
# Draws:       5 (3.1%)
# ----------------------------------------
# M1 Win Rate 95% CI: [40.4%, 55.9%]
# Z-Score (H0: p=0.5): -0.08
# Result: NO statistically significant difference

# 2. Capture the latest checkpoint (after distill)
NEW_MODEL=$(ls -t checkpoints/checkpoint_*.pkl 2>/dev/null | head -n 1)

if [ -z "$NEW_MODEL" ]; then
    echo "No new model found after distillation."
elif [ -z "$PREV_MODEL" ]; then
    echo "No previous model found to compare against."
elif [ "$PREV_MODEL" == "$NEW_MODEL" ]; then
    echo "Distillation did not produce a new checkpoint (or failed)."
else
    echo "New model: $NEW_MODEL"
    echo "Comparing models..."
    python src/compare_models.py \
      --model1 "$PREV_MODEL" \
      --model2 "$NEW_MODEL" \
      --n_games 10 \
      --league_decks train_data/validate.csv \
      --device cpu \
      --db_path matches.db
fi