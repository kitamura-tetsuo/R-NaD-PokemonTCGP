#!/bin/bash
set -e

bash ./scripts/setup.sh

export RAYON_NUM_THREADS=8
export PYTHONPATH=$PYTHONPATH:.:deckgym-core/python
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434
# uv run python train.py --max_steps 1010 --save_interval 1 --log_interval 1 --batch_size 8 --deck_id_1 "deckgym-core/example_decks/mewtwoex.txt" --deck_id_2 "deckgym-core/example_decks/blastoiseex.txt" --win_reward 1.0 --point_reward 0.1 --damage_reward 0.01 "$@"
uv run python train.py --batch_size 1 --accumulation_steps 128 --max_steps 20000 --save_interval 10 --log_interval 1 --test_interval -1 --test_games 4 --win_reward 1.0 --point_reward 0.1 --damage_reward 0.01 --timeout_reward 0.0 --league_decks_student "train_data/teacher.csv" --league_decks_teacher "train_data/teacher.csv" --enable_profiler --transformer_layers 3 --transformer_heads 2 --transformer_embed_dim 64 --transformer_seq_len 8 "$@" 
# uv run python train.py --batch_size 1 --accumulation_steps 128 --max_steps 20000 --save_interval 10 --log_interval 1 --test_interval -1 --test_games 4 --win_reward 1.0 --point_reward 0.1 --damage_reward 0.01 --timeout_reward 0.0 --league_decks_student "train_data/teacher.csv" --league_decks_teacher "train_data/teacher.csv" "$@" 

