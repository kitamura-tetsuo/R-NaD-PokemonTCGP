#!/bin/bash
set -e

bash ./scripts/setup.sh

export RAYON_NUM_THREADS=8
export PYTHONPATH=$PYTHONPATH:.:deckgym-core/python
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434
# uv run python train.py --max_steps 1010 --save_interval 1 --log_interval 1 --batch_size 8 --deck_id_1 "deckgym-core/example_decks/mewtwoex.txt" --deck_id_2 "deckgym-core/example_decks/blastoiseex.txt" --win_reward 1.0 --point_reward 0.1 --damage_reward 0.01 "$@"
uv run python train.py --num_buffers 4 --max_steps 200 --save_interval 10 --log_interval 1 --test_interval 10 --test_games 4 --batch_size 4 --win_reward 1.0 --point_reward 0.1 --damage_reward 0.01 --league_decks "deckgym-core/example_decks/mewtwoex.txt" "deckgym-core/example_decks/blastoiseex.txt" "deckgym-core/example_decks/benchmark-electric.txt" --league_rates 1 1 1 "$@" --enable_profiler
