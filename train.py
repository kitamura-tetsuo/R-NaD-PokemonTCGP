import argparse
import sys
import logging
import deckgym_openspiel
import os
import pickle
from src.rnad import train_loop, RNaDConfig, LeagueConfig, find_latest_checkpoint
from src.training.experiment import ExperimentManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_run_id_from_checkpoint(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data.get('metadata', {}).get('mlflow_run_id')
    except Exception as e:
        logging.warning(f"Failed to extract run_id from checkpoint {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train R-NaD agent on deckgym_ptcgp")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for network")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save/load checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Specific checkpoint file to resume from")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=10, help="Checkpoint interval")
    parser.add_argument("--deck_id_1", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck file for player 1")
    parser.add_argument("--deck_id_2", type=str, default="deckgym-core/example_decks/mewtwoex.txt", help="Path to deck file for player 2")
    parser.add_argument("--win_reward", type=float, default=1.0, help="Reward for winning")
    parser.add_argument("--point_reward", type=float, default=0.0, help="Reward per point gained")
    parser.add_argument("--damage_reward", type=float, default=0.0, help="Reward per damage dealt")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable JAX profiler")
    parser.add_argument("--profiler_dir", type=str, default="runs/profile", help="Directory to save profiler trace")
    parser.add_argument("--profile_start_step", type=int, default=0, help="Step to start profiling")
    parser.add_argument("--profile_num_steps", type=int, default=10, help="Number of steps to profile")
    
    # League expansion arguments
    parser.add_argument("--fixed_decks", type=str, nargs="+", default=None, help="List of decks that always participate in matches")
    parser.add_argument("--league_decks_student", type=str, default=None, help="CSV file for student league decks")
    parser.add_argument("--league_decks_teacher", type=str, default=None, help="CSV file for teacher league decks")

    parser.add_argument("--past_self_play", action="store_true", help="Enable past self-play (training against past checkpoints)")
    parser.add_argument("--test_interval", type=int, default=10, help="Interval for evaluating against baseline (step 0)")
    parser.add_argument("--test_games", type=int, default=8, help="Number of games for evaluation per deck pair")

    parser.add_argument("--unroll_length", type=int, default=200, help="Fixed unroll length for trajectory generation. max is 1000")
    parser.add_argument("--num_buffers", type=int, default=2, help="Number of double-buffering buffers (N-buffering)")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["mlp", "transformer"], help="Type of model torso")
    parser.add_argument("--transformer_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--transformer_heads", type=int, default=8, help="Number of transformer heads")
    parser.add_argument("--transformer_embed_dim", type=int, default=256, help="Embedding dimension for transformer")
    parser.add_argument("--transformer_seq_len", type=int, default=32, help="Sequence length for transformer")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--update_batch_size", type=int, default=None, help="Batch size for gradient updates (None means same as batch_size)")
    parser.add_argument("--timeout_reward", type=float, default=None, help="Fixed reward for timeout/draw (default: use bootstrap value)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of trajectory generation workers")
    parser.add_argument("--disable_mlflow", action="store_true", help="Disable MLflow logging")


    args = parser.parse_args()

    league_config = None
    if args.league_decks_student or args.league_decks_teacher:
        league_config = LeagueConfig.from_csv(args.league_decks_student, args.league_decks_teacher)
        if league_config and args.fixed_decks:
            # Update fixed decks if provided via CLI as well
            league_config = league_config._replace(fixed_decks=args.fixed_decks)

    config = RNaDConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        deck_id_1=args.deck_id_1,
        deck_id_2=args.deck_id_2,
        league_config=league_config,
        win_reward=args.win_reward,
        point_reward=args.point_reward,
        damage_reward=args.damage_reward,
        enable_profiler=args.enable_profiler,
        profiler_dir=args.profiler_dir,
        profile_start_step=args.profile_start_step,
        profile_num_steps=args.profile_num_steps,
        past_self_play=args.past_self_play,
        test_interval=args.test_interval,
        test_games=args.test_games,
        unroll_length=args.unroll_length,
        num_buffers=args.num_buffers,
        model_type=args.model_type,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_embed_dim=args.transformer_embed_dim,
        transformer_seq_len=args.transformer_seq_len,
        accumulation_steps=args.accumulation_steps,
        update_batch_size=args.update_batch_size,
        timeout_reward=args.timeout_reward,
        seed=args.seed,
        num_workers=args.num_workers
    )



    # Determine checkpoint to resume from (to get run_id)
    resume_checkpoint = args.resume_checkpoint
    if not resume_checkpoint:
        resume_checkpoint = find_latest_checkpoint(args.checkpoint_dir)

    run_id = None
    if resume_checkpoint:
        run_id = get_run_id_from_checkpoint(resume_checkpoint)
        if run_id:
            logging.info(f"Resuming MLflow run: {run_id}")

    # Initialize ExperimentManager
    experiment_manager = None
    if not args.disable_mlflow:
        experiment_manager = ExperimentManager(experiment_name="RNaD_Experiment", checkpoint_dir=args.checkpoint_dir, run_id=run_id)

    logging.info(f"Starting training with config: {config}")

    try:
        train_loop(
            config,
            experiment_manager=experiment_manager,
            checkpoint_dir=args.checkpoint_dir,
            resume_checkpoint=resume_checkpoint
        )
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
