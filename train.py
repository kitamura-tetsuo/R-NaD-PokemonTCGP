import argparse
import sys
import logging
import deckgym_openspiel
from src.rnad import train_loop, RNaDConfig, LeagueConfig
from src.training.experiment import ExperimentManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Train R-NaD agent on deckgym_ptcgp")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
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
<<<<<<< add-jax-profiler-15018126619909108173
    parser.add_argument("--enable_profiler", action="store_true", help="Enable JAX profiler")
    parser.add_argument("--profiler_dir", type=str, default="runs/profile", help="Directory to save profiler trace")
    parser.add_argument("--profile_start_step", type=int, default=10, help="Step to start profiling")
    parser.add_argument("--profile_num_steps", type=int, default=10, help="Number of steps to profile")
=======
    
    # League expansion arguments
    parser.add_argument("--league_decks", type=str, nargs="+", default=None, help="List of deck files for the league")
    parser.add_argument("--league_rates", type=float, nargs="+", default=None, help="Relative participation rates for the league decks")
    parser.add_argument("--fixed_decks", type=str, nargs="+", default=None, help="List of decks that always participate in matches")

    parser.add_argument("--past_self_play", action="store_true", help="Enable past self-play (training against past checkpoints)")
    parser.add_argument("--test_interval", type=int, default=10, help="Interval for evaluating against baseline (step 0)")
    parser.add_argument("--test_games", type=int, default=8, help="Number of games for evaluation per deck pair")
>>>>>>> main

    args = parser.parse_args()

    league_config = None
    if args.league_decks:
        rates = args.league_rates
        if rates is None:
            rates = [1.0] * len(args.league_decks)
        elif len(rates) != len(args.league_decks):
            raise ValueError("league_rates must have moving length as league_decks")
        
        league_config = LeagueConfig(
            decks=args.league_decks,
            rates=rates,
            fixed_decks=args.fixed_decks or []
        )

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
<<<<<<< add-jax-profiler-15018126619909108173
        enable_profiler=args.enable_profiler,
        profiler_dir=args.profiler_dir,
        profile_start_step=args.profile_start_step,
        profile_num_steps=args.profile_num_steps
=======
        past_self_play=args.past_self_play,
        test_interval=args.test_interval,
        test_games=args.test_games
>>>>>>> main
    )

    # Initialize ExperimentManager
    experiment_manager = ExperimentManager(experiment_name="RNaD_Experiment", checkpoint_dir=args.checkpoint_dir)

    logging.info(f"Starting training with config: {config}")

    try:
        train_loop(
            config,
            experiment_manager=experiment_manager,
            checkpoint_dir=args.checkpoint_dir,
            resume_checkpoint=args.resume_checkpoint
        )
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
