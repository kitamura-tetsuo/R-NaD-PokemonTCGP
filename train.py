import argparse
import sys
import logging
from src.rnad import train_loop, RNaDConfig
from src.training.experiment import ExperimentManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Train R-NaD agent on deckgym_ptcgp")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for network")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save/load checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Specific checkpoint file to resume from")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint interval")

    args = parser.parse_args()

    config = RNaDConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )

    # Initialize ExperimentManager
    experiment_manager = ExperimentManager(experiment_name="RNaD_Experiment")

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
