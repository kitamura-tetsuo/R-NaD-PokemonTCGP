import argparse
import sys
import logging
from src.rnad import train_loop, RNaDConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Train R-NaD agent on deckgym_ptcgp")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for network")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of residual blocks")

    args = parser.parse_args()

    config = RNaDConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks
    )

    logging.info(f"Starting training with config: {config}")

    try:
        train_loop(config)
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
