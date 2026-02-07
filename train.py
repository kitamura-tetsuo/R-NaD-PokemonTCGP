import torch
import torch.optim as optim
import logging
from src.models.transformer import UnifiedModel
from src.training.rnad import RNaDLearner, TrajectoryBuffer
from src.training.actor import SelfPlayWorker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Configuration
    obs_dim = 30574
    action_dim = 25000
    hidden_dim = 256
    num_blocks = 4
    learning_rate = 1e-4
    episodes_per_iteration = 100
    batch_size = 100 # Should match episodes_per_iteration if we update on all
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Using device: {device}")

    # Initialize models
    model = UnifiedModel(obs_dim, action_dim, hidden_dim, num_blocks).to(device)
    fixed_point_model = UnifiedModel(obs_dim, action_dim, hidden_dim, num_blocks).to(device)
    # Sync weights initially
    fixed_point_model.load_state_dict(model.state_dict())

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize learner
    learner = RNaDLearner(
        model=model,
        fixed_point_model=fixed_point_model,
        optimizer=optimizer,
        device=device
    )

    # Initialize worker
    worker = SelfPlayWorker(device=device)

    iteration = 0
    try:
        while True:
            iteration += 1
            logging.info(f"Starting iteration {iteration}")

            # Generate episodes
            buffer = TrajectoryBuffer(device=device)
            logging.info(f"Generating {episodes_per_iteration} episodes...")

            for _ in range(episodes_per_iteration):
                episode_data = worker.run_episode(model)
                buffer.add_episode(episode_data)

            # Sample batch
            # We sample all generated episodes for update
            batch = buffer.sample(batch_size=episodes_per_iteration)

            # Update model
            logging.info("Updating model...")
            metrics = learner.update(batch)

            logging.info(f"Iteration {iteration} metrics: {metrics}")

            # Update fixed point
            logging.info("Updating fixed point model...")
            learner.update_fixed_point()

            # Optional: Save checkpoint
            if iteration % 10 == 0:
                torch.save(model.state_dict(), f"model_checkpoint_{iteration}.pt")
                logging.info(f"Saved checkpoint to model_checkpoint_{iteration}.pt")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")

if __name__ == "__main__":
    main()
