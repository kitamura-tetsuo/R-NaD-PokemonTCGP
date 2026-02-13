import optuna
import jax
import time
import logging
import numpy as np
from src.rnad import RNaDConfig, RNaDLearner, TrajectoryGenerator, slice_batch

# Configure logging to show info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TARGET_GLOBAL_BATCH_SIZE = 128
WARMUP_STEPS = 50
MEASURE_STEPS = 50

def objective(trial):
    # 1. Suggest parameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    num_workers = trial.suggest_int("num_workers", 1, 16)

    # Calculate accumulation steps to maintain global batch size
    # Note: If batch_size > TARGET_GLOBAL_BATCH_SIZE, accumulation_steps becomes 1
    accumulation_steps = max(1, TARGET_GLOBAL_BATCH_SIZE // batch_size)

    # 2. Configure RNaD
    # Fixed heavy model settings and constraints
    config = RNaDConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        accumulation_steps=accumulation_steps,
        update_batch_size=32, # Fixed to prevent OOM
        unroll_length=200,    # Fixed default
        model_type="transformer",
        transformer_layers=4,
        transformer_embed_dim=128,
        transformer_seq_len=32,
        # Disable heavy logging/saving/eval for speed tuning
        test_interval=0,
        save_interval=100000,
        log_interval=100000,
        enable_profiler=False
    )

    learner = None
    generator = None

    try:
        # 3. Initialize Learner
        learner = RNaDLearner("deckgym_ptcgp", config)
        learner.init(jax.random.PRNGKey(0))

        # Check for invalid configuration (batch_size < update_batch_size)
        # This results in 0 micro-batches and no updates.
        if learner.num_micro_batches == 0:
            logging.warning(f"batch_size {batch_size} is smaller than update_batch_size {learner.sgd_batch_size}. Skipping.")
            return 0.0

        # 4. Initialize Generator
        generator = TrajectoryGenerator(learner, num_workers=num_workers)
        generator.start()

        # 5. Training Loop
        # We need to mimic the train_loop logic for updates

        # Warmup
        logging.info(f"Trial {trial.number}: Starting warmup ({WARMUP_STEPS} steps)...")
        for step in range(WARMUP_STEPS):
            for _ in range(accumulation_steps):
                large_batch = generator.get_batch()
                for i_micro in range(learner.num_micro_batches):
                    start = i_micro * learner.sgd_batch_size
                    end = start + learner.sgd_batch_size
                    micro_batch = slice_batch(large_batch, start, end)
                    learner.update(micro_batch, step)

        # Measurement
        logging.info(f"Trial {trial.number}: Starting measurement ({MEASURE_STEPS} steps)...")
        start_time = time.time()
        total_samples = 0

        for step in range(WARMUP_STEPS, WARMUP_STEPS + MEASURE_STEPS):
            metrics_accum = {}
            for _ in range(accumulation_steps):
                large_batch = generator.get_batch()

                # We can calculate samples from the large batch stats directly if available
                # But let's follow the standard update loop to be safe and consistent with training load

                # Note: large_batch['stats']['mean_episode_length'] is the mean length of episodes in this batch.
                # Total environment steps = mean_length * batch_size
                # We accumulate this for SPS calculation.

                # Use stats from the batch if available, otherwise fallback to 0 (should not happen if successful)
                batch_mean_len = large_batch['stats']['mean_episode_length']
                total_samples += batch_mean_len * batch_size

                for i_micro in range(learner.num_micro_batches):
                    start = i_micro * learner.sgd_batch_size
                    end = start + learner.sgd_batch_size
                    micro_batch = slice_batch(large_batch, start, end)
                    learner.update(micro_batch, step)

            # Optional: Report intermediate SPS for pruning?
            # For short measurement (50 steps), maybe not strictly necessary, but good practice.
            # current_sps = total_samples / (time.time() - start_time + 1e-6)
            # trial.report(current_sps, step)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()

        end_time = time.time()

        duration = end_time - start_time
        sps = total_samples / (duration + 1e-6)

        logging.info(f"Trial {trial.number}: Finished. SPS={sps:.2f}")
        return sps

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {e}")
        return 0.0
    finally:
        if generator:
            generator.stop()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("\n" + "="*40)
    print("Optimization Finished")
    print("="*40)
    print(f"Best SPS: {study.best_value:.2f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
