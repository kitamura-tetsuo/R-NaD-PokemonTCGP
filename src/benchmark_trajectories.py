import time
import jax
import logging
from src.rnad import RNaDLearner, RNaDConfig
import numpy as np

def benchmark():
    logging.basicConfig(level=logging.INFO)
    config = RNaDConfig(batch_size=8)  # Reduced further to 8 to fit on GTX 970 (4GB)
    
    learner = RNaDLearner("deckgym_ptcgp", config)
    learner.init(jax.random.PRNGKey(42))
    
    # Warmup
    print("Warmup...")
    _ = learner.generate_trajectories(jax.random.PRNGKey(0))
    
    print("Benchmarking...")
    num_iters = 10
    start_time = time.time()
    
    total_steps = 0
    for i in range(num_iters):
        batch = learner.generate_trajectories(jax.random.PRNGKey(i))
        # Batch['obs'] shape is [max_len, batch_size, ...]
        total_steps += batch['obs'].shape[0] * batch['obs'].shape[1]
    
    end_time = time.time()
    duration = end_time - start_time
    
    sps = total_steps / duration
    print(f"Total steps: {total_steps}")
    print(f"Duration: {duration:.2f}s")
    print(f"Steps per second (SPS): {sps:.2f}")

if __name__ == "__main__":
    benchmark()
