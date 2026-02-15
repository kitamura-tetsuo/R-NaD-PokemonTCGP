import argparse
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import json
import os
import logging
import time
from typing import Dict, Any, List, Tuple, Generator
from functools import partial

# Import from existing codebase
from src.rnad import RNaDConfig, RNaDLearner, find_latest_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str, batch_size: int) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Generator that yields batches of data from the JSONL file.
    Each batch contains 'obs' (inputs) and 'target_value' (labels).
    """
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        return

    data_entries = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'state_blob' in entry and 'oracle_value' in entry:
                    data_entries.append(entry)
            except json.JSONDecodeError:
                continue

    if not data_entries:
        logging.warning(f"No valid data found in {file_path}. Ensure miner.py has generated data with 'state_blob'.")
        return

    logging.info(f"Loaded {len(data_entries)} entries from {file_path}")
    
    # Shuffle data
    random_indices = np.random.permutation(len(data_entries))
    
    # Batching
    num_batches = len(data_entries) // batch_size
    for i in range(num_batches):
        batch_indices = random_indices[i * batch_size : (i + 1) * batch_size]
        batch_obs = []
        batch_targets = []
        
        for idx in batch_indices:
            entry = data_entries[idx]
            batch_obs.append(entry['state_blob'])
            batch_targets.append(entry['oracle_value'])
            
        yield {
            'obs': np.array(batch_obs, dtype=np.float32),
            'target_value': np.array(batch_targets, dtype=np.float32).reshape(-1, 1)
        }

def mse_loss_fn(params, batch, apply_fn):
    """
    Supervised MSE Loss for Value Network.
    """
    obs = batch['obs'] # (B, ObsDim)
    target = batch['target_value'] # (B, 1)
    
    # Forward pass
    # apply_fn returns (policy_logits, value)
    # We ignore policy_logits/mask for now as we focus on Value Distillation
    _, value_pred = apply_fn(params, jax.random.PRNGKey(0), obs)
    
    # MSE
    loss = jnp.mean((value_pred - target) ** 2)
    return loss, value_pred

@partial(jax.jit, static_argnums=(2, 3))
def train_step(params, opt_state, apply_fn, optimizer, batch):
    """
    Performs a single training step.
    """
    (loss, value_pred), grads = jax.value_and_grad(mse_loss_fn, has_aux=True)(
        params, batch, apply_fn
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss

import pickle

def distill(args):
    # 1. Setup Device
    if args.device == 'cpu':
        jax.config.update("jax_platform_name", "cpu")
    
    # Identify Checkpoint Path first
    checkpoint_path = args.checkpoint_dir
    # Support 'checkpoint' argument as alias or precise file
    if args.checkpoint:
         checkpoint_path = args.checkpoint

    if os.path.isdir(checkpoint_path):
        latest = find_latest_checkpoint(checkpoint_path)
        if latest:
            checkpoint_path = latest

    # 2. Prepare Config
    # We use a dummy config to initialize, then overwrite with checkpoint if available
    initial_config = RNaDConfig(
        deck_id_1="deckgym-core/example_decks/mewtwoex.txt", # Placeholder
        deck_id_2="deckgym-core/example_decks/mewtwoex.txt",
        batch_size=args.batch_size,
        update_batch_size=args.update_batch_size,
        accumulation_steps=args.accumulation_steps
    )

    # Peek at checkpoint to get correct model dimensions
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            logging.info(f"Peeking at checkpoint {checkpoint_path} for config...")
            with open(checkpoint_path, 'rb') as f:
                ckpt_data = pickle.load(f)
                if 'config' in ckpt_data:
                    loaded_cfg = ckpt_data['config']
                    logging.info(f"Found config in checkpoint. Embed dim: {getattr(loaded_cfg, 'transformer_embed_dim', 'N/A')}")
                    # We want to keep batch_size from args, but take model params from checkpoint
                    # Handle both RNaDConfig objects and dictionaries if implementation varies, 
                    # but typically it's NamedTuple or similar. getattr is safe.
                    
                    initial_config = initial_config._replace(
                        model_type=getattr(loaded_cfg, 'model_type', 'transformer'),
                        transformer_embed_dim=getattr(loaded_cfg, 'transformer_embed_dim', 64),
                        transformer_layers=getattr(loaded_cfg, 'transformer_layers', 2),
                        transformer_heads=getattr(loaded_cfg, 'transformer_heads', 4),
                        transformer_seq_len=getattr(loaded_cfg, 'transformer_seq_len', 16),
                        hidden_size=getattr(loaded_cfg, 'hidden_size', 256),
                        num_blocks=getattr(loaded_cfg, 'num_blocks', 4),
                        unroll_length=getattr(loaded_cfg, 'unroll_length', 200)
                    )
        except Exception as e:
            logging.warning(f"Failed to load config from checkpoint: {e}")

    # 2. Initialize Learner (to get network and structure)
    logging.info("Initializing RNaDLearner...")
    learner = RNaDLearner("deckgym_ptcgp", initial_config)
    
    # 3. Load Checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        learner.load_checkpoint(checkpoint_path)
    else:
        logging.warning("No checkpoint found. Initializing from scratch (random weights).")
        learner.init(jax.random.PRNGKey(args.seed))

    # 4. Setup Optimization
    # We use a separate optimizer for distillation, usually higher LR
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(learner.params)
    
    # 5. Training Loop
    logging.info(f"Starting distillation for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        data_loader = load_data(args.data_file, args.batch_size)
        
        for batch in data_loader:
            learner.params, opt_state, loss = train_step(
                learner.params, 
                opt_state, 
                learner.network.apply, 
                optimizer, 
                batch
            )
            epoch_loss += loss
            num_batches += 1
            
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            logging.info(f"Epoch {epoch+1}/{args.epochs} - MSE Loss: {avg_loss:.6f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % args.save_interval == 0:
                save_path = f"checkpoints/checkpoint_distilled_{int(time.time())}_ep{epoch+1}.pkl"
                # We save using learner's save method to maintain compatibility
                # Update learner's opt_state to the distillation one? 
                # Note: If we switch back to RL, the opt_state might be incompatible if structure differs.
                # Ideally we save params.
                learner.opt_state = opt_state # HACK: overwriting opt_state
                learner.save_checkpoint(save_path, epoch + 1, metadata={"type": "distilled", "mse_loss": float(avg_loss)})
        else:
            logging.warning("No data batches processed. Check data file.")
            break

    logging.info("Distillation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill Oracle values into the Value Network.")
    parser.add_argument("--data_file", type=str, default="data/mined_data.jsonl", help="Path to mined data JSONL.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing checkpoints.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file to load.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_interval", type=int, default=5, help="Save interval (epochs).")
    parser.add_argument("--update_batch_size", type=int, default=None, help="Update batch size for RNaDLearner config.")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Accumulation steps for RNaDLearner config.")
    
    args = parser.parse_args()
    distill(args)
