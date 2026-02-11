import os
import pickle
import argparse
import logging
import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow as tf
from jax.experimental import jax2tf
import numpy as np
import re
import pyspiel

# Import project modules
from src.rnad import RNaDConfig
from src.models import DeckGymNet, CardTransformerNet, TransformerNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file. If None, finds latest in checkpoints/")
    parser.add_argument("--output_dir", type=str, default="saved_model", help="Output directory for SavedModel")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing checkpoints if --checkpoint is not set")
    return parser.parse_args()

def load_checkpoint(args):
    if args.checkpoint:
        path = args.checkpoint
        if not os.path.exists(path):
             raise FileNotFoundError(f"Checkpoint {path} not found.")
    else:
        # Find latest
        if not os.path.exists(args.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory {args.checkpoint_dir} not found.")

        checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}.")

        # Sort by step
        def get_step(f):
            m = re.search(r"checkpoint_(\d+).pkl", f)
            return int(m.group(1)) if m else -1

        latest = max(checkpoints, key=get_step)
        path = os.path.join(args.checkpoint_dir, latest)

    logging.info(f"Loading checkpoint from {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    args = parse_args()
    data = load_checkpoint(args)

    params = data['params']
    config = data.get('config', RNaDConfig())

    logging.info(f"Model type from config: {config.model_type}")

    # Load embedding matrix if needed
    emb_path = "card_embeddings.npy"
    if os.path.exists(emb_path):
        embedding_matrix = jnp.array(np.load(emb_path))
        logging.info(f"Loaded embedding matrix from {emb_path}, shape: {embedding_matrix.shape}")
    else:
        logging.warning(f"Embedding matrix not found at {emb_path}. Using zero matrix.")
        embedding_matrix = jnp.zeros((10000, 26))

    # Load game to get shapes
    logging.info("Loading game environment to determine shapes...")
    game = pyspiel.load_game(
        "deckgym_ptcgp",
        {
            "deck_id_1": config.deck_id_1,
            "deck_id_2": config.deck_id_2,
            "max_game_length": config.unroll_length
        }
    )
    num_actions = game.num_distinct_actions()
    obs_shape = game.observation_tensor_shape() # List of ints
    logging.info(f"Num actions: {num_actions}, Obs shape: {obs_shape}")

    def forward(x):
        if config.model_type == "transformer":
            # NOTE: src/rnad.py uses CardTransformerNet
            net = CardTransformerNet(
                num_actions=num_actions,
                embedding_matrix=embedding_matrix,
                hidden_size=config.transformer_embed_dim,
                num_blocks=config.transformer_layers,
                num_heads=config.transformer_heads,
            )
        else:
            net = DeckGymNet(
                num_actions=num_actions,
                hidden_size=config.hidden_size,
                num_blocks=config.num_blocks
            )
        return net(x)

    network = hk.transform(forward)

    def inference_fn(obs):
        # obs: (Batch, Dim)
        rng = jax.random.PRNGKey(0)
        return network.apply(params, rng, obs)

    # Convert to TF
    # Input signature: We assume batch dimension is dynamic (None)
    # obs_shape is likely [Dim] (1D).
    # TensorSpec shape should be (None, *obs_shape)
    input_signature = [
        tf.TensorSpec(shape=(None, *obs_shape), dtype=tf.float32, name='obs')
    ]

    logging.info("Converting JAX function to TensorFlow...")
    # We specify polymorphic_shapes to allow dynamic batch size
    tf_fn = jax2tf.convert(
        inference_fn,
        with_gradient=False,
        polymorphic_shapes=["(b, ...)"]
    )

    class TFModule(tf.Module):
        def __init__(self, jax_fn):
            super().__init__()
            self._fn = jax_fn

        @tf.function(input_signature=input_signature, jit_compile=True)
        def __call__(self, obs):
            return self._fn(obs)

        @tf.function(input_signature=input_signature, jit_compile=True)
        def predict(self, obs):
            # Return dict for better signature
            logits, value = self._fn(obs)
            return {"policy": logits, "value": value}

    module = TFModule(tf_fn)

    logging.info(f"Saving SavedModel to {args.output_dir}")
    tf.saved_model.save(module, args.output_dir)
    logging.info("Done.")

if __name__ == "__main__":
    main()
