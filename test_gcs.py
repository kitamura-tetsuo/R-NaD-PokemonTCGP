import tensorflow as tf
import os

checkpoint_dir = "gs://r-nad-pokemontcgp-checkpoints/R-NaD-PokemonTCGP_experiments/checkpoints"
try:
    print(f"Checking {checkpoint_dir}...")
    exists = tf.io.gfile.exists(checkpoint_dir)
    print(f"Exists: {exists}")
    if exists:
        files = tf.io.gfile.listdir(checkpoint_dir)
        print(f"Files: {files}")
except Exception as e:
    print(f"Error: {e}")
