import optuna
import subprocess
import re
import logging
import sys
import numpy as np
import os
import shutil
import argparse

# Configure logging to output to stdout for visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def make_objective(args):
    def objective(trial):
        # 1. Suggest parameters
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048])
        num_workers = trial.suggest_categorical("num_workers", [2, 4, 8, 16, 32])

        # 2. Prepare command
        # Find train.py path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_py = os.path.join(script_dir, "..", "train.py")
        if not os.path.exists(train_py):
            train_py = "train.py" # Fallback to current dir if not found relative to script
        
        checkpoint_dir = f"tune_checkpoints/trial_{trial.number}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Use provided update_batch_size if specified, otherwise default to 64 for OOM safety
        update_bs = str(args.update_batch_size) if args.update_batch_size else "64"

        cmd = [
            sys.executable, train_py,
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
            "--update_batch_size", update_bs,
            "--max_steps", str(args.max_steps),
            "--unroll_length", str(args.unroll_length),
            "--transformer_layers", str(args.transformer_layers),
            "--transformer_heads", str(args.transformer_heads),
            "--transformer_embed_dim", str(args.transformer_embed_dim),
            "--transformer_seq_len", str(args.transformer_seq_len),
            "--log_interval", "1",
            "--save_interval", "1000",       # Do not save checkpoints during tuning
            "--checkpoint_dir", checkpoint_dir,
            "--disable_mlflow",              # Disable MLflow during tuning
            "--seed", "42"
        ]

        logging.info(f"Trial {trial.number}: testing batch_size={batch_size}, num_workers={num_workers}")
        logging.info(f"Command: {' '.join(cmd)}")

        try:
            # 3. Execute train.py as a subprocess
            # We capture stdout/stderr to extract SPS values
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            sps_values = []
            # Pattern to find 'sps': 123.45 in the log output
            sps_pattern = re.compile(r"'sps': ([\d\.\+e-]+)")

            for line in process.stdout:
                # We can optionally log the subprocess output for debugging
                # logging.debug(line.strip())
                
                match = sps_pattern.search(line)
                if match:
                    sps = float(match.group(1))
                    sps_values.append(sps)
                    # Optional: log intermediate results
                    if len(sps_values) % 10 == 0:
                        logging.info(f"Trial {trial.number} Progress: Steps={len(sps_values)}, Latest SPS={sps:.2f}")

            process.wait()

            # 4. Handle failure (OOM, etc.)
            if process.returncode != 0:
                logging.warning(f"Trial {trial.number} failed with return code {process.returncode} (likely OOM or crash).")
                # Cleanup
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                return float('-inf')

            if not sps_values:
                logging.warning(f"Trial {trial.number}: No SPS values found in logs.")
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                return float('-inf')

            # 5. Calculate objective value
            # Ignore the first few steps (compilation period)
            if len(sps_values) > 2:
                # Discard early samples to avoid compilation/init bias
                # For short runs (max_steps=5), we might only have a few.
                # If we have enough, skip the first 2.
                final_sps = np.mean(sps_values[min(2, len(sps_values)-1):])
            else:
                final_sps = np.mean(sps_values)

            logging.info(f"Trial {trial.number} finished. Calculated SPS: {final_sps:.2f}")

            # Cleanup trial-specific checkpoint directory
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)

            return final_sps

        except Exception as e:
            logging.error(f"Trial {trial.number} encountered an error: {e}")
            return float('-inf')
    return objective

def main():
    parser = argparse.ArgumentParser(description="Performance tuning for R-NaD using Optuna")
    parser.add_argument("--unroll_length", type=int, default=200, help="Fixed unroll length for trajectory generation. max is 1000")
    parser.add_argument("--transformer_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--transformer_heads", type=int, default=8, help="Number of transformer heads")
    parser.add_argument("--transformer_embed_dim", type=int, default=256, help="Embedding dimension for transformer")
    parser.add_argument("--transformer_seq_len", type=int, default=32, help="Sequence length for transformer")
    parser.add_argument("--update_batch_size", type=int, default=None, help="Batch size for gradient updates (None means same as batch_size)")
    parser.add_argument("--max_steps", type=int, default=5, help="Number of steps per trial")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials to execute in this run")
    parser.add_argument("--storage", type=str, default="sqlite:///performance_tuning.db", help="Optuna storage URL (e.g., sqlite:///tuning.db)")
    parser.add_argument("--study_name", type=str, default="performance_tuning", help="Name of the Optuna study")
    args = parser.parse_args()

    # Setup base tuning directory
    os.makedirs("tune_checkpoints", exist_ok=True)
    
    # Create or load study with RDB storage for persistence
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True
    )
    
    # Run optimization
    logging.info(f"Starting/Resuming study '{args.study_name}' using storage '{args.storage}'")
    try:
        study.optimize(make_objective(args), n_trials=args.n_trials)
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user.")

    # Show results
    if len(study.trials) > 0:
        print("\n" + "="*60)
        print(" PERFORMANCE TUNING RESULTS ".center(60, "="))
        print("="*60)
        try:
            print(f"Best Trial:  {study.best_trial.number}")
            print(f"Best SPS:    {study.best_value:.2f}")
            print("-" * 60)
            print("Best Parameters:")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")
        except ValueError:
            print("No successful trials found.")
        print("="*60)

        # Suggest next steps
        try:
            params = study.best_trial.params
            print("\nNext steps:")
            print(f"Use these parameters in your training script:")
            print(f"python train.py --batch_size {params['batch_size']} --num_workers {params['num_workers']} --update_batch_size {args.update_batch_size or 64} ...")
        except ValueError:
            pass

if __name__ == "__main__":
    main()
