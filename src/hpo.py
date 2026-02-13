import optuna
import mlflow
import os
import logging
import sys
import argparse
import jax
import numpy as np

# Adjust path to import src modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rnad import train_loop, RNaDConfig, LeagueConfig
from src.training.experiment import ExperimentManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Detect TPU once
try:
    devices = jax.devices()
    HAS_TPU = any(d.platform == 'tpu' for d in devices)
except Exception:
    HAS_TPU = False

def objective(trial: optuna.Trial, args: argparse.Namespace):
    # 1. Sample Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    entropy_schedule_start = trial.suggest_float("entropy_schedule_start", 0.05, 0.5)
    
    if HAS_TPU:
        # Standard search space for TPU
        transformer_embed_dim = trial.suggest_categorical("transformer_embed_dim", [64, 128, 256])
        transformer_layers = trial.suggest_categorical("transformer_layers", [2, 3, 4, 8, 12])
        transformer_heads = trial.suggest_categorical("transformer_heads", [2, 4, 8])
        transformer_seq_len = trial.suggest_categorical("transformer_seq_len", [4, 8, 16, 32])
    else:
        # Tiny search space for CPU/GPU testing to avoid OOM
        logging.info("Non-TPU environment detected. Using tiny transformer parameters for testing.")
        transformer_embed_dim = trial.suggest_categorical("transformer_embed_dim", [4, 8, 16, 256])
        transformer_layers = trial.suggest_categorical("transformer_layers", [1, 2, 12])
        transformer_heads = trial.suggest_categorical("transformer_heads", [1, 2, 8])
        transformer_seq_len = trial.suggest_categorical("transformer_seq_len", [1, 32])
    
    # 2. League Config Assembly
    league_config = None
    if args.league_decks_student or args.league_decks_teacher:
        league_config = LeagueConfig.from_csv(args.league_decks_student, args.league_decks_teacher)

    # 3. Config Assembly
    config = RNaDConfig(
        learning_rate=learning_rate,
        entropy_schedule_start=entropy_schedule_start,
        transformer_embed_dim=transformer_embed_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_seq_len=transformer_seq_len,
        
        # Optimization settings from CLI
        max_steps=args.max_steps,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        
        # Scaling settings from CLI
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        update_batch_size=args.update_batch_size,
        num_workers=args.num_workers,
        
        # Reward settings from CLI
        win_reward=args.win_reward,
        point_reward=args.point_reward,
        damage_reward=args.damage_reward,
        timeout_reward=args.timeout_reward,
        
        # Environment settings
        league_config=league_config,
        model_type="transformer",
        test_games=args.test_games
    )
    
    # 4. Experiment Manager (MLflow)
    run_name = f"hpo_trial_{trial.number}"
    experiment_manager = ExperimentManager(experiment_name="RNaD_HPO", run_id=None)
    
    # Log trial params to MLflow
    experiment_manager.log_params(config)
    experiment_manager.log_params({"trial_number": trial.number})
    
    # 5. Run Training
    # Save HPO checkpoints in output_dir/checkpoints_hpo
    hpo_checkpoint_root = os.path.join(args.output_dir, "checkpoints_hpo")
    checkpoint_dir = os.path.join(hpo_checkpoint_root, run_name)
    
    try:
        train_loop(
            config=config,
            experiment_manager=experiment_manager,
            checkpoint_dir=checkpoint_dir,
            trial=trial
        )
    except optuna.TrialPruned:
        logging.info(f"Trial {trial.number} pruned.")
        mlflow.end_run()
        raise
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {e}")
        mlflow.end_run()
        raise e
        
    # 6. Retrieve result
    frozen_trial = trial.study.get_trials()[-1] # Get latest finished trial
    if frozen_trial.intermediate_values:
        final_winrate = max(frozen_trial.intermediate_values.values())
        mlflow.log_metric("final_max_winrate", final_winrate)
        mlflow.end_run()
        return final_winrate
    else:
        mlflow.end_run()
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="RNaD Hyperparameter Optimization with Optuna")
    
    # Fixed parameters (not optimized but configurable)
    parser.add_argument("--batch_size", type=int, default=128 if HAS_TPU else 1)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--update_batch_size", type=int, default=None, help="Batch size for gradient updates (None means same as batch_size)")
    parser.add_argument("--num_workers", type=int, default=32 if HAS_TPU else 8)
    parser.add_argument("--test_games", type=int, default=32 if HAS_TPU else 4)
    parser.add_argument("--win_reward", type=float, default=1.0)
    parser.add_argument("--point_reward", type=float, default=0.1)
    parser.add_argument("--damage_reward", type=float, default=0.01)
    parser.add_argument("--timeout_reward", type=float, default=None)
    parser.add_argument("--league_decks_student", type=str, default=None)
    parser.add_argument("--league_decks_teacher", type=str, default=None)
    
    # Optimization loop parameters
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save hpo.db and best_params.txt")
    parser.add_argument("--storage", type=str, default=None, help="Database storage URL. Defaults to sqlite:///{output_dir}/hpo.db")
    
    args = parser.parse_args()
    
    # Ensure output_dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    # Set default storage
    if args.storage is None:
        args.storage = f"sqlite:///{os.path.join(args.output_dir, 'hpo.db')}"
    
    # If study_name not provided, use default based on environment
    if args.study_name is None:
        args.study_name = "rnad_hpo" if HAS_TPU else "rnad_hpo_tiny"
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2000)
    )
    
    # Seed HPO with train.py defaults if this is a fresh study or if requested
    # check if study has any trials
    if len(study.trials) == 0:
        logging.info("Enqueuing initial trial with train.py defaults...")
        study.enqueue_trial({
            "learning_rate": 3e-4,
            "entropy_schedule_start": 0.1,
            "transformer_embed_dim": 256,
            "transformer_layers": 12,
            "transformer_heads": 8,
            "transformer_seq_len": 32
        })
    
    logging.info(f"Starting HPO (Study: {args.study_name}, TPU: {HAS_TPU})...")
    logging.info(f"Fixed params: batch_size={args.batch_size}, update_batch_size={args.update_batch_size}, accumulation={args.accumulation_steps}, workers={args.num_workers}")
    logging.info(f"Output directory: {args.output_dir}")
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    logging.info("HPO Complete.")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")
    
    # Save best params to file in output_dir
    best_params_path = os.path.join(args.output_dir, "best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(str(study.best_params))
    logging.info(f"Saved best params to {best_params_path}")

if __name__ == "__main__":
    main()
