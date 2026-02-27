import mlflow
import orbax.checkpoint
import os
from typing import Any, Dict
import logging

class ExperimentManager:
    def __init__(self, experiment_name: str, checkpoint_dir: str = "checkpoints", run_id: str = None, log_checkpoints: bool = False):
        mlflow.set_experiment(experiment_name=experiment_name)
        self.log_checkpoints = log_checkpoints

        # Configure CheckpointManager
        # Start a run explicitly if not already active
        if run_id:
            if mlflow.active_run() and mlflow.active_run().info.run_id != run_id:
                mlflow.end_run()
            if not mlflow.active_run():
                try:
                    mlflow.start_run(run_id=run_id)
                except Exception as e:
                    logging.warning(f"Failed to resume run {run_id}: {e}. Starting a new run.")
                    mlflow.start_run()
        elif not mlflow.active_run():
            mlflow.start_run()

        self.run_id = mlflow.active_run().info.run_id

        # We save checkpoints using run_id to avoid conflicts
        if checkpoint_dir.startswith("gs://"):
            self.checkpoint_dir = os.path.join(checkpoint_dir, self.run_id)
        elif checkpoint_dir.startswith("/gcs/"):
            # For GCS FUSE, we keep it as is to avoid abspath resolving to local root if not intended
            self.checkpoint_dir = os.path.join(checkpoint_dir, self.run_id)
        else:
            self.checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, self.run_id))

        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
        # Use StandardCheckpointer for PyTrees
        self.checkpointer = orbax.checkpoint.StandardCheckpointer()
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self.checkpoint_dir,
            self.checkpointer,
            options=options
        )
        logging.info(f"Initialized ExperimentManager with checkpoint dir: {self.checkpoint_dir}, log_checkpoints: {self.log_checkpoints}")

    def log_params(self, config: Any):
        """Logs configuration parameters to MLflow."""
        if hasattr(config, '_asdict'):
            params = dict(config._asdict())
        elif isinstance(config, dict):
            params = dict(config)
        else:
            try:
                params = dict(vars(config))
            except TypeError:
                params = {"config": str(config)}

        # Define keys to exclude from parameters and which ones should be tags
        exclude_from_params = ["batch_size", "max_steps", "deck_id_1", "deck_id_2", "log_interval", "save_interval", "league_config", "accumulation_steps", "damage_reward", "enable_profiler", "model_type", "num_buffers", "num_workers", "past_self_play", "point_reward", "profile_num_steps", "profile_start_step", "profiler_dir", "runs/profile", "seed", "test_games", "test_interval", "timeout_reward", "transformer", "tuned_batch_size", "tuned_num_workers", "tuned_update_batch_size", "update_batch_size", "win_reward", "unroll_length"]
        save_as_tags = ["batch_size", "max_steps", "log_interval", "save_interval", "unroll_length"]

        # Set tags
        for key in save_as_tags:
            if key in params:
                mlflow.set_tag(key, str(params[key]))

        # Filter params and ensure values are loggable (int, float, string, bool)
        filtered_params = {}
        for k, v in params.items():
            if k in exclude_from_params:
                continue
            
            # Explicitly ensure transformer parameters and others are logged correctly
            # MLflow log_params accepts string, int, float
            if isinstance(v, (int, float, str, bool)) or v is None:
                filtered_params[k] = v
            else:
                # Convert complex objects (like LeagueConfig) to string to avoid MLflow errors
                filtered_params[k] = str(v)
        
        if filtered_params:
            mlflow.log_params(filtered_params)

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """Logs metrics to MLflow."""
        flat_metrics = {}

        def flatten(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten(v, prefix + k + '.')
                else:
                    try:
                        flat_metrics[prefix + k] = float(v)
                    except (ValueError, TypeError):
                        pass # Skip non-numeric metrics

        flatten(metrics)
        mlflow.log_metrics(flat_metrics, step=step)

    def save_model(self, step: int, params: Any, fixed_params: Any = None, opt_state: Any = None):
        """Saves model parameters and state using Orbax and logs as MLflow artifact."""
        # Save locally
        ckpt_dict = {'params': params}
        if fixed_params is not None:
            ckpt_dict['fixed_params'] = fixed_params
        if opt_state is not None:
            ckpt_dict['opt_state'] = opt_state

        save_args = orbax.checkpoint.args.StandardSave(ckpt_dict)
        self.checkpoint_manager.save(step, args=save_args)

        # Ensure save is complete before logging artifact
        self.checkpoint_manager.wait_until_finished()

        # Log as artifact
        # Path to the specific step checkpoint
        step_checkpoint_path = os.path.join(self.checkpoint_dir, str(step))

        if self.log_checkpoints and os.path.exists(step_checkpoint_path):
            # Log the directory as an artifact in a 'checkpoints' folder in MLflow
            mlflow.log_artifacts(step_checkpoint_path, artifact_path=f"checkpoints/step_{step}")
            logging.info(f"Saved checkpoint for step {step} to MLflow.")
        elif not self.log_checkpoints:
            logging.info(f"Skipping checkpoint upload to MLflow for step {step} (disabled).")
        else:
            logging.warning(f"Checkpoint path {step_checkpoint_path} does not exist.")
