import mlflow
import orbax.checkpoint
import os
from typing import Any, Dict
import logging

class ExperimentManager:
    def __init__(self, experiment_name: str, checkpoint_dir: str = "checkpoints", run_id: str = None):
        mlflow.set_experiment(experiment_name=experiment_name)

        # Configure CheckpointManager
        # Start a run explicitly if not already active
        if run_id:
            if mlflow.active_run() and mlflow.active_run().info.run_id != run_id:
                mlflow.end_run()
            if not mlflow.active_run():
                mlflow.start_run(run_id=run_id)
        elif not mlflow.active_run():
            mlflow.start_run()

        self.run_id = mlflow.active_run().info.run_id

        # We save checkpoints locally first, using run_id to avoid conflicts
        self.checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, self.run_id))
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
        # Use StandardCheckpointer for PyTrees
        self.checkpointer = orbax.checkpoint.StandardCheckpointer()
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self.checkpoint_dir,
            self.checkpointer,
            options=options
        )
        logging.info(f"Initialized ExperimentManager with checkpoint dir: {self.checkpoint_dir}")

    def log_params(self, config: Any):
        """Logs configuration parameters to MLflow."""
        if hasattr(config, '_asdict'):
            params = config._asdict()
        elif isinstance(config, dict):
            params = config
        else:
            try:
                params = vars(config)
            except TypeError:
                params = {"config": str(config)}

        mlflow.log_params(params)

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

    def save_model(self, step: int, params: Any):
        """Saves model parameters using Orbax and logs as MLflow artifact."""
        # Save locally
        save_args = orbax.checkpoint.args.StandardSave(params)
        self.checkpoint_manager.save(step, args=save_args)

        # Ensure save is complete before logging artifact
        self.checkpoint_manager.wait_until_finished()

        # Log as artifact
        # Path to the specific step checkpoint
        step_checkpoint_path = os.path.join(self.checkpoint_dir, str(step))

        if os.path.exists(step_checkpoint_path):
            # Log the directory as an artifact in a 'checkpoints' folder in MLflow
            mlflow.log_artifacts(step_checkpoint_path, artifact_path=f"checkpoints/step_{step}")
            logging.info(f"Saved checkpoint for step {step} to MLflow.")
        else:
            logging.warning(f"Checkpoint path {step_checkpoint_path} does not exist.")
