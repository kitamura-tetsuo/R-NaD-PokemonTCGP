import shutil
import os
import unittest
import jax.numpy as jnp
from src.training.experiment import ExperimentManager
import mlflow

class TestExperimentManager(unittest.TestCase):
    def setUp(self):
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        if os.path.exists("mlruns"):
            shutil.rmtree("mlruns")
        self.mgr = ExperimentManager("test_experiment")

    def tearDown(self):
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        if os.path.exists("mlruns"):
            shutil.rmtree("mlruns")
        # End run if active
        if mlflow.active_run():
             mlflow.end_run()

    def test_log_params(self):
        config = {'a': 1, 'b': 2}
        self.mgr.log_params(config)

    def test_log_metrics(self):
        metrics = {'loss': 0.1, 'acc': 0.9}
        self.mgr.log_metrics(0, metrics)

    def test_save_model(self):
        params = {'w': jnp.zeros((2, 2))}
        self.mgr.save_model(0, params)

        # Check if checkpoint exists in the manager's checkpoint_dir
        self.assertTrue(os.path.exists(os.path.join(self.mgr.checkpoint_dir, "0")))

        # Check if MLflow artifact exists (simulated check)
        self.assertTrue(os.path.exists("mlruns"))

if __name__ == '__main__':
    unittest.main()
