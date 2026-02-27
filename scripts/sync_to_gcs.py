#!/usr/bin/env python3
import os
import re
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local project directory (assumed to be the current directory)
BASE_DIR = os.getcwd()
DEST_BASE = "gs://r-nad-pokemontcgp-checkpoints/R-NaD-PokemonTCGP_experiments"

def run_gsutil(cmd_list):
    """Executes a gsutil command."""
    full_cmd = ["gsutil"] + cmd_list
    logging.info(f"Running: {' '.join(full_cmd)}")
    try:
        subprocess.run(full_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error: {e}")
    except FileNotFoundError:
        logging.error("gsutil command not found. Please ensure Google Cloud SDK is installed and in your PATH.")

def sync_ckpts_in_dir(path, is_root=False, run_id=None):
    """Syncs oldest and newest .pkl checkpoints in a directory."""
    if not os.path.exists(path):
        return
    files = os.listdir(path)
    ckpt_files = []
    for f in files:
        if f.endswith(".pkl"):
            m = re.search(r"(\d+)", f)
            if m:
                ckpt_files.append((int(m.group(1)), f))
    
    if ckpt_files:
        ckpt_files.sort()
        targets = [ckpt_files[0][1], ckpt_files[-1][1]]
        targets = list(set(targets))
        
        dest_label = "root" if is_root else run_id
        logging.info(f"  Syncing {dest_label}: targets {targets}")
        
        for t in targets:
            src = os.path.join(path, t)
            if is_root:
                # ルート直下のものは checkpoints/ 直下に置く
                dst = f"{DEST_BASE}/checkpoints/{t}"
            else:
                # ランIDがあるものは checkpoints/run_id/ 直下に置く
                dst = f"{DEST_BASE}/checkpoints/{run_id}/{t}"
            run_gsutil(["cp", src, dst])

def main():
    # 1. mlflow のデータは全て同期 (mlruns フォルダを同期)
    mlruns_path = os.path.join(BASE_DIR, "mlruns")
    if os.path.exists(mlruns_path):
        logging.info("Syncing all mlflow data (mlruns)...")
        run_gsutil(["-m", "rsync", "-r", mlruns_path, f"{DEST_BASE}/mlruns"])
    else:
        logging.warning(f"MlFlow directory not found at {mlruns_path}")

    # mlflow.db も同期
    mlflow_db = os.path.join(BASE_DIR, "mlflow.db")
    if os.path.exists(mlflow_db):
        logging.info("Syncing mlflow.db...")
        run_gsutil(["cp", mlflow_db, f"{DEST_BASE}/mlflow.db"])

    # 2. チェックポイントは各ランごとに最古と最新のみ同期
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    if os.path.exists(ckpt_dir):
        logging.info("Syncing oldest and newest .pkl checkpoints...")
        
        # A. ルートディレクトリを同期
        sync_ckpts_in_dir(ckpt_dir, is_root=True)
        
        # B. 各ランIDディレクトリを同期
        for run_id in os.listdir(ckpt_dir):
            run_path = os.path.join(ckpt_dir, run_id)
            if os.path.isdir(run_path):
                sync_ckpts_in_dir(run_path, is_root=False, run_id=run_id)
    else:
        logging.warning(f"Checkpoint directory not found at {ckpt_dir}")

if __name__ == "__main__":
    main()
