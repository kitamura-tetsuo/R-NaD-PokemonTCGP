import os
import time
import psutil
import pandas as pd
import tensorflow as tf
from tensorflow.python.profiler import profiler_client
import re
import sys

def get_tpu_util(tpu_addr):
    try:
        # ポートを8466に変換
        addr = tpu_addr.replace('8470', '8466')
        res = profiler_client.monitor(addr, duration_ms=1000, level=2)
        match = re.search(r"Utilization of TPU Matrix Units .*?:\s+([\d\.]+)%", res)
        return float(match.group(1)) if match else 0.0
    except:
        return 0.0

def main():
    tpu_addr = os.environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        print("Error: COLAB_TPU_ADDR not found.")
        sys.exit(1)

    print("Time,TPU_MXU,CPU,RAM")
    # CSVのヘッダーを作成
    with open("stats.csv", "w") as f:
        f.write("timestamp,tpu_mxu,cpu,ram\n")

    try:
        while True:
            t = time.strftime("%H:%M:%S")
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            tpu = get_tpu_util(tpu_addr)
            
            log_line = f"{t},{tpu},{cpu},{ram}"
            print(log_line) # ターミナルに表示
            
            with open("stats.csv", "a") as f:
                f.write(log_line + "\n")
            
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()