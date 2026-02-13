import time
import psutil
import subprocess
import re
import sys
from datetime import datetime

def get_tpu_vm_util():
    try:
        result = subprocess.run(['tpu-info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        
        match = re.search(r"(\d+\.?\d*)\s*%", output)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception:
        return 0.0

def main():
    print("Monitoring TPU VM (via tpu-info)...")
    
    with open("stats.csv", "w") as f:
        f.write("timestamp,tpu_util,cpu,ram\n")

    try:
        while True:
            t = datetime.now().strftime("%H:%M:%S")
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            
            tpu = get_tpu_vm_util()
            
            log_line = f"{t},{tpu},{cpu},{ram}"
            print(log_line)
            
            with open("stats.csv", "a") as f:
                f.write(log_line + "\n")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()