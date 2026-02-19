import hashlib
import sys
import os
from tqdm import tqdm

def dedup_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    output_path = filepath + ".tmp"
    seen_hashes = set()
    
    print(f"Processing {filepath}...")
    
    # Get file size for progress bar
    total_size = os.path.getsize(filepath)
    
    with open(filepath, 'rb') as f_in, open(output_path, 'wb') as f_out:
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        for line in f_in:
            progress_bar.update(len(line))
            
            # Remove whitespace from ends for robust comparison if desired, 
            # but usually for JSONL strict line duplication is checking exact bytes.
            # User said "duplicate lines", so we check exact line bytes.
            line_hash = hashlib.md5(line).digest()
            
            if line_hash not in seen_hashes:
                seen_hashes.add(line_hash)
                f_out.write(line)
                
        progress_bar.close()
    
    # Check if we actually removed anything
    original_lines = 0 # Not counting lines to save time, but could.
    # We can rely on file size diff.
    
    new_size = os.path.getsize(output_path)
    
    print(f"Original size: {total_size / (1024*1024):.2f} MB")
    print(f"New size:      {new_size / (1024*1024):.2f} MB")
    
    if new_size < total_size:
        print("Duplicates removed. Overwriting original file.")
        os.replace(output_path, filepath)
    else:
        print("No duplicates found. Keeping original file.")
        os.remove(output_path)

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "mined_data.jsonl"
    dedup_file(target_file)
