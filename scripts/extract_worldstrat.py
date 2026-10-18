"""
WorldStrat High-Resolution Archive Extraction Utility
Extracts hr_dataset.tar.gz safely with progress tracking and selective extraction support.
"""

import os
import tarfile
import argparse
from tqdm import tqdm

def extract_worldstrat(tar_path, output_dir, mode="selective", low_res_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isfile(tar_path):
        raise FileNotFoundError(f"Archive not found at: {tar_path}")
        
    matching_prefixes = set()
    if mode == "selective":
        if low_res_dir and os.path.exists(low_res_dir):
            matching_prefixes = set(os.listdir(low_res_dir))
            print(f"[Selective Mode] Found {len(matching_prefixes)} target locations in {low_res_dir}")
        else:
            print("[Warning] low_res_dir not specified or not found. Falling back to extracting all files.")
            mode = "all"
            
    print(f"Opening archive: {tar_path} (Mode: {mode})")
    
    with tarfile.open(tar_path, "r:gz") as tar:
        # Stream members sequentially to prevent memory exhaustion
        extracted_count = 0
        for member in tqdm(tar, desc="Processing archive"):
            # Check if member belongs to a target location in selective mode
            top_folder = member.name.split("/")[0] if "/" in member.name else member.name
            
            if mode == "selective" and top_folder not in matching_prefixes:
                continue
                
            tar.extract(member, path=output_dir)
            extracted_count += 1
            
        print(f"\nExtraction complete! Extracted {extracted_count} items to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WorldStrat High-Res Dataset safely.")
    parser.add_argument("--tar", default="/content/drive/MyDrive/Project_Folder/Datasets/worldstrat_high/hr_dataset.tar.gz")
    parser.add_argument("--out", default="/content/drive/MyDrive/Project_Folder/Datasets/worldstrat_high/extracted")
    parser.add_argument("--low", default="/content/drive/MyDrive/Project_Folder/Datasets/worldstrat_low")
    parser.add_argument("--mode", default="selective", choices=["selective", "all"])
    args = parser.parse_args()
    
    extract_worldstrat(args.tar, args.out, mode=args.mode, low_res_dir=args.low)
