import os
import sys
from PIL import Image
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

results_dir = r'G:\My Drive\Project_Folder\Results'
files = [f for f in os.listdir(results_dir) if 'Full' in f]

print("Found files:", files)
for f in sorted(files):
    fpath = os.path.join(results_dir, f)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"\n--- {f} ({size_mb:.2f} MB) ---")
    if f.endswith('.png') or f.endswith('.jpg'):
        im = Image.open(fpath)
        print(f"Dimensions: {im.size} (Width x Height), Mode: {im.mode}")
        arr = np.array(im)
        print(f"Array shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, max: {arr.max()}, mean: {arr.mean():.2f}")
    elif f.endswith('.csv'):
        with open(fpath, 'r', encoding='utf-8') as cf:
            print("Content:\n" + cf.read().strip())

