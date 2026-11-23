import zipfile, os, glob, re
import numpy as np
from PIL import Image

def analyze_zip(zip_path):
    print(f"=== Analyzing {zip_path} ===")
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.endswith('.tif') or name.endswith('.tiff'):
                with z.open(name) as f:
                    im = Image.open(f)
                    arr = np.array(im, dtype=np.float32)
                    print(f"Band: {os.path.basename(name)}")
                    print(f"  Shape: {arr.shape}, Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}, P2: {np.percentile(arr, 2)}, P98: {np.percentile(arr, 98)}")
            
analyze_zip('data/Browser_images_new1.zip')

