import os, glob, re, zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

# Load data from Browser_images_new1.zip or Browser_images.zip
zip_path = 'data/Browser_images_new1.zip'
if not os.path.exists(zip_path):
    zip_path = 'G:/My Drive/Project_Folder/Datasets/Browser_images_new1.zip'

BAND_NAMES = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
from PIL import Image

bands = {}
with zipfile.ZipFile(zip_path, 'r') as z:
    for name in z.namelist():
        fname = os.path.basename(name).upper()
        for b in BAND_NAMES:
            if re.search(rf'(^|[_\.-]){b}([_\.-]|$)', fname):
                with z.open(name) as f:
                    bands[b] = np.array(Image.open(f), dtype=np.float32)

print("Loaded bands:", list(bands.keys()))
ref_shape = bands['B02'].shape

# Stack into 10 bands
stack = []
for b in BAND_NAMES:
    arr = bands[b]
    if arr.shape != ref_shape:
        arr = cv2.resize(arr, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_LINEAR)
    stack.append(arr)

raw_10b = np.stack(stack)
p99 = np.percentile(raw_10b[np.isfinite(raw_10b)], 99)
div = 10000.0 if p99 > 100.0 else 1.0
raw_norm = np.clip(raw_10b / div, 0.0, 1.0)

# Simulate 8x upscale
raw_t = torch.from_numpy(raw_norm).unsqueeze(0)
in_8x = F.interpolate(raw_t, scale_factor=8, mode='bicubic', align_corners=False).squeeze(0).numpy()

# Also load the VHR GeoTIFF from Google Drive
import tifffile
vhr_tif = tifffile.imread(r'G:\My Drive\Project_Folder\Results\sen2sr_cascaded_1.5m_vhr.tif')
print("VHR tif shape:", vhr_tif.shape)

# Let's test the photographic display pipeline on row 1 (True Color RGB: B04, B03, B02 = [2, 1, 0])
def photographic_rgb(arr_3ch, ref_bounds=None, gamma=1.45, clahe_clip=0.0, edge_boost=0.0):
    # arr_3ch: (3, H, W) in [Red, Green, Blue] order
    img = arr_3ch.transpose(1, 2, 0).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0)
    
    if ref_bounds is None:
        lo = np.percentile(img, 1.5)
        hi = np.percentile(img, 98.5)
    else:
        lo, hi = ref_bounds
        
    # 1. Percentile contrast stretch
    stretched = np.clip((img - lo) / max(hi - lo, 1e-5), 0.0, 1.0)
    
    # 2. Gamma correction (removes blown-out bright appearance, restores shadow depth)
    gamma_corrected = np.power(stretched, gamma)
    
    # 3. LAB luminance sharpening if requested
    if clahe_clip > 0 or edge_boost > 0:
        u8 = (gamma_corrected * 255.0).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        if clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
        if edge_boost > 0:
            l_f = l.astype(np.float32)
            blur = cv2.GaussianBlur(l_f, (0, 0), sigmaX=1.1)
            edge = l_f - blur
            l = np.clip(l_f + edge_boost * edge, 0, 255).astype(np.uint8)
            
        res = cv2.merge([l, a, b])
        return cv2.cvtColor(res, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0, (lo, hi)
        
    return gamma_corrected, (lo, hi)

# Reference bounds from native input
rgb_in = in_8x[[2, 1, 0]]
rgb_vhr = vhr_tif[[2, 1, 0]]

_, bounds = photographic_rgb(rgb_in, gamma=1.45)

old_linear = np.clip((rgb_in.transpose(1,2,0) - np.percentile(rgb_in, 2)) / (np.percentile(rgb_in, 98) - np.percentile(rgb_in, 2)), 0, 1)
new_p1, _ = photographic_rgb(rgb_in, ref_bounds=bounds, gamma=1.45)
new_vhr_sharp, _ = photographic_rgb(rgb_vhr, ref_bounds=bounds, gamma=1.45, clahe_clip=1.6, edge_boost=1.5)

print("Old Linear RGB mean:", old_linear.mean(axis=(0,1)))
print("New Calibrated Native RGB mean:", new_p1.mean(axis=(0,1)))
print("New Calibrated VHR RGB mean:", new_vhr_sharp.mean(axis=(0,1)))

