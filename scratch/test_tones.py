import tifffile
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image

# Load the 10-band 1.5m VHR GeoTIFF
arr = tifffile.imread(r'G:\My Drive\Project_Folder\Results\sen2sr_cascaded_1.5m_vhr.tif')
# BAND_NAMES = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
# RGB = [B04, B03, B02] = [2, 1, 0]
rgb_raw = arr[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)

print("RGB raw shape:", rgb_raw.shape, "min:", rgb_raw.min(), "max:", rgb_raw.max(), "mean:", rgb_raw.mean(axis=(0,1)))

# Method 0: Current notebook method (linear 2-98% percentile)
lo = np.percentile(rgb_raw, 2)
hi = np.percentile(rgb_raw, 98)
m0 = np.clip((rgb_raw - lo) / (hi - lo), 0.0, 1.0)

# Method 1: Natural Gamma Correction (eliminates over-bright washed out look)
# Remote sensing standard: gamma 1.5 - 1.8 + percentile stretch with 1% lower and 99% upper
gamma = 1.6
m1_linear = np.clip((rgb_raw - np.percentile(rgb_raw, 1)) / (np.percentile(rgb_raw, 99) - np.percentile(rgb_raw, 1)), 0.0, 1.0)
m1_gamma = np.power(m1_linear, gamma)

# Method 2: CLAHE + High-Frequency Edge Enhancement in LAB color space
# (Preserves true spectral colors, boosts structural contrast & building sharpness)
rgb_u8 = (m1_gamma * 255.0).astype(np.uint8)
lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
l_chan, a_chan, b_chan = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
l_clahe = clahe.apply(l_chan)

# High-frequency unsharp mask on luminance channel only
l_float = l_clahe.astype(np.float32)
l_blur = gaussian_filter(l_float, sigma=1.2)
l_edge = l_float - l_blur
l_crisp = np.clip(l_float + 1.2 * l_edge, 0, 255).astype(np.uint8)

lab_crisp = cv2.merge([l_crisp, a_chan, b_chan])
m2_crisp = cv2.cvtColor(lab_crisp, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

# Method 3: Balanced Photographic Satellite Tone Mapping (Natural Google Earth Look)
# 1. Shadow boost + Highlight compression + Edge sharpening
m3 = np.zeros_like(rgb_raw)
for c in range(3):
    ch = rgb_raw[:, :, c]
    p_min = np.percentile(ch, 1.0)
    p_max = np.percentile(ch, 98.5)
    ch_norm = np.clip((ch - p_min) / (p_max - p_min), 0.0, 1.0)
    # Gamma 1.4 for natural midtones
    ch_gamma = np.power(ch_norm, 1.4)
    # Edge sharpening
    blur = gaussian_filter(ch_gamma, sigma=1.0)
    edge = ch_gamma - blur
    m3[:, :, c] = np.clip(ch_gamma + 1.1 * edge, 0.0, 1.0)

# Create side-by-side comparison of a 400x400 crop showing Graphic Era buildings
h, w, _ = rgb_raw.shape
cy, cx = h // 2, w // 2
crop_sz = 350

crops = [
    ("Current Blown-out / Bright", m0[cy-crop_sz//2:cy+crop_sz//2, cx-crop_sz//2:cx+crop_sz//2]),
    ("Gamma Corrected (1.6)", m1_gamma[cy-crop_sz//2:cy+crop_sz//2, cx-crop_sz//2:cx+crop_sz//2]),
    ("CLAHE + High-Acuity LAB", m2_crisp[cy-crop_sz//2:cy+crop_sz//2, cx-crop_sz//2:cx+crop_sz//2]),
    ("Photographic Satellite VHR", m3[cy-crop_sz//2:cy+crop_sz//2, cx-crop_sz//2:cx+crop_sz//2])
]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
for idx, (title, crop) in enumerate(crops):
    axes[idx].imshow(crop)
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('scratch/tone_and_clarity_test.png', dpi=200, bbox_inches='tight')
print("Saved scratch/tone_and_clarity_test.png successfully!")

