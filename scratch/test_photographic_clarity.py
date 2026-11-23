import tifffile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the saved VHR GeoTIFF from Google Drive
tif_path = r'G:\My Drive\Project_Folder\Results\sen2sr_cascaded_1.5m_vhr.tif'
vhr_data = tifffile.imread(tif_path) # (10, 776, 760)

# Also let's inspect the 4 stages of RGB from the comparison pipeline
# BAND_NAMES: B02=0, B03=1, B04=2, B05=3, B06=4, B07=5, B08=6, B8A=7, B11=8, B12=9
# Stage 2 RGB is bands [2, 1, 0] = [B04, B03, B02]
vhr_rgb = vhr_data[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)

print("VHR RGB shape:", vhr_rgb.shape)

# Let's design the optimal Photographic VHR Display Function:
def enhance_photographic_clarity(rgb_in, gamma=1.5, clahe_strength=1.5, edge_strength=1.4):
    """
    1. Calibrated Percentile Normalization: 1% dark-level to 99% highlight.
    2. Gamma Correction: Transforms linear physical radiance into perceptually balanced sRGB,
       eliminating the over-bright, washed-out appearance.
    3. Luminance Contrast & Edge Sharpening in LAB space:
       Sharpen building footprints, roads, and parcel lines without shifting spectral hue.
    """
    h, w, c = rgb_in.shape
    out = np.zeros_like(rgb_in)
    
    # 1. Percentile stretch with natural bounds
    p_low = np.percentile(rgb_in, 1.0)
    p_high = np.percentile(rgb_in, 99.0)
    norm = np.clip((rgb_in - p_low) / max(p_high - p_low, 1e-6), 0.0, 1.0)
    
    # 2. Gamma compression (balances bright rooftops and deep shadows)
    norm_gamma = np.power(norm, gamma)
    
    # 3. LAB Luminance Sharpening
    u8 = (norm_gamma * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Local contrast enhancement via CLAHE
    if clahe_strength > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_strength, tileGridSize=(8, 8))
        l = clahe.apply(l)
    
    # High-acuity edge boost on luminance
    l_f = l.astype(np.float32)
    l_blur = cv2.GaussianBlur(l_f, (0, 0), sigmaX=1.0)
    l_edge = l_f - l_blur
    l_crisp = np.clip(l_f + edge_strength * l_edge, 0, 255).astype(np.uint8)
    
    lab_crisp = cv2.merge([l_crisp, a, b])
    rgb_crisp = cv2.cvtColor(lab_crisp, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return rgb_crisp

# Test across multiple parameters
p_orig = np.clip((vhr_rgb - np.percentile(vhr_rgb, 2)) / (np.percentile(vhr_rgb, 98) - np.percentile(vhr_rgb, 2)), 0, 1)
p_balanced = enhance_photographic_clarity(vhr_rgb, gamma=1.4, clahe_strength=1.2, edge_strength=1.0)
p_crisp = enhance_photographic_clarity(vhr_rgb, gamma=1.5, clahe_strength=1.8, edge_strength=1.5)
p_ultra = enhance_photographic_clarity(vhr_rgb, gamma=1.6, clahe_strength=2.2, edge_strength=1.8)

# Center crop on Graphic Era University campus
cy, cx = vhr_rgb.shape[0] // 2, vhr_rgb.shape[1] // 2
cs = 320

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(p_orig[cy-cs//2:cy+cs//2, cx-cs//2:cx+cs//2])
axes[0].set_title("Previous: Blown-out & Blurred (Linear 2-98%)", fontsize=11, fontweight='bold', color='red')

axes[1].imshow(p_balanced[cy-cs//2:cy+cs//2, cx-cs//2:cx+cs//2])
axes[1].set_title("Balanced: Gamma 1.4 + Moderate Sharpness", fontsize=11, fontweight='bold')

axes[2].imshow(p_crisp[cy-cs//2:cy+cs//2, cx-cs//2:cx+cs//2])
axes[2].set_title("High-Clarity VHR: Gamma 1.5 + CLAHE + Edge Acuity", fontsize=11, fontweight='bold', color='green')

axes[3].imshow(p_ultra[cy-cs//2:cy+cs//2, cx-cs//2:cx+cs//2])
axes[3].set_title("Ultra-Crisp VHR: Deep Shadows & Sharp Rooflines", fontsize=11, fontweight='bold')

for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('scratch/photographic_clarity_showcase.png', dpi=200, bbox_inches='tight')
print("Showcase saved to scratch/photographic_clarity_showcase.png")

