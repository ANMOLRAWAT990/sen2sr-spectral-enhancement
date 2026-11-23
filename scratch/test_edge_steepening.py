import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load the VHR image from Google Drive
import tifffile
vhr_data = tifffile.imread(r'G:\My Drive\Project_Folder\Results\sen2sr_cascaded_1.5m_vhr.tif')
# RGB = B04, B03, B02 = bands [2, 1, 0]
rgb = vhr_data[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)

# Normalize
p1 = np.percentile(rgb, 1.5)
p99 = np.percentile(rgb, 98.5)
rgb_norm = np.clip((rgb - p1) / (p99 - p1), 0.0, 1.0)
rgb_gamma = np.power(rgb_norm, 1.45)

# 1. Test Shock Filter (Kramer-Bruckner / Osher-Rudin PDE edge steepening)
# Shock filtering sharpens edges by locally dilating on the convex side and eroding on the concave side
def shock_filter(img, iterations=5, dt=0.5):
    u = np.copy(img)
    for _ in range(iterations):
        for c in range(u.shape[2]):
            ch = u[:, :, c]
            lap = cv2.Laplacian(ch, cv2.CV_32F)
            dilated = cv2.dilate(ch, np.ones((3, 3), np.uint8))
            eroded = cv2.erode(ch, np.ones((3, 3), np.uint8))
            # If Laplacian < 0 (convex), edge is dilated; if > 0 (concave), edge is eroded
            flow = np.where(lap < 0, dilated - ch, eroded - ch)
            u[:, :, c] = np.clip(ch + dt * flow, 0.0, 1.0)
    return u

# 2. Test Morphological Gradient / Building Edge Snapping
def building_edge_enhancement(img, shock_iters=4, unsharp_strength=1.8):
    # Step A: PDE Shock filter to turn blurry edge transitions into sharp step edges
    steep = shock_filter(img, iterations=shock_iters, dt=0.5)
    
    # Step B: LAB Luminance High-Frequency Gradient Injection
    u8 = (steep * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Morphological gradient (detects building outlines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(l, cv2.MORPH_GRADIENT, kernel)
    
    # Edge steepening
    l_f = l.astype(np.float32)
    blur = cv2.GaussianBlur(l_f, (0, 0), sigmaX=0.8)
    high_freq = l_f - blur
    l_sharp = np.clip(l_f + unsharp_strength * high_freq + 0.3 * grad.astype(np.float32), 0, 255).astype(np.uint8)
    
    res = cv2.merge([l_sharp, a, b])
    return cv2.cvtColor(res, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

# Center crop on Graphic Era campus structures
cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
cs = 300

crop_base = rgb_gamma[cy-cs//2:cy+cs//2, cx-cs//2:cx+cs//2]
crop_shock = shock_filter(crop_base, iterations=5)
crop_building_snap = building_edge_enhancement(crop_base, shock_iters=4, unsharp_strength=2.0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(crop_base); axes[0].set_title("Current: Regression Model Output\n('Enhanced Blur' - Smooth Edges)", fontsize=11, fontweight='bold', color='red')
axes[1].imshow(crop_shock); axes[1].set_title("Shock Filter (PDE Edge Steepening)\nTurns Blurry Slopes into Sharp Steps", fontsize=11, fontweight='bold')
axes[2].imshow(crop_building_snap); axes[2].set_title("Morphological Building Boundary Engine\nCrisp Building Outlines & Straight Rooflines", fontsize=11, fontweight='bold', color='green')

for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('scratch/edge_steepening_test.png', dpi=200, bbox_inches='tight')
print("Saved scratch/edge_steepening_test.png successfully!")

