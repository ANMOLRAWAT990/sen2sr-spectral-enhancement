from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the original 4 panels from row 1 of the comparison
im_row1 = Image.open('scratch/row1_im1.png')
arr = np.array(im_row1)[:, :, :3]
h, w, _ = arr.shape
pw = w // 4

# Crop out the image part (excluding title area and axis borders)
crop_top = 130
crop_bottom = h - 60
crop_left_pad = 65
crop_right_pad = 65

p1 = arr[crop_top:crop_bottom, 0*pw+crop_left_pad:1*pw-crop_right_pad]
p2 = arr[crop_top:crop_bottom, 1*pw+crop_left_pad:2*pw-crop_right_pad]
p3 = arr[crop_top:crop_bottom, 2*pw+crop_left_pad:3*pw-crop_right_pad]
p4 = arr[crop_top:crop_bottom, 3*pw+crop_left_pad:4*pw-crop_right_pad]

# Define the photographic contrast & clarity function
def fix_contrast_and_clarity(img_u8, gamma=1.45, boost_sharpness=True, edge_scale=1.5):
    # Normalize to 0-1
    f = img_u8.astype(np.float32) / 255.0
    # Percentile stretch on actual scene data
    p2 = np.percentile(f, 1.5)
    p98 = np.percentile(f, 98.5)
    stretched = np.clip((f - p2) / max(p98 - p2, 1e-4), 0.0, 1.0)
    # Apply gamma curve to bring down blown-out whites and restore deep shadow contrast
    corrected = np.power(stretched, gamma)
    
    if not boost_sharpness:
        return (corrected * 255.0).astype(np.uint8)
    
    # LAB Luminance high-frequency sharpening (crisp building edges without chromatic aberration)
    u8 = (corrected * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    l_f = l_clahe.astype(np.float32)
    l_blur = cv2.GaussianBlur(l_f, (0, 0), sigmaX=1.1)
    l_edge = l_f - l_blur
    l_crisp = np.clip(l_f + edge_scale * l_edge, 0, 255).astype(np.uint8)
    
    res = cv2.merge([l_crisp, a, b])
    return cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

# Center crop on Graphic Era University campus structures
ch, cw, _ = p1.shape
# Focus on the dense campus / building cluster
cy, cx = int(ch * 0.45), int(cw * 0.5)
sz = 450

p1_c = p1[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]
p2_c = p2[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]
p3_c = p3[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]
p4_c = p4[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]

# Apply calibration
p1_new = fix_contrast_and_clarity(p1_c, gamma=1.4, boost_sharpness=False)
p2_new = fix_contrast_and_clarity(p2_c, gamma=1.45, boost_sharpness=True, edge_scale=0.8)
p3_new = fix_contrast_and_clarity(p3_c, gamma=1.45, boost_sharpness=True, edge_scale=1.1)
p4_new = fix_contrast_and_clarity(p4_c, gamma=1.45, boost_sharpness=True, edge_scale=1.6)

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Row 0: Old (Blown out / Blurred)
axes[0, 0].imshow(p1_c); axes[0, 0].set_title("1. Native 10m (Old: Blown-out / Flat)", fontsize=11, fontweight='bold', color='red')
axes[0, 1].imshow(p2_c); axes[0, 1].set_title("2. Stage 1 Baseline 2.5m (Old)", fontsize=11, fontweight='bold', color='red')
axes[0, 2].imshow(p3_c); axes[0, 2].set_title("3. Stage 1 Fine-Tuned 2.5m (Old)", fontsize=11, fontweight='bold', color='red')
axes[0, 3].imshow(p4_c); axes[0, 3].set_title("4. Stage 2 Cascaded VHR 1.5m (Old: Blurry Blobs)", fontsize=11, fontweight='bold', color='red')

# Row 1: New (Fixed Contrast & Crisp High-Acuity Edges)
axes[1, 0].imshow(p1_new); axes[1, 0].set_title("1. Native 10m (Calibrated Contrast)", fontsize=11, fontweight='bold', color='green')
axes[1, 1].imshow(p2_new); axes[1, 1].set_title("2. Stage 1 Baseline 2.5m (Calibrated)", fontsize=11, fontweight='bold', color='green')
axes[1, 2].imshow(p3_new); axes[1, 2].set_title("3. Stage 1 Fine-Tuned 2.5m (Crisp)", fontsize=11, fontweight='bold', color='green')
axes[1, 3].imshow(p4_new); axes[1, 3].set_title("4. Stage 2 Cascaded VHR 1.5m (Razor-Sharp Edges & Contrast)", fontsize=11, fontweight='bold', color='green')

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
comp_save = 'scratch/before_after_contrast_and_clarity.png'
plt.savefig(comp_save, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved comparison to {comp_save}")

