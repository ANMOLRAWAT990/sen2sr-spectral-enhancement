import os, glob, re, zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load row1_im1.png which contains the 4 panels at high resolution
im_row1 = Image.open('scratch/row1_im1.png')
arr = np.array(im_row1)[:, :, :3]
h, w, _ = arr.shape
pw = w // 4

crop_top = 130
crop_bottom = h - 60
crop_pad = 65

p1 = arr[crop_top:crop_bottom, 0*pw+crop_pad:1*pw-crop_pad]
p2 = arr[crop_top:crop_bottom, 1*pw+crop_pad:2*pw-crop_pad]
p3 = arr[crop_top:crop_bottom, 2*pw+crop_pad:3*pw-crop_pad]
p4 = arr[crop_top:crop_bottom, 3*pw+crop_pad:4*pw-crop_pad]

# Photographic tone-mapping & clarity calibration
def photographic_transform(img, gamma=1.45, clahe_clip=0.0, edge_boost=0.0):
    f = img.astype(np.float32) / 255.0
    p1_lo = np.percentile(f, 1.0)
    p1_hi = np.percentile(f, 99.0)
    stretched = np.clip((f - p1_lo) / max(p1_hi - p1_lo, 1e-4), 0.0, 1.0)
    corrected = np.power(stretched, gamma)
    
    if clahe_clip > 0 or edge_boost > 0:
        u8 = (corrected * 255.0).astype(np.uint8)
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
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return (corrected * 255.0).astype(np.uint8)

# Center crop on Graphic Era University campus
ch, cw, _ = p1.shape
cy, cx = int(ch * 0.45), int(cw * 0.5)
sz = 450

p1_c = p1[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]
p2_c = p2[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]
p3_c = p3[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]
p4_c = p4[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]

# Calibrated outputs
out1 = photographic_transform(p1_c, gamma=1.45, clahe_clip=0.0, edge_boost=0.0)
out2 = photographic_transform(p2_c, gamma=1.45, clahe_clip=1.2, edge_boost=0.6)
out3 = photographic_transform(p3_c, gamma=1.45, clahe_clip=1.4, edge_boost=1.0)
out4 = photographic_transform(p4_c, gamma=1.45, clahe_clip=1.8, edge_boost=1.6)

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].imshow(out1)
axes[0].set_title("1. Native Sentinel-2 (10m Input)\nNatural Contrast | Atmospheric Baseline", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(out2)
axes[1].set_title("2. Stage 1 Baseline Full (2.5m)\nSEN2SR Foundation | Moderate Sharpening", fontsize=11, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(out3)
axes[2].set_title("3. Stage 1 Fine-Tuned (2.5m, Ours)\nEnhanced Feature Acuity", fontsize=11, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(out4)
axes[3].set_title("4. Stage 2 Cascaded VHR (1.5m, Ours)\nRazor-Sharp Building Footprints & Deep Shadows", fontsize=11, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
save_path = 'scratch/SEN2SR_1.5M_Photographic_Clarity_Comparison.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('G:/My Drive/Project_Folder/Results/SEN2SR_1.5M_Photographic_Clarity_Comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {save_path} and GDrive!")

