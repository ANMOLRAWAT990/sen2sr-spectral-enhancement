import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

H, W = 540, 944
overall_fine = np.random.rand(H, W).astype(np.float32)
overall_base = overall_fine + np.random.normal(0, 0.01, (H, W)).astype(np.float32)
overall_input = overall_base + np.random.normal(0, 0.02, (H, W)).astype(np.float32)

ndvi_input = overall_input.copy()
ndvi_base = overall_base.copy()
ndvi_fine = overall_fine.copy()

ndwi_input = overall_input.copy()
ndwi_base = overall_base.copy()
ndwi_fine = overall_fine.copy()

ndbi_input = overall_input.copy()
ndbi_base = overall_base.copy()
ndbi_fine = overall_fine.copy()

OUTPUT_DIR = "scratch"

# Test plotting
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8])

r1_y0, r1_y1 = int(H * 0.35), int(H * 0.70)
r1_x0, r1_x1 = int(W * 0.35), int(W * 0.65)
r2_y0, r2_y1 = int(H * 0.25), int(H * 0.60)
r2_x0, r2_x1 = int(W * 0.05), int(W * 0.35)

rois = [
    ("ROI 1: Water-Land Riverbank Delineation", r1_y0, r1_y1, r1_x0, r1_x1, ndwi_input, ndwi_base, ndwi_fine, 'BrBG', -1, 1),
    ("ROI 2: Agricultural & Vegetation Canopy Acuity", r2_y0, r2_y1, r2_x0, r2_x1, ndvi_input, ndvi_base, ndvi_fine, 'RdYlGn', -1, 1)
]

for row_idx, (roi_name, y0, y1, x0, x1, a_in, a_b, a_f, cmap, vmin, vmax) in enumerate(rois):
    crop_in = a_in[y0:y1, x0:x1]
    crop_base = a_b[y0:y1, x0:x1]
    crop_fine = a_f[y0:y1, x0:x1]
    crop_diff = crop_fine - crop_base
    
    ax0 = fig.add_subplot(gs[row_idx, 0])
    ax0.imshow(crop_in, cmap=cmap, vmin=vmin, vmax=vmax)
    ax0.set_title(f"{roi_name}\n1. Normal 10m Input", fontsize=10, fontweight='bold')
    ax0.axis('off')
    
    ax1 = fig.add_subplot(gs[row_idx, 1])
    ax1.imshow(crop_base, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"2. Pretrained Baseline (2.5m)", fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[row_idx, 2])
    ax2.imshow(crop_fine, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f"3. Fine-Tuned (2.5m, Ours)", fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[row_idx, 3])
    lim = max(float(np.percentile(np.abs(crop_diff[np.isfinite(crop_diff)]), 98)), 0.02)
    im_d = ax3.imshow(crop_diff, cmap='coolwarm', vmin=-lim, vmax=lim)
    ax3.set_title(f"Improvement Spotlight (Ours − Base)\n±{lim:.3f} Acuity Delta", fontsize=10, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im_d, ax=ax3, fraction=0.046, pad=0.04)

ax_transect = fig.add_subplot(gs[2, :])
transect_y = int(H * 0.52)
transect_in = ndwi_input[transect_y, :]
transect_base = ndwi_base[transect_y, :]
transect_fine = ndwi_fine[transect_y, :]
x_coords = np.arange(W)

ax_transect.plot(x_coords, transect_in, label='1. Normal Captured Input (10m Upscaled)', color='#9E9E9E', linestyle='--', linewidth=1.8)
ax_transect.plot(x_coords, transect_base, label='2. Pretrained Baseline SEN2SR (2.5m)', color='#1976D2', linewidth=1.5, alpha=0.85)
ax_transect.plot(x_coords, transect_fine, label='3. Fine-Tuned SEN2SR (2.5m, Compound Loss, Ours)', color='#D32F2F', linewidth=2.2)
ax_transect.set_title(f"1D Spatial Transect Profile across River-Land Boundary (Y = {transect_y} px)\nSteeper slope proves crisper boundary transition without ringing", fontsize=11, fontweight='bold')
ax_transect.set_xlabel("Horizontal Pixel Coordinate (X)", fontsize=10)
ax_transect.set_ylabel("NDWI Index Value", fontsize=10)
ax_transect.legend(loc='lower right', frameon=True, fontsize=10)
ax_transect.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("scratch/test_out.png", dpi=150)
plt.close()
print("FULL TEST COMPLETED WITH SUCCESS!")
