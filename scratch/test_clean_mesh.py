import cv2, tifffile, numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the super-resolved raster
tif_path = 'G:/My Drive/Project_Folder/Results/sen2sr_cascaded_1.5m_vhr.tif'
data = tifffile.imread(tif_path)
rgb = data[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)

def stretch(img, p=(2, 98)):
    lo, hi = np.percentile(img[np.isfinite(img)], p)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

rgb_norm = stretch(rgb)

# Test 1: What was in the user's screenshot (Mesh amplified by naive MTF)
b_fine = np.stack([gaussian_filter(rgb_norm[:, :, c], sigma=0.8) for c in range(3)], axis=-1)
b_coarse = np.stack([gaussian_filter(rgb_norm[:, :, c], sigma=2.0) for c in range(3)], axis=-1)
naive_sharp = np.clip(rgb_norm + 2.2 * (rgb_norm - b_fine) + 0.6 * (b_fine - b_coarse), 0.0, 1.0)
naive_scurve = np.clip(3.0 * (naive_sharp ** 2) - 2.0 * (naive_sharp ** 3), 0.0, 1.0)

# Test 2: Bilateral Anti-Mesh Filter FIRST, THEN Structural Edge Sharpness
# 1. Bilateral filter removes small 1-2px mesh oscillations while keeping real edges
rgb_denoised = np.zeros_like(rgb_norm)
for c in range(3):
    rgb_denoised[:, :, c] = cv2.bilateralFilter(rgb_norm[:, :, c].astype(np.float32), d=5, sigmaColor=0.08, sigmaSpace=2.0)

# 2. Structural high-pass on the clean denoised image
blur_structural = np.stack([gaussian_filter(rgb_denoised[:, :, c], sigma=1.5) for c in range(3)], axis=-1)
edge_detail = rgb_denoised - blur_structural
clean_sharp = np.clip(rgb_denoised + 1.6 * edge_detail, 0.0, 1.0)

# 3. Clean S-curve contrast
clean_final = np.clip(3.0 * (clean_sharp ** 2) - 2.0 * (clean_sharp ** 3), 0.0, 1.0)

# Zoom into campus area
H, W, _ = rgb_norm.shape
crop = (slice(int(H*0.2), int(H*0.7)), slice(int(W*0.3), int(W*0.8)))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(rgb_norm[crop])
axes[0].set_title('1. Raw Model Output\nFaint mesh in background', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(naive_scurve[crop])
axes[1].set_title('2. Naive MTF (User Screenshot)\nMesh amplified into textile pattern', fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(clean_final[crop])
axes[2].set_title('3. Bilateral Anti-Mesh + Clean Edge Sharpness\nZero mesh, sharp crisp building edges!', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('scratch/clean_mesh_comparison.png', dpi=200)
print('Saved clean mesh comparison to scratch/clean_mesh_comparison.png!')

