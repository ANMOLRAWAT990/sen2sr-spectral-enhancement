import tifffile, numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

tif_path = 'G:/My Drive/Project_Folder/Results/sen2sr_cascaded_1.5m_vhr.tif'
data = tifffile.imread(tif_path)

# Extract True Color RGB (Red: 2, Green: 1, Blue: 0)
rgb = data[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)

# Current Display Scaling (percentile 2-98)
def stretch(img, p=(2, 98)):
    lo, hi = np.percentile(img[np.isfinite(img)], p)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

rgb_current = stretch(rgb)

# 1. Unsharp Masking / High-Frequency MTF Restoration:
# High-resolution commercial sensors (Maxar, Planet) use MTF deconvolution:
# Detail = Image - GaussianBlur(Image, sigma)
# Sharpened = Image + amount * Detail
blurred = np.stack([gaussian_filter(rgb_current[:, :, c], sigma=1.2) for c in range(3)], axis=-1)
detail = rgb_current - blurred
rgb_sharp = np.clip(rgb_current + 1.8 * detail, 0.0, 1.0)

# 2. Local Dynamic Range & Edge Boost (matching Google Earth contrast):
# Google Earth has high local contrast between roofs, shadows, and roads
# Using unsharp detail + contrast gamma curve
gamma = 0.85  # slightly brings out shadow structures
rgb_enhanced = np.clip(rgb_sharp ** gamma, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(rgb_current)
axes[0].set_title('1. Current Model Output (1.5m VHR)\nSoft transitions / Linear stretch', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(rgb_sharp)
axes[1].set_title('2. With MTF High-Frequency Restoration\nSharp building edges & street corridors', fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(rgb_enhanced)
axes[2].set_title('3. Google Earth-Style Clarity\nCrisp roof geometry & enhanced contrast', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('scratch/geu_clarity_comparison.png', dpi=200)
print('Saved GEU clarity comparison to scratch/geu_clarity_comparison.png!')

