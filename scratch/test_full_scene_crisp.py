import tifffile, numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the super-resolved 10-band raster
tif_path = 'G:/My Drive/Project_Folder/Results/sen2sr_cascaded_1.5m_vhr.tif'
data = tifffile.imread(tif_path)

BAND_NAMES = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
BAND_MAP = {name: i for i, name in enumerate(BAND_NAMES)}

b02 = data[BAND_MAP['B02']]
b03 = data[BAND_MAP['B03']]
b04 = data[BAND_MAP['B04']]
b08 = data[BAND_MAP['B08']]
b11 = data[BAND_MAP['B11']]

def apply_universal_crisp(img_chw, mtf=2.0, gamma=0.88):
    # Normalize to [0, 1] per channel
    res = np.zeros_like(img_chw, dtype=np.float32)
    for c in range(img_chw.shape[0]):
        ch = img_chw[c]
        p2, p98 = np.percentile(ch[np.isfinite(ch)], (2, 98))
        norm = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
        
        # MTF optical restoration
        b_fine = gaussian_filter(norm, sigma=0.8)
        b_coarse = gaussian_filter(norm, sigma=2.0)
        hf_fine = norm - b_fine
        hf_coarse = b_fine - b_coarse
        
        sharp = np.clip(norm + (mtf * hf_fine) + (0.6 * hf_coarse), 0.0, 1.0)
        # S-curve perceptual contrast
        scurve = 3.0 * (sharp ** 2) - 2.0 * (sharp ** 3)
        res[c] = np.clip(scurve ** gamma, 0.0, 1.0)
    return res

rgb_raw = np.stack([b04, b03, b02], axis=0)
cir_raw = np.stack([b08, b04, b03], axis=0)

rgb_crisp = apply_universal_crisp(rgb_raw, mtf=2.2).transpose(1, 2, 0)
cir_crisp = apply_universal_crisp(cir_raw, mtf=2.2).transpose(1, 2, 0)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(rgb_crisp)
axes[0].set_title('True Color RGB (1.5m Google Earth-Style Crisp VHR)', fontsize=13, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cir_crisp)
axes[1].set_title('Color Infrared CIR (1.5m Google Earth-Style Crisp VHR)', fontsize=13, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('scratch/full_scene_crisp_preview.png', dpi=200)
print('Saved full scene crisp preview to scratch/full_scene_crisp_preview.png!')

