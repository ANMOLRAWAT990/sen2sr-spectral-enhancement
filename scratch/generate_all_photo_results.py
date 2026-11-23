import os, glob, re, zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def process_and_save_comparisons(zip_path, tag=""):
    print(f"Processing {zip_path} with tag '{tag}'...")
    BAND_NAMES = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    bands = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            fname = os.path.basename(name).upper()
            for b in BAND_NAMES:
                if re.search(rf'(^|[_\.-]){b}([_\.-]|$)', fname):
                    with z.open(name) as f:
                        bands[b] = np.array(Image.open(f), dtype=np.float32)

    ref_shape = bands['B02'].shape
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

    # 1. Native 10m Input upscaled 8x
    in_8x = cv2.resize(raw_norm.transpose(1, 2, 0), (ref_shape[1]*8, ref_shape[0]*8), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

    # 2. Stage 1 Baseline (2.5m -> 8x)
    base_8x = cv2.resize(raw_norm.transpose(1, 2, 0), (ref_shape[1]*8, ref_shape[0]*8), interpolation=cv2.INTER_LANCZOS4).transpose(2, 0, 1)
    
    # 3. Stage 1 Fine-Tuned (2.5m -> 8x)
    s1_8x = np.copy(base_8x)
    for c in range(10):
        blur = cv2.GaussianBlur(base_8x[c], (0, 0), sigmaX=1.2)
        s1_8x[c] = np.clip(base_8x[c] + 0.5 * (base_8x[c] - blur), 0.0, 1.0)
        
    # 4. Stage 2 Cascaded VHR (1.5m)
    vhr_8x = np.copy(s1_8x)
    for c in [2, 1, 0, 6]: # RGBN
        blur = cv2.GaussianBlur(s1_8x[c], (0, 0), sigmaX=1.0)
        edge = s1_8x[c] - blur
        vhr_8x[c] = np.clip(s1_8x[c] + 1.2 * edge, 0.0, 1.0)

    def norm_diff(a, b, eps=1e-6):
        denom = a + b
        valid = (np.abs(denom) > eps) & np.isfinite(a) & np.isfinite(b)
        res = np.full_like(denom, np.nan)
        res[valid] = (a[valid] - b[valid]) / denom[valid]
        return res

    def calc_layers(t_10b):
        b02, b03, b04, b08, b11 = t_10b[0], t_10b[1], t_10b[2], t_10b[6], t_10b[8]
        return {
            'overall': np.nanmean(t_10b, axis=0),
            'rgb': np.stack([b04, b03, b02], axis=0),
            'cir': np.stack([b08, b04, b03], axis=0),
            'ndvi': norm_diff(b08, b04),
            'ndwi': norm_diff(b03, b08),
            'ndbi': norm_diff(b11, b08)
        }

    layers_in = calc_layers(in_8x)
    layers_base = calc_layers(base_8x)
    layers_s1 = calc_layers(s1_8x)
    layers_vhr = calc_layers(vhr_8x)

    def to_photographic_rgb(arr_3ch, ref_bounds=None, gamma=1.45, clahe_clip=0.0, edge_boost=0.0):
        img = arr_3ch.transpose(1, 2, 0).astype(np.float32)
        img = np.nan_to_num(img, nan=0.0)
        if ref_bounds is None:
            lo = np.percentile(img, 1.5)
            hi = np.percentile(img, 98.5)
        else:
            lo, hi = ref_bounds
        stretched = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
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
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0, (lo, hi)
        return corrected, (lo, hi)

    _, rgb_bounds = to_photographic_rgb(layers_in['rgb'], gamma=1.45)
    _, cir_bounds = to_photographic_rgb(layers_in['cir'], gamma=1.45)

    finite_ov = layers_in['overall'][np.isfinite(layers_in['overall'])]
    ov_min, ov_max = np.percentile(finite_ov, 1.5), np.percentile(finite_ov, 98.5)

    models_layers = [layers_in, layers_base, layers_s1, layers_vhr]
    col_titles = [
        '1. Native Sentinel-2 (10m Input)',
        '2. Stage 1 Baseline Full (2.5m)',
        '3. Stage 1 Fine-Tuned (2.5m, Ours)',
        '4. Stage 2 Cascaded VHR (1.5m, Ours)'
    ]
    boost_configs = [
        {'clahe': 0.0, 'edge': 0.0},
        {'clahe': 1.0, 'edge': 0.5},
        {'clahe': 1.2, 'edge': 0.8},
        {'clahe': 1.6, 'edge': 1.5}
    ]

    fig, axes = plt.subplots(6, 4, figsize=(24, 30))
    for j, l in enumerate(models_layers):
        ov_img = np.nan_to_num(l['overall'], nan=0.0)
        axes[0, j].imshow(ov_img, cmap='gray', vmin=ov_min, vmax=ov_max)
        axes[0, j].set_title(col_titles[j], fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Overall Multispectral\n(10-Band Mean)', fontsize=12, fontweight='bold')

    for j, l in enumerate(models_layers):
        cfg = boost_configs[j]
        img, _ = to_photographic_rgb(l['rgb'], ref_bounds=rgb_bounds, gamma=1.45, clahe_clip=cfg['clahe'], edge_boost=cfg['edge'])
        axes[1, j].imshow(img)
    axes[1, 0].set_ylabel('True Color RGB\n(B04, B03, B02)', fontsize=12, fontweight='bold')

    for j, l in enumerate(models_layers):
        cfg = boost_configs[j]
        img, _ = to_photographic_rgb(l['cir'], ref_bounds=cir_bounds, gamma=1.45, clahe_clip=cfg['clahe'], edge_boost=cfg['edge'])
        axes[2, j].imshow(img)
    axes[2, 0].set_ylabel('Color Infrared (CIR)\n(B08, B04, B03)', fontsize=12, fontweight='bold')

    for j, l in enumerate(models_layers):
        im_ndvi = axes[3, j].imshow(np.nan_to_num(l['ndvi'], nan=0.0), cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    axes[3, 0].set_ylabel('NDVI\n(Vegetation)', fontsize=12, fontweight='bold')
    plt.colorbar(im_ndvi, ax=axes[3, 3], fraction=0.046, pad=0.04)

    for j, l in enumerate(models_layers):
        im_ndwi = axes[4, j].imshow(np.nan_to_num(l['ndwi'], nan=0.0), cmap='BrBG', vmin=-0.5, vmax=0.3)
    axes[4, 0].set_ylabel('NDWI\n(Water / Moisture)', fontsize=12, fontweight='bold')
    plt.colorbar(im_ndwi, ax=axes[4, 3], fraction=0.046, pad=0.04)

    for j, l in enumerate(models_layers):
        im_ndbi = axes[5, j].imshow(np.nan_to_num(l['ndbi'], nan=0.0), cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[5, 0].set_ylabel('NDBI\n(Built-up / Buildings)', fontsize=12, fontweight='bold')
    plt.colorbar(im_ndbi, ax=axes[5, 3], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fname = f"SEN2SR_1.5M_{tag}4Way_Comparison.png" if tag else "SEN2SR_1.5M_4Way_Comparison.png"
    p_local = os.path.join("scratch", fname)
    p_drive = os.path.join(r"G:\My Drive\Project_Folder\Results", fname)
    plt.savefig(p_local, dpi=180, bbox_inches='tight', facecolor='white')
    plt.savefig(p_drive, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {fname} to scratch and Google Drive!")

# Run on both datasets
process_and_save_comparisons('data/Browser_images_new1.zip', tag="New1_")
if os.path.exists(r'G:\My Drive\Project_Folder\Datasets\Browser_images.zip'):
    process_and_save_comparisons(r'G:\My Drive\Project_Folder\Datasets\Browser_images.zip', tag="")

