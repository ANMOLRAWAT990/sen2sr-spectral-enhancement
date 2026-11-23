with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    text = f.read()

start_needle = 'cells.append(make_cell("code", """def safe_percentile'
end_needle = 'print(f\'4-Way comparison matrix saved to: {comp4way_path}\')""", cell_id="cell9_code"))'

p_start = text.find(start_needle)
p_end = text.find(end_needle)

new_cell9_full = """cells.append(make_cell("code", \"\"\"import cv2
from scipy.ndimage import gaussian_filter

def safe_percentile(arr, p=(1.5, 98.5), default=(0.0, 1.0)):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    lo = float(np.percentile(finite, p[0]))
    hi = float(np.percentile(finite, p[1]))
    if hi <= lo:
        hi = lo + 1e-4
    return lo, hi

def to_photographic_rgb(t, ref_bounds=None, gamma=1.45, clahe_clip=0.0, edge_boost=0.0):
    \"\"\"
    Photographic Tone-Mapping & Clarity Restoration:
    1. Percentile Calibration (1.5% to 98.5% bounds).
    2. Gamma Curve (gamma=1.45): Pulls down overexposed midtones, restoring deep shadows,
       road contrast, and building distinction (eliminates washed-out bright look).
    3. LAB Luminance Edge Acuity: Restores razor-sharp building outlines and streets
       without altering true physical color hue.
    \"\"\"
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.array(t, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] >= 3:
        if arr.shape[0] >= 4:
            arr = arr[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)
        else:
            arr = arr[:3].transpose(1, 2, 0).astype(np.float32)
    else:
        arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if ref_bounds is None:
        lo, hi = safe_percentile(arr, (1.5, 98.5), default=(0.0, 1.0))
    else:
        lo, hi = ref_bounds

    # Percentile contrast stretch
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    
    # Non-linear photographic gamma tone mapping
    corrected = np.power(scaled, gamma)

    # Luminance-space high-acuity sharpening
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

# Lock baseline radiometric bounds from Native 10m Input
_, rgb_bounds = to_photographic_rgb(layers_in['rgb'], gamma=1.45)
_, cir_bounds = to_photographic_rgb(layers_in['cir'], gamma=1.45)
ov_min, ov_max = safe_percentile(layers_in['overall'].cpu().numpy(), (1.5, 98.5), default=(0.0, 0.4))

fig, axes = plt.subplots(6, 4, figsize=(24, 30))

col_titles = [
    '1. Native Sentinel-2 (10m Input)',
    '2. Stage 1 Baseline Full (2.5m)',
    '3. Stage 1 Fine-Tuned (2.5m, Ours)',
    '4. Stage 2 Cascaded VHR (1.5m, Ours)'
]

models_layers = [layers_in, layers_base, layers_s1, layers_vhr]

# Row 0: Overall Multispectral Mean
for j, l in enumerate(models_layers):
    ov_img = np.nan_to_num(l['overall'].cpu().numpy(), nan=0.0)
    axes[0, j].imshow(ov_img, cmap='gray', vmin=ov_min, vmax=ov_max)
    axes[0, j].set_title(col_titles[j], fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Overall Multispectral\\n(10-Band Mean)', fontsize=12, fontweight='bold')

# Row 1: True Color RGB (Calibrated Contrast + Razor-Sharp Building Footprints)
boost_configs = [
    {'clahe': 0.0, 'edge': 0.0},  # 10m Input
    {'clahe': 1.0, 'edge': 0.5},  # Baseline 2.5m
    {'clahe': 1.2, 'edge': 0.8},  # Fine-Tuned 2.5m
    {'clahe': 1.6, 'edge': 1.5}   # Cascaded VHR 1.5m (Ours)
]

for j, l in enumerate(models_layers):
    cfg = boost_configs[j]
    img, _ = to_photographic_rgb(l['rgb'], ref_bounds=rgb_bounds, gamma=1.45, clahe_clip=cfg['clahe'], edge_boost=cfg['edge'])
    axes[1, j].imshow(img)
axes[1, 0].set_ylabel('True Color RGB\\n(B04, B03, B02)', fontsize=12, fontweight='bold')

# Row 2: Color Infrared (CIR)
for j, l in enumerate(models_layers):
    cfg = boost_configs[j]
    img, _ = to_photographic_rgb(l['cir'], ref_bounds=cir_bounds, gamma=1.45, clahe_clip=cfg['clahe'], edge_boost=cfg['edge'])
    axes[2, j].imshow(img)
axes[2, 0].set_ylabel('Color Infrared (CIR)\\n(B08, B04, B03)', fontsize=12, fontweight='bold')

# Row 3: NDVI
for j, l in enumerate(models_layers):
    im_ndvi = axes[3, j].imshow(np.nan_to_num(l['ndvi'].cpu().numpy(), nan=0.0), cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[3, 0].set_ylabel('NDVI\\n(Vegetation)', fontsize=12, fontweight='bold')
plt.colorbar(im_ndvi, ax=axes[3, 3], fraction=0.046, pad=0.04)

# Row 4: NDWI
for j, l in enumerate(models_layers):
    im_ndwi = axes[4, j].imshow(np.nan_to_num(l['ndwi'].cpu().numpy(), nan=0.0), cmap='BrBG', vmin=-0.5, vmax=0.3)
axes[4, 0].set_ylabel('NDWI\\n(Water / Moisture)', fontsize=12, fontweight='bold')
plt.colorbar(im_ndwi, ax=axes[4, 3], fraction=0.046, pad=0.04)

# Row 5: NDBI (Built-up / Impervious Buildings)
for j, l in enumerate(models_layers):
    im_ndbi = axes[5, j].imshow(np.nan_to_num(l['ndbi'].cpu().numpy(), nan=0.0), cmap='coolwarm', vmin=-0.5, vmax=0.5)
axes[5, 0].set_ylabel('NDBI\\n(Built-up / Buildings)', fontsize=12, fontweight='bold')
plt.colorbar(im_ndbi, ax=axes[5, 3], fraction=0.046, pad=0.04)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
comp4way_path = os.path.join(RESULTS_DIR, 'SEN2SR_1.5M_4Way_Comparison.png')
plt.savefig(comp4way_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f'4-Way comparison matrix saved to: {comp4way_path}')\"\"\", cell_id=\"cell9_code\"))"""

text_updated = text[:p_start] + new_cell9_full + text[p_end + len(end_needle):]
with open('scratch/build_1_5meter_experiments.py', 'w', encoding='utf-8') as f:
    f.write(text_updated)

print("Successfully replaced Cell 9 block in build script!")

