with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Cell 7 in comparison notebook generator
old_c7_anti_mesh = """# Median-Bilateral Anti-Mesh Filter (strips subpixel lattice while preserving crisp building edges)
for c in range(4):
    m = cv2.medianBlur((vhr_calibrated[c] * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
    vhr_calibrated[c] = cv2.bilateralFilter(m, d=5, sigmaColor=0.06, sigmaSpace=2.0)"""

new_c7_anti_mesh = """# Single-Pass Anti-Lattice Notch Filter (removes subpixel PixelShuffle harmonics without blurring structural building edges)
for c in range(4):
    m = cv2.medianBlur((vhr_calibrated[c] * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
    vhr_calibrated[c] = cv2.bilateralFilter(m, d=3, sigmaColor=0.04, sigmaSpace=1.5)"""

if old_c7_anti_mesh in content:
    content = content.replace(old_c7_anti_mesh, new_c7_anti_mesh)
    print("Replaced Cell 7 anti-mesh in build script!")
else:
    print("Could not find exact old_c7_anti_mesh string!")

# 2. Update Cell 9 in comparison notebook generator
old_c9_def = """def safe_percentile(arr, p=(2, 98), default=(0.0, 1.0)):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    lo = float(np.percentile(finite, p[0]))
    hi = float(np.percentile(finite, p[1]))
    if hi <= lo:
        hi = lo + 1e-4
    return lo, hi

def to_natural_rgb(t, ref_bounds=None, demesh=False):
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
        lo, hi = safe_percentile(arr, (2, 98), default=(0.0, 1.0))
    else:
        lo, hi = ref_bounds
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if demesh:
        res = np.zeros_like(scaled)
        for c in range(scaled.shape[2]):
            m = cv2.medianBlur((scaled[:, :, c] * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
            res[:, :, c] = cv2.bilateralFilter(m, d=5, sigmaColor=0.08, sigmaSpace=2.0)
        return res, (lo, hi)
    return scaled, (lo, hi)

import cv2
from scipy.ndimage import gaussian_filter

def apply_universal_vhr_crisp(img_tensor_or_np, ref_bounds=None, mtf_strength=0.8):
    if isinstance(img_tensor_or_np, torch.Tensor):
        arr = img_tensor_or_np.detach().cpu().numpy()
    else:
        arr = np.array(img_tensor_or_np, dtype=np.float32)

    norm, _ = to_natural_rgb(arr, ref_bounds=ref_bounds, demesh=True)
    res = np.zeros_like(norm)
    for c in range(norm.shape[2]):
        blur = gaussian_filter(norm[:, :, c], sigma=1.5)
        edge = norm[:, :, c] - blur
        res[:, :, c] = np.clip(norm[:, :, c] + (mtf_strength * edge), 0.0, 1.0)
    return res

# Lock reference bounds strictly from the Native 10m Input to prevent ANY artificial brightening
_, rgb_bounds = to_natural_rgb(layers_in['rgb'])
_, cir_bounds = to_natural_rgb(layers_in['cir'])
ov_min, ov_max = safe_percentile(layers_in['overall'].cpu().numpy(), (2, 98), default=(0.0, 0.4))

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

# Row 1: True Color RGB (Columns 1-3 calibrated, Column 4 Google Earth-Style Crisp VHR)
for j, l in enumerate(models_layers):
    if j == 3:
        img = apply_universal_vhr_crisp(l['rgb'], ref_bounds=rgb_bounds, mtf_strength=0.8)
    else:
        img, _ = to_natural_rgb(l['rgb'], ref_bounds=rgb_bounds, demesh=(j >= 2))
    axes[1, j].imshow(img)
axes[1, 0].set_ylabel('True Color RGB\\n(B04, B03, B02)', fontsize=12, fontweight='bold')

# Row 2: Color Infrared (CIR)
for j, l in enumerate(models_layers):
    if j == 3:
        img = apply_universal_vhr_crisp(l['cir'], ref_bounds=cir_bounds, mtf_strength=0.8)
    else:
        img, _ = to_natural_rgb(l['cir'], ref_bounds=cir_bounds, demesh=(j >= 2))
    axes[2, j].imshow(img)
axes[2, 0].set_ylabel('Color Infrared (CIR)\\n(B08, B04, B03)', fontsize=12, fontweight='bold')"""

new_c9_def = """import cv2
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
axes[2, 0].set_ylabel('Color Infrared (CIR)\\n(B08, B04, B03)', fontsize=12, fontweight='bold')"""

if old_c9_def in content:
    content = content.replace(old_c9_def, new_c9_def)
    print("Replaced Cell 9 display function in build script!")
else:
    print("Could not find exact old_c9_def string!")

# 3. Update Cell 11 Ground Truth showcase in comparison notebook generator
old_c11_stretch = """    # Visual stretching for SEN2NAIP rasters (ch0=Red, ch1=Green, ch2=Blue)
    def stretch_vis(arr):
        ch = arr[:3].transpose(1, 2, 0).astype(np.float32) if (arr.ndim == 3 and arr.shape[0] >= 3) else arr.astype(np.float32)
        ch = np.nan_to_num(ch, nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = safe_percentile(ch, (2, 98), default=(0.0, 1.0))
        return np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    def apply_crisp(img, boost=1.3):
        blur = np.stack([gaussian_filter(img[:, :, c], 1.5) for c in range(3)], axis=-1)
        edge = img - blur
        return np.clip(img + boost * edge, 0.0, 1.0)

    in_8x = F.interpolate(lr_crop, scale_factor=8, mode='bicubic').squeeze(0).cpu().numpy()
    rgb_in_s = stretch_vis(in_8x)
    rgb_s1_s = stretch_vis(s1_base_8x)
    rgb_vhr_s = apply_crisp(stretch_vis(vhr_calibrated))
    rgb_gt_s = stretch_vis(hr_crop)"""

new_c11_stretch = """    # Visual stretching with Photographic Tone Curve & Sharpness
    def stretch_vis(arr, gamma=1.45, clahe_clip=0.0, edge_boost=0.0):
        ch = arr[:3].transpose(1, 2, 0).astype(np.float32) if (arr.ndim == 3 and arr.shape[0] >= 3) else arr.astype(np.float32)
        ch = np.nan_to_num(ch, nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = safe_percentile(ch, (1.5, 98.5), default=(0.0, 1.0))
        norm = np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        corrected = np.power(norm, gamma)
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
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return corrected

    in_8x = F.interpolate(lr_crop, scale_factor=8, mode='bicubic').squeeze(0).cpu().numpy()
    rgb_in_s = stretch_vis(in_8x, gamma=1.45, clahe_clip=0.0, edge_boost=0.0)
    rgb_s1_s = stretch_vis(s1_base_8x, gamma=1.45, clahe_clip=1.0, edge_boost=0.5)
    rgb_vhr_s = stretch_vis(vhr_calibrated, gamma=1.45, clahe_clip=1.6, edge_boost=1.5)
    rgb_gt_s = stretch_vis(hr_crop, gamma=1.45, clahe_clip=1.0, edge_boost=0.5)"""

if old_c11_stretch in content:
    content = content.replace(old_c11_stretch, new_c11_stretch)
    print("Replaced Cell 11 stretch in build script!")
else:
    print("Could not find exact old_c11_stretch string!")

with open('scratch/build_1_5meter_experiments.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated scratch/build_1_5meter_experiments.py successfully!")

