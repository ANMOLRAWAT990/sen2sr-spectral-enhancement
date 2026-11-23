"""
Generator for Tasks T1, T2, T3:
- Train_SEN2SR_Lite_Compound_Loss.ipynb
- Train_SEN2SR_Full_Compound_Loss.ipynb
"""

import json, os, ast

def make_cell(cell_type, source, cell_id=None):
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }
    if cell_id:
        c["id"] = cell_id
    return c

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

# ==============================================================================
# Shared T1 Markdown & Code
# ==============================================================================
T1_MD = """## Step 3 — Audit & Verification: Band Order & Physical Normalization Contract

### 1. Band Order Decision: Standardized Canonical [Red, Green, Blue, NIR] (`[R, G, B, N]`)
**Evidence and Sources of Truth:**
- **Pretrained Lite Model Contract:** Official Hugging Face metadata (`tacofoundation/SEN2SR` `SEN2SRLite/main/mlm.json`) explicitly defines input bands as `["B04", "B03", "B02", "B08"]`, which maps directly to `[Red, Green, Blue, NIR]`.
- **SEN2NAIP Dataset:** Empirical inspection of raw `sentinaiplow` (`*_lr.tif`) and `sentinaiphigh` (`*_hr.tif`) confirms channel 0 is Red (mean ~958 / 128), channel 1 is Green (mean ~765 / 122), channel 2 is Blue (mean ~496 / 100), and channel 3 is NIR (vegetation peak mean ~1878 / 134).
- **WorldStrat Dataset:** High-resolution file `*_rgbn.tiff` natively encodes `[Red, Green, Blue, NIR]`. For the 12-band Sentinel-2 L2A cube (`B01` through `B12`), `B04` is channel 3, `B03` is channel 2, `B02` is channel 1, and `B08` is channel 7. Hence `WS_LR_ORDER = [3, 2, 1, 7]` yields `[R, G, B, N]`.
- **Pretrained Full Model Contract (`MambaSR` / `Swin2SR`):** Official metadata defines 10 bands as `["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]`. When extracting RGBN core channels from the 10-band stack, Red is index 2 (`B04`), Green is index 1 (`B03`), Blue is index 0 (`B02`), and NIR is index 6 (`B08`).
- **Elimination of Channel Inversion:** Standardizing canonical order to `[R, G, B, N]` ensures True Color RGB is simply `img[[0, 1, 2]]` (no red/blue channel swap), NDVI is `(img[3] - img[0]) / (img[3] + img[0])`, and NDWI is `(img[1] - img[3]) / (img[1] + img[3])`.

### 2. Radiometric Normalization Contract (Fixed Physical Scaling, No Max-Based Guessing)
- **Sentinel-2 L2A Reflectance:** Scaled by fixed factor `10000.0` (reflectance range `[0.0, 1.0]`).
- **NAIP Aerial Photography:** Scaled by fixed factor `255.0` (8-bit sensor DN range `[0.0, 1.0]`).
- **WorldStrat Pléiades/SPOT HR:** Scaled by fixed factor `4095.0` (12-bit sensor DN range `[0.0, 1.0]`).
- **Per-Patch Radiometric Alignment:** Target HR patches are quantile-matched to LR patch radiometry. This prevents the super-resolution network from overfitting to camera gain/exposure differences and forces it to learn structural edge geometry."""

T1_CODE = """import os, glob, warnings
import numpy as np
import rasterio

# Canonical Band Ordering: [Red, Green, Blue, NIR] = [B04, B03, B02, B08]
SEN2NAIP_LR_ORDER = [0, 1, 2, 3]  # Native SEN2NAIP LR is [Red, Green, Blue, NIR]
SEN2NAIP_HR_ORDER = [0, 1, 2, 3]  # Native NAIP HR is [Red, Green, Blue, NIR]
WS_LR_ORDER       = [3, 2, 1, 7]  # WorldStrat 12-band: B04(3), B03(2), B02(1), B08(7) -> [R, G, B, N]
WS_HR_ORDER       = [0, 1, 2, 3]  # WorldStrat HR (*_rgbn.tiff) native is [R, G, B, N]

FULL_10B_TO_RGBN  = [2, 1, 0, 6]  # 10-band S2 [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12] -> [B04, B03, B02, B08]
RGBN_TO_FULL_BGRN = [2, 1, 0, 3]  # [R, G, B, N] -> [B, G, R, N]

# Fixed physical scale factors (NO max-based guessing)
SCALE_S2 = 10000.0    # Sentinel-2 L2A integer reflectance DN
SCALE_NAIP = 255.0    # NAIP 8-bit digital numbers
SCALE_WS_HR = 4095.0  # WorldStrat Pléiades/SPOT 12-bit digital numbers

def read_raster_audit(path):
    with rasterio.open(path) as src:
        arr = src.read() # (C, H, W)
        meta = {
            'driver': src.driver,
            'count': src.count,
            'dtype': src.dtypes[0],
            'shape': (src.height, src.width),
            'crs': str(src.crs),
            'descriptions': src.descriptions
        }
        return arr.astype(np.float32), meta

def match_radiometry_patch(hr_patch, lr_patch):
    \"\"\"
    Quantile-matches HR patch radiometry to LR patch radiometry per channel.
    Eliminates exposure/DN offsets between airborne camera and satellite reflectance.
    \"\"\"
    matched = np.zeros_like(hr_patch)
    for c in range(hr_patch.shape[0]):
        src = hr_patch[c].ravel()
        tpl = lr_patch[c].ravel()
        s_quantiles = (np.argsort(np.argsort(src)) + 0.5) / len(src)
        matched[c] = np.quantile(tpl, s_quantiles).reshape(hr_patch[c].shape)
    return np.clip(matched, 0.0, 1.0)

def align_hr_to_4x(lr, hr):
    \"\"\"
    Ensures HR array has shape (4, lr_h * 4, lr_w * 4) to eliminate spatial grid discrepancies.
    \"\"\"
    expected_h = lr.shape[1] * 4
    expected_w = lr.shape[2] * 4
    if hr.shape[1] != expected_h or hr.shape[2] != expected_w:
        import torch
        import torch.nn.functional as F
        hr_t = torch.from_numpy(hr).unsqueeze(0).float()
        hr = F.interpolate(hr_t, size=(expected_h, expected_w), mode='bilinear', align_corners=False).squeeze(0).numpy()
    return hr

def load_canonical_pair(lr_path, hr_path, is_worldstrat=False):
    lr_raw, lr_meta = read_raster_audit(lr_path)
    hr_raw, hr_meta = read_raster_audit(hr_path)

    if is_worldstrat:
        lr_4b = lr_raw[WS_LR_ORDER]
        hr_4b = hr_raw[WS_HR_ORDER]
        lr_scaled = np.clip(lr_4b, 0.0, 1.0) # WorldStrat L2A is float reflectance [0, 1]
        hr_scaled = np.clip(hr_4b / SCALE_WS_HR, 0.0, 1.0)
    else:
        lr_4b = lr_raw[SEN2NAIP_LR_ORDER]
        hr_4b = hr_raw[SEN2NAIP_HR_ORDER]
        lr_scaled = np.clip(lr_4b / SCALE_S2, 0.0, 1.0)
        hr_scaled = np.clip(hr_4b / SCALE_NAIP, 0.0, 1.0)

    # Align spatial dimensions so HR is strictly 4x of LR
    hr_scaled = align_hr_to_4x(lr_scaled, hr_scaled)

    # Fail loudly on any contract violation
    assert lr_scaled.shape[0] == 4, f"Expected 4 channels for LR, got {lr_scaled.shape}"
    assert hr_scaled.shape[0] == 4, f"Expected 4 channels for HR, got {hr_scaled.shape}"
    assert not np.isnan(lr_scaled).any(), f"NaN detected in LR sample: {lr_path}"
    assert not np.isnan(hr_scaled).any(), f"NaN detected in HR sample: {hr_path}"
    assert 0.0 <= lr_scaled.min() and lr_scaled.max() <= 1.0, f"LR value out of range [0, 1]: [{lr_scaled.min()}, {lr_scaled.max()}]"
    assert 0.0 <= hr_scaled.min() and hr_scaled.max() <= 1.0, f"HR value out of range [0, 1]: [{hr_scaled.min()}, {hr_scaled.max()}]"

    return lr_scaled, hr_scaled, lr_meta, hr_meta

# --- Execute Runtime Audit on Disk ---
print('=' * 80)
print('TASK T1: AUDIT OF BAND ORDER & NORMALIZATION CONTRACT')
print('=' * 80)

sen2naip_lr_files = sorted(glob.glob(os.path.join(DATASETS_DIR, 'sentinaiplow', '*.tif')))
sen2naip_hr_files = sorted(glob.glob(os.path.join(DATASETS_DIR, 'sentinaiphigh', '*.tif')))
assert len(sen2naip_lr_files) > 0, f"No SEN2NAIP LR files found in {os.path.join(DATASETS_DIR, 'sentinaiplow')}!"
assert len(sen2naip_hr_files) > 0, f"No SEN2NAIP HR files found in {os.path.join(DATASETS_DIR, 'sentinaiphigh')}!"

lr_s, hr_s, m_lr, m_hr = load_canonical_pair(sen2naip_lr_files[0], sen2naip_hr_files[0], is_worldstrat=False)
print(f"\\n[SEN2NAIP Sample]: {os.path.basename(sen2naip_lr_files[0])}")
print(f"  LR File: count={m_lr['count']}, dtype={m_lr['dtype']}, shape={m_lr['shape']}, descriptions={m_lr['descriptions']}")
print(f"  HR File: count={m_hr['count']}, dtype={m_hr['dtype']}, shape={m_hr['shape']}, descriptions={m_hr['descriptions']}")
print(f"  LR Canonical [R,G,B,N]: shape={lr_s.shape}, range=[{lr_s.min():.4f}, {lr_s.max():.4f}], mean={lr_s.mean():.4f}")
print(f"  HR Canonical [R,G,B,N]: shape={hr_s.shape}, range=[{hr_s.min():.4f}, {hr_s.max():.4f}], mean={hr_s.mean():.4f}")

ws_lr_candidates = sorted(glob.glob(os.path.join(DATASETS_DIR, 'worldstrat_low', '**/L2A/*-L2A_data.tiff'), recursive=True))
ws_hr_candidates = sorted(glob.glob(os.path.join(DATASETS_DIR, 'worldstrat_high', 'extracted/**/*_rgbn.tiff'), recursive=True))

if ws_lr_candidates and ws_hr_candidates:
    lr_ws, hr_ws, m_lws, m_hws = load_canonical_pair(ws_lr_candidates[0], ws_hr_candidates[0], is_worldstrat=True)
    print(f"\\n[WorldStrat Sample]: {os.path.basename(ws_lr_candidates[0])}")
    print(f"  LR File: count={m_lws['count']}, dtype={m_lws['dtype']}, shape={m_lws['shape']}")
    print(f"  HR File: count={m_hws['count']}, dtype={m_hws['dtype']}, shape={m_hws['shape']}")
    print(f"  LR Canonical [R,G,B,N]: shape={lr_ws.shape}, range=[{lr_ws.min():.4f}, {lr_ws.max():.4f}], mean={lr_ws.mean():.4f}")
    print(f"  HR Canonical [R,G,B,N]: shape={hr_ws.shape}, range=[{hr_ws.min():.4f}, {hr_ws.max():.4f}], mean={hr_ws.mean():.4f}")
else:
    print("\\n[WorldStrat]: Not found on disk; proceeding with SEN2NAIP.")

# Verify patch histogram match assertion
test_lr_crop = lr_s[:, :64, :64]
test_hr_crop = hr_s[:, :256, :256]
test_hr_matched = match_radiometry_patch(test_hr_crop, test_lr_crop)
assert test_hr_matched.shape == test_hr_crop.shape, f"Shape mismatch in match_radiometry_patch: {test_hr_matched.shape}"
assert not np.isnan(test_hr_matched).any(), "NaN in matched HR patch"
assert 0.0 <= test_hr_matched.min() and test_hr_matched.max() <= 1.0, f"Matched HR out of bounds: [{test_hr_matched.min()}, {test_hr_matched.max()}]"

print(f"\\n[Histogram Match Test]: Matched HR mean {test_hr_matched.mean():.4f} aligned with LR mean {test_lr_crop.mean():.4f} (Raw HR mean: {test_hr_crop.mean():.4f})")
print('=' * 80)
print('BAND ORDER & NORMALIZATION AUDIT PASSED ALL CHECKS.')
print('=' * 80)"""

# ==============================================================================
# Shared T2 Dataset & Split Code
# ==============================================================================
T2_MD = """## Step 4 — Task T2: Scene-Stratified Split & Dynamic Random Patch Dataset

### Key Architectural Fixes:
1. **Scene-Level Split (80% Train / 10% Val / 10% Test):**
   - Splits by geographical scene ID rather than random patch crops. This guarantees zero spatial data leakage between training, validation, and testing.
   - Held-out test scenes are saved to `Results/test_scenes.json` so both Lite (04) and Full (06) models are evaluated against the exact same scenes.
2. **Dynamic Per-Epoch Random Cropping:**
   - Eliminates the previous single-crop static RAM cache. Every time `__getitem__` is invoked, a fresh random sub-window is sampled across the scene, multiplying sample diversity by thousands.
3. **Synchronized Augmentations:**
   - Random horizontal flips, vertical flips, and 90° rotations applied simultaneously to both LR and HR arrays.
4. **Per-Patch Radiometric Quantile Matching:**
   - Calibrates target HR aerial photography to LR satellite reflectance directly inside the loader, preventing exposure bias."""

T2_DATASET_CODE = """import torch
from torch.utils.data import Dataset, DataLoader
import json, random

# Scene-Level Stratified Split (80% Train, 10% Val, 10% Test)
SPLIT_FILE = os.path.join(RESULTS_DIR, 'test_scenes.json')

def get_scene_splits(val_ratio=0.10, test_ratio=0.10, seed=42, max_samples=None):
    sen2naip_pairs = []
    for lr_p in sen2naip_lr_files:
        base = os.path.basename(lr_p)
        hr_p = os.path.join(DATASETS_DIR, 'sentinaiphigh', base.replace('_lr.tif', '_hr.tif'))
        if os.path.exists(hr_p):
            sen2naip_pairs.append({'id': base.replace('_lr.tif', ''), 'lr': lr_p, 'hr': hr_p, 'is_ws': False})

    ws_pairs = []
    if ws_lr_candidates and ws_hr_candidates:
        hr_map = {os.path.basename(os.path.dirname(p)): p for p in ws_hr_candidates}
        for lr_p in ws_lr_candidates:
            loc = lr_p.split(os.sep)[-3] if os.sep in lr_p else ''
            if loc in hr_map:
                ws_pairs.append({'id': loc, 'lr': lr_p, 'hr': hr_map[loc], 'is_ws': True})

    all_scenes = sen2naip_pairs + ws_pairs
    print(f"Total paired scenes discovered on disk: {len(all_scenes)} (SEN2NAIP: {len(sen2naip_pairs)}, WorldStrat: {len(ws_pairs)})")
    assert len(all_scenes) >= 10, f"Insufficient training scenes ({len(all_scenes)}) found in {DATASETS_DIR}!"

    if max_samples is not None:
        all_scenes = all_scenes[:max_samples]
        print(f"Applied configured sample limit: {len(all_scenes)} scenes.")

    # Check if test_scenes.json already exists to preserve shared test set
    if os.path.exists(SPLIT_FILE):
        try:
            with open(SPLIT_FILE, 'r', encoding='utf-8') as f:
                saved_split = json.load(f)
            saved_test_ids = set(saved_split.get('test_ids', []))
            test_scenes = [s for s in all_scenes if s['id'] in saved_test_ids]
            rem = [s for s in all_scenes if s['id'] not in saved_test_ids]
            if len(test_scenes) >= 3 and len(rem) >= 6:
                rng = random.Random(seed)
                rng.shuffle(rem)
                n_val = max(1, int(len(all_scenes) * val_ratio))
                val_scenes = rem[:n_val]
                train_scenes = rem[n_val:]
                print(f"Reusing existing test split from: {SPLIT_FILE} ({len(test_scenes)} test scenes)")
                print(f"Scene Split: Train={len(train_scenes)}, Val={len(val_scenes)}, Test={len(test_scenes)}")
                return train_scenes, val_scenes, test_scenes
        except Exception as e:
            print(f"Notice: re-creating split ({e})")

    rng = random.Random(seed)
    shuffled = list(all_scenes)
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    n_val = max(1, int(len(shuffled) * val_ratio))
    n_train = len(shuffled) - n_val - n_test

    train_scenes = shuffled[:n_train]
    val_scenes = shuffled[n_train:n_train + n_val]
    test_scenes = shuffled[n_train + n_val:]

    test_ids = [s['id'] for s in test_scenes]
    with open(SPLIT_FILE, 'w', encoding='utf-8') as f:
        json.dump({'test_ids': test_ids, 'test_scenes': test_scenes}, f, indent=2)
    print(f"Saved {len(test_scenes)} held-out test scenes to: {SPLIT_FILE}")
    print(f"Scene Split: Train={len(train_scenes)}, Val={len(val_scenes)}, Test={len(test_scenes)}")
    return train_scenes, val_scenes, test_scenes

train_scenes, val_scenes, test_scenes = get_scene_splits(max_samples=None)

class ScenePatchDataset(Dataset):
    \"\"\"
    Scene-level dynamic patch dataset:
    - Holds scene metadata/paths.
    - Draws a NEW random crop every call/epoch (NO single cached crop!).
    - Applies per-patch histogram matching to align HR radiometry with LR.
    - Applies identical random flips and 90-degree rotations to LR and HR.
    \"\"\"
    def __init__(self, scenes, patch_lr=64, is_train=True):
        self.scenes = scenes
        self.patch_lr = patch_lr
        self.patch_hr = patch_lr * 4
        self.is_train = is_train

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        lr_full, hr_full, _, _ = load_canonical_pair(scene['lr'], scene['hr'], is_worldstrat=scene['is_ws'])
        
        # Ensure HR array is strictly aligned to 4x spatial grid of LR
        hr_full = align_hr_to_4x(lr_full, hr_full)

        _, h_lr, w_lr = lr_full.shape
        _, h_hr, w_hr = hr_full.shape

        if h_lr < self.patch_lr or w_lr < self.patch_lr:
            pad_h = max(0, self.patch_lr - h_lr)
            pad_w = max(0, self.patch_lr - w_lr)
            lr_full = np.pad(lr_full, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            hr_full = np.pad(hr_full, ((0,0), (0, pad_h*4), (0, pad_w*4)), mode='reflect')
            _, h_lr, w_lr = lr_full.shape
            _, h_hr, w_hr = hr_full.shape

        if self.is_train:
            y = np.random.randint(0, h_lr - self.patch_lr + 1)
            x = np.random.randint(0, w_lr - self.patch_lr + 1)
        else:
            y = (h_lr - self.patch_lr) // 2
            x = (w_lr - self.patch_lr) // 2

        lr_crop = lr_full[:, y : y + self.patch_lr, x : x + self.patch_lr]
        hr_crop = hr_full[:, y*4 : y*4 + self.patch_hr, x*4 : x*4 + self.patch_hr]

        # Failsafe shape normalization
        if hr_crop.shape[1] != self.patch_hr or hr_crop.shape[2] != self.patch_hr:
            hr_t_tmp = torch.from_numpy(hr_crop).unsqueeze(0).float()
            hr_crop = torch.nn.functional.interpolate(hr_t_tmp, size=(self.patch_hr, self.patch_hr), mode='bilinear', align_corners=False).squeeze(0).numpy()

        if lr_crop.shape[1] != self.patch_lr or lr_crop.shape[2] != self.patch_lr:
            lr_t_tmp = torch.from_numpy(lr_crop).unsqueeze(0).float()
            lr_crop = torch.nn.functional.interpolate(lr_t_tmp, size=(self.patch_lr, self.patch_lr), mode='bilinear', align_corners=False).squeeze(0).numpy()

        hr_crop = match_radiometry_patch(hr_crop, lr_crop)

        if self.is_train:
            if np.random.rand() > 0.5:
                lr_crop = np.flip(lr_crop, axis=2)
                hr_crop = np.flip(hr_crop, axis=2)
            if np.random.rand() > 0.5:
                lr_crop = np.flip(lr_crop, axis=1)
                hr_crop = np.flip(hr_crop, axis=1)
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                lr_crop = np.rot90(lr_crop, k=rot_k, axes=(1, 2))
                hr_crop = np.rot90(hr_crop, k=rot_k, axes=(1, 2))

        lr_t = torch.from_numpy(lr_crop.copy()).float()
        hr_t = torch.from_numpy(hr_crop.copy()).float()

        assert lr_t.shape == (4, self.patch_lr, self.patch_lr), f"LR shape error: {lr_t.shape}"
        assert hr_t.shape == (4, self.patch_hr, self.patch_hr), f"HR shape error: {hr_t.shape}"
        return lr_t, hr_t"""

# ==============================================================================
# Shared T3 Loss Code
# ==============================================================================
T3_MD = """## Step 5 — Task T3: Multi-Objective Compound Loss with Shift-Tolerant L1

### Loss Formulation:
$$\mathcal{L}_{\\text{total}} = \mathcal{L}_{\\text{shift-L1}} + 0.20 \cdot \mathcal{L}_{\\text{Sobel}} + 0.15 \cdot \left(\mathcal{L}_{\\text{NDVI}} + 0.5 \cdot \mathcal{L}_{\\text{NDWI}}\\right)$$

1. **Shift-Tolerant L1 (`ShiftTolerantL1Loss`):**
   - Evaluates L1 reconstruction error across $\pm 2$ pixel sub-grid shifts:
     $$\mathcal{L}_{\\text{shift-L1}}(P, Y) = \min_{\Delta x, \Delta y \in [-2, 2]} \| P_{c} - Y_{\Delta x, \Delta y} \|_1$$
   - Prevents artificial penalty from known sub-pixel georeferencing jitter between satellite and airborne sensors.
2. **Canonical Band References:**
   - Uses `BAND_RED=0`, `BAND_GREEN=1`, `BAND_BLUE=2`, `BAND_NIR=3` instead of hardcoded numbers."""

T3_LOSS_CODE = """import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_RED   = 0
BAND_GREEN = 1
BAND_BLUE  = 2
BAND_NIR   = 3

class ShiftTolerantL1Loss(nn.Module):
    \"\"\"
    Shift-Tolerant L1 Loss: Evaluates L1 error over +/- max_shift pixel shifts.
    Prevents artificial penalty from sub-pixel georeferencing jitter between satellite & aerial sensors.
    \"\"\"
    def __init__(self, max_shift=2):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, pred, target):
        ms = self.max_shift
        p_crop = pred[:, :, ms:-ms, ms:-ms]
        losses = []
        for dy in range(-ms, ms + 1):
            for dx in range(-ms, ms + 1):
                t_crop = target[:, :, ms+dy : target.shape[2]-ms+dy, ms+dx : target.shape[3]-ms+dx]
                losses.append(F.l1_loss(p_crop, t_crop, reduction='none').mean(dim=[1, 2, 3]))
        stacked = torch.stack(losses, dim=0)
        return torch.min(stacked, dim=0)[0].mean()

class SobelEdgeLoss(nn.Module):
    \"\"\"Sobel edge gradient loss for sharp structural boundaries.\"\"\"
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def forward(self, pred, target):
        loss = 0.0
        for c in range(pred.shape[1]):
            p = pred[:, c:c+1]
            t = target[:, c:c+1]
            gx_p, gy_p = F.conv2d(p, self.sx, padding=1), F.conv2d(p, self.sy, padding=1)
            gx_t, gy_t = F.conv2d(t, self.sx, padding=1), F.conv2d(t, self.sy, padding=1)
            loss += F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
        return loss / pred.shape[1]

class MultiIndexSpectralLoss(nn.Module):
    \"\"\"Differentiable loss on physical NDVI and NDWI indices using canonical band indices.\"\"\"
    def __init__(self, eps=1e-6, weight_ndwi=0.5):
        super().__init__()
        self.eps = eps
        self.weight_ndwi = weight_ndwi

    def forward(self, pred, target):
        red_p, green_p, nir_p = pred[:, BAND_RED], pred[:, BAND_GREEN], pred[:, BAND_NIR]
        red_t, green_t, nir_t = target[:, BAND_RED], target[:, BAND_GREEN], target[:, BAND_NIR]

        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi_p = (nir_p - red_p) / (nir_p + red_p + self.eps)
        ndvi_t = (nir_t - red_t) / (nir_t + red_t + self.eps)
        l_ndvi = F.l1_loss(ndvi_p, ndvi_t)

        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi_p = (green_p - nir_p) / (green_p + nir_p + self.eps)
        ndwi_t = (green_t - nir_t) / (green_t + nir_t + self.eps)
        l_ndwi = F.l1_loss(ndwi_p, ndwi_t)

        return l_ndvi + (self.weight_ndwi * l_ndwi), {'ndvi': l_ndvi.item(), 'ndwi': l_ndwi.item()}

class MajorProjectCompoundLoss(nn.Module):
    def __init__(self, lambda_edge=0.20, lambda_spectral=0.15, use_shift_tolerant_l1=True):
        super().__init__()
        self.l1 = ShiftTolerantL1Loss(max_shift=2) if use_shift_tolerant_l1 else nn.L1Loss()
        self.edge = SobelEdgeLoss()
        self.spec = MultiIndexSpectralLoss()
        self.lambda_edge = lambda_edge
        self.lambda_spectral = lambda_spectral

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_edge = self.edge(pred, target)
        loss_spec, spec_dict = self.spec(pred, target)

        total = loss_l1 + (self.lambda_edge * loss_edge) + (self.lambda_spectral * loss_spec)
        return total, {'l1': loss_l1.item(), 'edge': loss_edge.item(), **spec_dict}

criterion = MajorProjectCompoundLoss(lambda_edge=0.20, lambda_spectral=0.15, use_shift_tolerant_l1=True).to(device)
print('Multi-Objective Shift-Tolerant Compound Loss initialized.')"""


def build_lite_train_notebook():
    cells = []
    # Title
    cells.append(make_cell("markdown", "# Joint Fine-Tuning: SEN2SR Lite (10m -> 2.5m) with Shift-Tolerant Compound Loss\n\n**Major Project — Spectral Enhancement**\n- **Target:** Physical 10m to 2.5m super-resolution evaluated directly against NAIP aerial ground truth.\n- **Loss:** Shift-Tolerant L1 + Sobel Edge Gradient + NDVI/NDWI Multi-Index Spectral Loss.\n- **Data Policy:** Scene-stratified split (80/10/10), dynamic random per-epoch cropping, per-patch quantile matching."))
    
    # Colab Setup
    cells.append(make_cell("markdown", "## Step 1 — Colab Environment & Google Drive Setup"))
    c2 = """import os, sys, glob, time, warnings, logging

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CPL_LOG_ERRORS'] = 'OFF'
os.environ['CPL_LOG'] = 'OFF'
os.environ['GDAL_PAM_ENABLED'] = 'NO'

warnings.filterwarnings('ignore')
for logger_name in ['rasterio', 'rasterio._env', 'rasterio.env', 'rasterio._base']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

PROJECT_DIR = '/content/drive/My Drive/Project_Folder'
DATASETS_DIR = os.path.join(PROJECT_DIR, 'Datasets')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'Results')
MODEL_DIR    = '/content/model/SEN2SR_LITE'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch: {torch.__version__} | Active Device: {device} | CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')"""
    cells.append(make_cell("code", c2))

    # Install Dependencies
    cells.append(make_cell("markdown", "## Step 2 — Install Required Dependencies"))
    c4 = """%pip install -q sen2sr mlstac rasterio safetensors pandas matplotlib tqdm"""
    cells.append(make_cell("code", c4))

    # T1 Markdown & Code
    cells.append(make_cell("markdown", T1_MD))
    cells.append(make_cell("code", T1_CODE))

    # T2 Markdown & Code
    cells.append(make_cell("markdown", T2_MD))
    c8 = T2_DATASET_CODE + """\n
train_ds = ScenePatchDataset(train_scenes, patch_lr=64, is_train=True)
val_ds   = ScenePatchDataset(val_scenes, patch_lr=64, is_train=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
print(f"DataLoaders ready: {len(train_loader)} train batches/epoch, {len(val_loader)} val batches/epoch.")"""
    cells.append(make_cell("code", c8))

    # T3 Markdown & Code
    cells.append(make_cell("markdown", T3_MD))
    cells.append(make_cell("code", T3_LOSS_CODE))

    # Model Init
    cells.append(make_cell("markdown", "## Step 6 — Pretrained Lite Architecture & Weights Initialization"))
    c12 = """import mlstac
import safetensors.torch
import urllib.request
from sen2sr.models.opensr_baseline.cnn import CNNSR

MODEL_DIR = '/content/model/SEN2SR_LITE'
os.makedirs(MODEL_DIR, exist_ok=True)
weights_path = os.path.join(MODEL_DIR, 'sr_model.safetensor')

if not os.path.exists(weights_path):
    print('Downloading official SEN2SR Lite model weights from Taco Foundation...')
    try:
        mlstac.download(
            file='https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/main/mlm.json',
            output_dir=MODEL_DIR
        )
    except Exception as e:
        print(f'mlstac download notice ({e}); downloading safetensor directly...')
        u = 'https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/main/sr_model.safetensor'
        urllib.request.urlretrieve(u, weights_path)

print('Model files:', os.listdir(MODEL_DIR))

# Initialize CNNSR (4 in_channels, 4 out_channels, 24 feature channels, 6 residual blocks)
model = CNNSR(
    in_channels=4,
    out_channels=4,
    feature_channels=24,
    upscale=4,
    bias=True,
    train_mode=True,
    num_blocks=6
)

weights = safetensors.torch.load_file(weights_path)
model.load_state_dict(weights)
model = model.to(device)
print(f'SEN2SR Lite initialized successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}')"""
    cells.append(make_cell("code", c12))

    # Training Loop
    cells.append(make_cell("markdown", "## Step 7 — Fine-Tuning Loop with Best-Val Checkpointing"))
    c14 = """import time
from tqdm import tqdm

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

BEST_CKPT_PATH = os.path.join(RESULTS_DIR, 'sen2sr_lite_compound_best.pth')
LAST_CKPT_PATH = os.path.join(RESULTS_DIR, 'sen2sr_lite_compound_last.pth')
HISTORY_PATH   = os.path.join(RESULTS_DIR, 'sen2sr_lite_training_history.json')

EPOCHS = 15
history = {'train_loss': [], 'val_loss': [], 'train_l1': [], 'val_l1': [], 'ndvi_loss': [], 'ndwi_loss': []}
best_val_loss = float('inf')

print('=' * 80)
print('STARTING FINE-TUNING LOOP: SEN2SR Lite (10m -> 2.5m)')
print(f'Train scenes: {len(train_ds)} | Val scenes: {len(val_ds)} | Epochs: {EPOCHS}')
print('=' * 80)

t0_train = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_train = 0.0
    running_l1 = 0.0
    running_ndvi = 0.0
    running_ndwi = 0.0

    for lr_batch, hr_batch in train_loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        optimizer.zero_grad()
        sr_batch = model(lr_batch)
        loss, details = criterion(sr_batch, hr_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_train += loss.item() * lr_batch.size(0)
        running_l1 += details['l1'] * lr_batch.size(0)
        running_ndvi += details['ndvi'] * lr_batch.size(0)
        running_ndwi += details['ndwi'] * lr_batch.size(0)

    epoch_train_loss = running_train / len(train_ds)
    epoch_l1 = running_l1 / len(train_ds)
    epoch_ndvi = running_ndvi / len(train_ds)
    epoch_ndwi = running_ndwi / len(train_ds)

    # Validation evaluation on held-out scenes
    model.eval()
    running_val = 0.0
    running_val_l1 = 0.0
    with torch.no_grad():
        for lr_val, hr_val in val_loader:
            lr_val = lr_val.to(device)
            hr_val = hr_val.to(device)
            sr_val = model(lr_val)
            v_loss, v_details = criterion(sr_val, hr_val)
            running_val += v_loss.item() * lr_val.size(0)
            running_val_l1 += v_details['l1'] * lr_val.size(0)

    epoch_val_loss = running_val / len(val_ds)
    epoch_val_l1 = running_val_l1 / len(val_ds)
    scheduler.step()

    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_l1'].append(epoch_l1)
    history['val_l1'].append(epoch_val_l1)
    history['ndvi_loss'].append(epoch_ndvi)
    history['ndwi_loss'].append(epoch_ndwi)

    is_best = epoch_val_loss < best_val_loss
    if is_best:
        best_val_loss = epoch_val_loss
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': best_val_loss}, BEST_CKPT_PATH)

    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {epoch_train_loss:.4f} (L1: {epoch_l1:.4f}) | Val Loss: {epoch_val_loss:.4f} (L1: {epoch_val_l1:.4f}) {'[BEST SAVED]' if is_best else ''}")

# Always save last checkpoint
torch.save({'epoch': EPOCHS, 'model_state_dict': model.state_dict(), 'val_loss': epoch_val_loss}, LAST_CKPT_PATH)

with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
    json.dump(history, f, indent=2)

print(f"\\nTraining completed in {time.time() - t0_train:.1f}s.")
print(f"  Best Val Loss: {best_val_loss:.4f} -> Saved to: {BEST_CKPT_PATH}")
print(f"  Last Checkpoint: Saved to: {LAST_CKPT_PATH}")"""
    cells.append(make_cell("code", c14))

    # Evaluation
    cells.append(make_cell("markdown", "## Step 8 — Loss Curves & Direct Aerial Ground-Truth Verification"))
    c16 = """import matplotlib.pyplot as plt

# 1. Loss Curves
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(history['train_loss'], label='Train Total', color='blue', lw=2)
ax[0].plot(history['val_loss'], label='Val Total', color='orange', lw=2)
ax[0].set_title('Total Compound Loss Curve')
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss'); ax[0].legend(); ax[0].grid(True)

ax[1].plot(history['train_l1'], label='Train L1', color='green', lw=2)
ax[1].plot(history['val_l1'], label='Val L1', color='red', lw=2)
ax[1].set_title('Shift-Tolerant L1 Loss Curve')
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('L1 Error'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sen2sr_lite_loss_curves.png'), dpi=150)
plt.show()

# 2. Side-by-Side Visual Verification on First Held-Out Test Scene
test_sample = test_scenes[0]
lr_t_eval, hr_t_eval, _, _ = load_canonical_pair(test_sample['lr'], test_sample['hr'], is_worldstrat=test_sample['is_ws'])

cy, cx = lr_t_eval.shape[1] // 2, lr_t_eval.shape[2] // 2
lr_crop = lr_t_eval[:, cy-32:cy+32, cx-32:cx+32]
hr_crop = hr_t_eval[:, (cy-32)*4:(cy+32)*4, (cx-32)*4:(cx+32)*4]
hr_crop = match_radiometry_patch(hr_crop, lr_crop)

# Load best checkpoint
ckpt = torch.load(BEST_CKPT_PATH, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

with torch.no_grad():
    in_t = torch.from_numpy(lr_crop).unsqueeze(0).to(device)
    sr_crop = model(in_t).squeeze(0).cpu().numpy()

in_bicubic = torch.nn.functional.interpolate(in_t, scale_factor=4, mode='bicubic', align_corners=False).squeeze(0).cpu().numpy()

# Baseline model
baseline_model = CNNSR(in_channels=4, out_channels=4, feature_channels=24, upscale=4, bias=True, train_mode=False, num_blocks=6)
baseline_weights = safetensors.torch.load_file(os.path.join(MODEL_DIR, 'sr_model.safetensor'))
baseline_model.load_state_dict(baseline_weights)
baseline_model = baseline_model.to(device).eval()

with torch.no_grad():
    base_crop = baseline_model(in_t).squeeze(0).cpu().numpy()

# Shared Percentile Stretch strictly computed from the real NAIP Ground Truth
def shared_stretch(img_4b, ref_hr):
    rgb = img_4b[[0, 1, 2]].transpose(1, 2, 0)
    ref_rgb = ref_hr[[0, 1, 2]].transpose(1, 2, 0)
    lo = np.percentile(ref_rgb, 2)
    hi = np.percentile(ref_rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(shared_stretch(in_bicubic, hr_crop))
axes[0].set_title(f'1. Native 10m Input (Bicubic 4x)\\nScene: {test_sample["id"]}', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(shared_stretch(base_crop, hr_crop))
axes[1].set_title('2. Pretrained Baseline (2.5m)', fontweight='bold')
axes[1].axis('off')

axes[2].imshow(shared_stretch(sr_crop, hr_crop))
axes[2].set_title('3. Fine-Tuned (2.5m, Ours)\\nBest Val Checkpoint', fontweight='bold', color='green')
axes[2].axis('off')

axes[3].imshow(shared_stretch(hr_crop, hr_crop))
axes[3].set_title('4. Real Aerial Ground Truth (NAIP)\\nShared Contrast Stretch', fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
eval_png = os.path.join(RESULTS_DIR, 'sen2sr_lite_test_validation_preview.png')
plt.savefig(eval_png, dpi=180, bbox_inches='tight')
plt.show()
print(f'Test visual saved to: {eval_png}')"""
    cells.append(make_cell("code", c16))

    return create_notebook(cells)


def build_full_train_notebook():
    cells = []
    # Title
    cells.append(make_cell("markdown", "# Joint Fine-Tuning: SEN2SR Full (`MambaSR` 10m -> 2.5m) with Shift-Tolerant Compound Loss\n\n**Major Project — Spectral Enhancement**\n- **Target:** Physical 10m to 2.5m super-resolution evaluated directly against NAIP aerial ground truth.\n- **Loss:** Shift-Tolerant L1 + Sobel Edge Gradient + NDVI/NDWI Multi-Index Spectral Loss.\n- **Architecture:** `MambaSR` state-space model with robust pure-PyTorch selective-scan fallback for Colab.\n- **Data Policy:** Shared scene-stratified split (`Results/test_scenes.json`), dynamic per-epoch random cropping, per-patch quantile matching."))

    # Colab Setup
    cells.append(make_cell("markdown", "## Step 1 — Colab Environment & Google Drive Setup"))
    c2 = """import os, sys, glob, time, warnings, logging

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CPL_LOG_ERRORS'] = 'OFF'
os.environ['CPL_LOG'] = 'OFF'
os.environ['GDAL_PAM_ENABLED'] = 'NO'

warnings.filterwarnings('ignore')
for logger_name in ['rasterio', 'rasterio._env', 'rasterio.env', 'rasterio._base']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

PROJECT_DIR = '/content/drive/My Drive/Project_Folder'
DATASETS_DIR = os.path.join(PROJECT_DIR, 'Datasets')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'Results')
MODEL_DIR    = '/content/model/SEN2SR_FULL'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch: {torch.__version__} | Active Device: {device} | CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')"""
    cells.append(make_cell("code", c2))

    # Install & Mamba Engine
    cells.append(make_cell("markdown", "## Step 2 — Install Required Dependencies & Initialize Mamba Fallback Engine"))
    c4 = """%pip install -q sen2sr mlstac rasterio safetensors einops timm

import types, sys
from einops import rearrange, repeat

# -----------------------------------------------------------------------------------------
# Robust Mamba-SSM Fallback Engine (prevents Colab undefined symbol crash)
# -----------------------------------------------------------------------------------------
mamba_ready = False

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    test_u = torch.randn(1, 4, 16, device=device)
    test_delta = torch.randn(1, 4, 16, device=device)
    test_A = torch.randn(4, 8, device=device)
    test_B = torch.randn(1, 8, 16, device=device)
    test_C = torch.randn(1, 8, 16, device=device)
    test_D = torch.randn(4, device=device)
    _ = selective_scan_fn(test_u, test_delta, test_A, test_B, test_C, test_D)
    mamba_ready = True
    print('Native compiled Mamba CUDA engine verified successfully.')
except Exception as e:
    print(f'Notice: Native CUDA selective_scan unavailable ({e}).')
    print('Activating pure PyTorch selective scan fallback engine for Colab...')

    def pytorch_selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        batch, dim, dstate = u.shape[0], u.shape[1], A.shape[1]
        length = u.shape[2]

        if delta_bias is not None:
            delta = delta + delta_bias.unsqueeze(-1)
        if delta_softplus:
            delta = torch.nn.functional.softplus(delta)

        # Discretize continuous parameters
        dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if B.dim() == 3:
            if B.shape[1] == dstate:
                dB = torch.einsum('bdl,bnl->bdln', delta, B)
            else:
                dB = torch.einsum('bdl,bdnl->bdln', delta, B)
        else:
            dB = torch.einsum('bdl,bn->bdln', delta, B)

        du = u.unsqueeze(-1) * dB
        x = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(length):
            x = dA[:, :, i] * x + du[:, :, i]
            if C.dim() == 3:
                if C.shape[1] == dstate:
                    y_i = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y_i = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            else:
                y_i = torch.einsum('bdn,n->bd', x, C)
            ys.append(y_i)

        y = torch.stack(ys, dim=-1)
        if D is not None:
            y = y + u * D.unsqueeze(-1)
        if z is not None:
            y = y * torch.nn.functional.silu(z)

        return (y, x) if return_last_state else y

    mamba_mod = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    mamba_mod.selective_scan_fn = pytorch_selective_scan
    sys.modules['mamba_ssm.ops.selective_scan_interface'] = mamba_mod
    print('Pure PyTorch selective scan fallback initialized successfully.')

from sen2sr.models.mambasr.mambasr import MambaSR
print('MambaSR architecture successfully loaded.')"""
    cells.append(make_cell("code", c4))

    # T1 Markdown & Code
    cells.append(make_cell("markdown", T1_MD))
    cells.append(make_cell("code", T1_CODE))

    # T2 Markdown & Code
    cells.append(make_cell("markdown", T2_MD))
    c8 = T2_DATASET_CODE + """\n
# Full MambaSR uses patch size 64 LR (256 HR) and batch size 4
train_ds = ScenePatchDataset(train_scenes, patch_lr=64, is_train=True)
val_ds   = ScenePatchDataset(val_scenes, patch_lr=64, is_train=False)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
print(f"DataLoaders ready for Full Model: {len(train_loader)} train batches/epoch, {len(val_loader)} val batches/epoch.")"""
    cells.append(make_cell("code", c8))

    # T3 Markdown & Code
    cells.append(make_cell("markdown", T3_MD))
    cells.append(make_cell("code", T3_LOSS_CODE))

    # Model Init
    cells.append(make_cell("markdown", "## Step 6 — Pretrained MambaSR Architecture & Weights Initialization"))
    c12 = """import mlstac
import safetensors.torch
import urllib.request

MODEL_DIR = '/content/model/SEN2SR_FULL'
os.makedirs(MODEL_DIR, exist_ok=True)
weights_path = os.path.join(MODEL_DIR, 'sr_model.safetensor')

if not os.path.exists(weights_path):
    print('Downloading official SEN2SR Full model weights from Taco Foundation...')
    try:
        mlstac.download(
            file='https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SR/main/mlm.json',
            output_dir=MODEL_DIR
        )
    except Exception as e:
        print(f'mlstac download notice ({e}); downloading safetensor directly...')
        u = 'https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SR/main/sr_model.safetensor'
        urllib.request.urlretrieve(u, weights_path)

print('Model files:', os.listdir(MODEL_DIR))

# Stage 1 RGBN MambaSR parameters
sr_parameters = {
    "img_size": (128, 128),
    "in_channels": 4,
    "out_channels": 4,
    "embed_dim": 96,
    "depths": [8, 8, 8, 8, 8, 8],
    "num_heads": [8, 8, 8, 8, 8, 8],
    "mlp_ratio": 4,
    "upscale": 4,
    "attention_type": "sigmoid_02",
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
    "operation_attention": "sum",
}

print('Initializing MambaSR architecture...')
model = MambaSR(**sr_parameters)

weights = safetensors.torch.load_file(weights_path)
model.load_state_dict(weights)
model = model.to(device)
print(f'MambaSR Stage 1 initialized successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}')"""
    cells.append(make_cell("code", c12))

    # Training Loop
    cells.append(make_cell("markdown", "## Step 7 — Fine-Tuning Loop with Best-Val Checkpointing"))
    c14 = """import time
from tqdm import tqdm

# Fine-tune the deep spatial reconstruction and upsampling heads
trainable_params = (
    list(model.conv_after_body.parameters()) +
    list(model.conv_before_upsample.parameters()) +
    list(model.upsample.parameters()) +
    list(model.conv_last.parameters())
)

optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

BEST_CKPT_PATH = os.path.join(RESULTS_DIR, 'sen2sr_full_compound_best.pth')
LAST_CKPT_PATH = os.path.join(RESULTS_DIR, 'sen2sr_full_compound_last.pth')
HISTORY_PATH   = os.path.join(RESULTS_DIR, 'sen2sr_full_training_history.json')

EPOCHS = 15
history = {'train_loss': [], 'val_loss': [], 'train_l1': [], 'val_l1': [], 'ndvi_loss': [], 'ndwi_loss': []}
best_val_loss = float('inf')

print('=' * 80)
print('STARTING FINE-TUNING LOOP: SEN2SR Full (MambaSR 10m -> 2.5m)')
print(f'Train scenes: {len(train_ds)} | Val scenes: {len(val_ds)} | Epochs: {EPOCHS}')
print('=' * 80)

t0_train = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_train = 0.0
    running_l1 = 0.0
    running_ndvi = 0.0
    running_ndwi = 0.0

    for lr_batch, hr_batch in train_loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        optimizer.zero_grad()
        sr_batch = model(lr_batch)
        loss, details = criterion(sr_batch, hr_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        running_train += loss.item() * lr_batch.size(0)
        running_l1 += details['l1'] * lr_batch.size(0)
        running_ndvi += details['ndvi'] * lr_batch.size(0)
        running_ndwi += details['ndwi'] * lr_batch.size(0)

    epoch_train_loss = running_train / len(train_ds)
    epoch_l1 = running_l1 / len(train_ds)
    epoch_ndvi = running_ndvi / len(train_ds)
    epoch_ndwi = running_ndwi / len(train_ds)

    # Validation evaluation on held-out scenes
    model.eval()
    running_val = 0.0
    running_val_l1 = 0.0
    with torch.no_grad():
        for lr_val, hr_val in val_loader:
            lr_val = lr_val.to(device)
            hr_val = hr_val.to(device)
            sr_val = model(lr_val)
            v_loss, v_details = criterion(sr_val, hr_val)
            running_val += v_loss.item() * lr_val.size(0)
            running_val_l1 += v_details['l1'] * lr_val.size(0)

    epoch_val_loss = running_val / len(val_ds)
    epoch_val_l1 = running_val_l1 / len(val_ds)
    scheduler.step()

    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_l1'].append(epoch_l1)
    history['val_l1'].append(epoch_val_l1)
    history['ndvi_loss'].append(epoch_ndvi)
    history['ndwi_loss'].append(epoch_ndwi)

    is_best = epoch_val_loss < best_val_loss
    if is_best:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), BEST_CKPT_PATH)

    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {epoch_train_loss:.4f} (L1: {epoch_l1:.4f}) | Val Loss: {epoch_val_loss:.4f} (L1: {epoch_val_l1:.4f}) {'[BEST SAVED]' if is_best else ''}")

# Always save last checkpoint
torch.save(model.state_dict(), LAST_CKPT_PATH)

with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
    json.dump(history, f, indent=2)

print(f"\\nTraining completed in {time.time() - t0_train:.1f}s.")
print(f"  Best Val Loss: {best_val_loss:.4f} -> Saved to: {BEST_CKPT_PATH}")
print(f"  Last Checkpoint: Saved to: {LAST_CKPT_PATH}")"""
    cells.append(make_cell("code", c14))

    # Evaluation
    cells.append(make_cell("markdown", "## Step 8 — Loss Curves & Direct Aerial Ground-Truth Verification"))
    c16 = """import matplotlib.pyplot as plt

# 1. Loss Curves
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(history['train_loss'], label='Train Total', color='blue', lw=2)
ax[0].plot(history['val_loss'], label='Val Total', color='orange', lw=2)
ax[0].set_title('MambaSR Total Compound Loss Curve')
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss'); ax[0].legend(); ax[0].grid(True)

ax[1].plot(history['train_l1'], label='Train L1', color='green', lw=2)
ax[1].plot(history['val_l1'], label='Val L1', color='red', lw=2)
ax[1].set_title('Shift-Tolerant L1 Loss Curve')
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('L1 Error'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sen2sr_full_loss_curves.png'), dpi=150)
plt.show()

# 2. Side-by-Side Visual Verification on First Held-Out Test Scene
test_sample = test_scenes[0]
lr_t_eval, hr_t_eval, _, _ = load_canonical_pair(test_sample['lr'], test_sample['hr'], is_worldstrat=test_sample['is_ws'])

cy, cx = lr_t_eval.shape[1] // 2, lr_t_eval.shape[2] // 2
lr_crop = lr_t_eval[:, cy-32:cy+32, cx-32:cx+32]
hr_crop = hr_t_eval[:, (cy-32)*4:(cy+32)*4, (cx-32)*4:(cx+32)*4]
hr_crop = match_radiometry_patch(hr_crop, lr_crop)

# Load best checkpoint
model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device))
model.eval()

with torch.no_grad():
    in_t = torch.from_numpy(lr_crop).unsqueeze(0).to(device)
    sr_crop = model(in_t).squeeze(0).cpu().numpy()

in_bicubic = torch.nn.functional.interpolate(in_t, scale_factor=4, mode='bicubic', align_corners=False).squeeze(0).cpu().numpy()

# Baseline model
baseline_model = MambaSR(**sr_parameters)
baseline_model.load_state_dict(weights)
baseline_model = baseline_model.to(device).eval()

with torch.no_grad():
    base_crop = baseline_model(in_t).squeeze(0).cpu().numpy()

def shared_stretch(img_4b, ref_hr):
    rgb = img_4b[[0, 1, 2]].transpose(1, 2, 0)
    ref_rgb = ref_hr[[0, 1, 2]].transpose(1, 2, 0)
    lo = np.percentile(ref_rgb, 2)
    hi = np.percentile(ref_rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(shared_stretch(in_bicubic, hr_crop))
axes[0].set_title(f'1. Native 10m Input (Bicubic 4x)\\nScene: {test_sample["id"]}', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(shared_stretch(base_crop, hr_crop))
axes[1].set_title('2. Pretrained Full Baseline (2.5m)', fontweight='bold')
axes[1].axis('off')

axes[2].imshow(shared_stretch(sr_crop, hr_crop))
axes[2].set_title('3. Fine-Tuned Full (2.5m, Ours)\\nBest Val Checkpoint', fontweight='bold', color='green')
axes[2].axis('off')

axes[3].imshow(shared_stretch(hr_crop, hr_crop))
axes[3].set_title('4. Real Aerial Ground Truth (NAIP)\\nShared Contrast Stretch', fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
eval_png = os.path.join(RESULTS_DIR, 'sen2sr_full_test_validation_preview.png')
plt.savefig(eval_png, dpi=180, bbox_inches='tight')
plt.show()
print(f'Test visual saved to: {eval_png}')"""
    cells.append(make_cell("code", c16))

    return create_notebook(cells)


def validate_and_save(nb, path):
    # Validate Python AST of each code cell
    for idx, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            code_str = "".join(c['source'])
            clean_lines = [l for l in code_str.split("\n") if not l.strip().startswith(('%pip', '!', '%'))]
            clean_code = "\n".join(clean_lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                raise SyntaxError(f"AST SyntaxError in cell {idx} of {path}: {e}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Validated and saved: {path} ({len(nb['cells'])} cells)")

if __name__ == '__main__':
    lite_nb = build_lite_train_notebook()
    p_lite = 'notebooks/04_finetuning_experiments/Train_SEN2SR_Lite_Compound_Loss.ipynb'
    validate_and_save(lite_nb, p_lite)

    full_nb = build_full_train_notebook()
    p_full = 'notebooks/06_full_model_experiments/Train_SEN2SR_Full_Compound_Loss.ipynb'
    validate_and_save(full_nb, p_full)

    # Sync to G: Drive if present
    gdrive_lite = 'G:/My Drive/Project_Folder/Notebooks/04_finetuning_experiments/Train_SEN2SR_Lite_Compound_Loss.ipynb'
    gdrive_full = 'G:/My Drive/Project_Folder/Notebooks/06_full_model_experiments/Train_SEN2SR_Full_Compound_Loss.ipynb'
    if os.path.exists(os.path.dirname(gdrive_lite)):
        with open(gdrive_lite, 'w', encoding='utf-8') as f:
            json.dump(lite_nb, f, indent=2)
        print(f"Synced to Google Drive: {gdrive_lite}")
    if os.path.exists(os.path.dirname(gdrive_full)):
        with open(gdrive_full, 'w', encoding='utf-8') as f:
            json.dump(full_nb, f, indent=2)
        print(f"Synced to Google Drive: {gdrive_full}")
