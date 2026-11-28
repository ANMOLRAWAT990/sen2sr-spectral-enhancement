"""
Generator for 04c_Lite_ResidualSR_Cache_And_LSGAN.ipynb
Built strictly from the SEN2SR Major Project Meeting Notes, Plan and Build Spec (Sections 10-16).
Writes to a new notebook in Folder 4 and uses a dedicated checkpoint directory (exp_04c_residual_sr)
so no existing notebooks, checkpoints, or results are overwritten.
"""

import json, os, ast

def make_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

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

def build_04c_notebook():
    cells = []

    # Cell 0: Title & Spec Notes
    cells.append(make_cell("markdown", """# SEN2SR Lite (v2): ResidualSR, Memory-Mapped Patch Cache & Sharpness-Guarded LSGAN

**Major Project — 4x Sentinel-2 RGB+NIR Super-Resolution (10 m → 2.5 m)**

> [!NOTE]
> **Build Status:** Built from the project spec (Sections 10–16). All training cells are resumable and isolated under `Results/exp_04c_residual_sr/` so existing checkpoints (`sen2sr_lite_*.pth`) and `test_scenes.json` are never overwritten.
> Run on a **GPU (Tesla T4)** runtime; Cell 2 fails fast on CPU.

### Key Architectural & Engineering Upgrades
1. **Cell A (Fail-Fast GPU & Hardware Check):** Verifies CUDA T4 availability and `/content` disk headroom before any heavy step.
2. **Cells B1–B2 (Memory-Mapped Patch Cache):** Precomputes `float16` `64×64 -> 256×256` patches in `/content/cache/*.npy` with `match_radiometry_patch` and `q_psnr` pair-quality scores so `__getitem__` does zero raster I/O or `argsort` during training.
3. **Cell C (`ResidualSR` + `VGGMulti` + `lrcons`):**
   - `ResidualSR`: `bicubic(x) + learned_residual` with zero-initialized output conv and nearest-neighbor upsampling (starts at bicubic quality, no PixelShuffle checkerboard).
   - `VGGMulti`: Multi-layer VGG19 loss on stretched (`x / 0.3`) true-colour (`[R,G,B]`) and NIR false-colour (`[NIR,R,G]`).
   - `lrcons`: Avg-pool LR-consistency loss protecting Spectral Angle Mapper (SAM).
4. **Cell D1 (Phase 1 — Pixel/Spectral Pretraining):** 10 epochs with EMA (`0.999`), resumable every epoch to Drive.
5. **Cell D2 (Phase 2 — LSGAN Fine-Tuning):** 25 epochs with 2-epoch loss ramp, resumable, selecting the best checkpoint by `abs(HF_ratio - 1.0)` subject to `SAM <= 1.70°`.
6. **Cells E–F (Evaluation & Alpha-Blend Trade-Off):** Adds `HF-ratio`, `LPIPS`, and `alpha`-blend curve (`[0, 0.25, 0.5, 0.75, 1.0]`), plus no-reference checks on Indian Copernicus captures."""))

    # Step 1: Setup & Cell A
    cells.append(make_cell("markdown", "## Step 1 — Environment, Paths & Fail-Fast GPU Check (Cell A)"))
    c1 = """import os, sys, glob, time, json, math, random, warnings, logging

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CPL_LOG_ERRORS'] = 'OFF'
os.environ['CPL_LOG'] = 'OFF'
os.environ['GDAL_PAM_ENABLED'] = 'NO'

warnings.filterwarnings('ignore')
for logger_name in ['rasterio', 'rasterio._env', 'rasterio.env', 'rasterio._base']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

PROJECT_DIR  = '/content/drive/My Drive/Project_Folder'
DATASETS_DIR = os.path.join(PROJECT_DIR, 'Datasets')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'Results')
EXP_DIR      = os.path.join(RESULTS_DIR, 'exp_04c_residual_sr')
MODEL_DIR    = '/content/model/SEN2SR_LITE'
SPLIT_FILE   = os.path.join(RESULTS_DIR, 'test_scenes.json')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Cell A — fail fast on CPU
assert torch.cuda.is_available(), "Runtime > Change runtime type > GPU (T4), then re-run from Cell 1"
torch.backends.cudnn.benchmark = True
device = torch.device('cuda')
print(f"PyTorch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
print(f"Experiment output dir (isolated): {EXP_DIR}")"""
    cells.append(make_cell("code", c1))

    # Hardware check
    c1b = """!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
!free -h
!df -h /content"""
    cells.append(make_cell("code", c1b))

    # Step 2: Dependencies
    cells.append(make_cell("markdown", "## Step 2 — Install Dependencies (including `lpips`)"))
    c2 = """%pip install -q sen2sr mlstac rasterio safetensors torchvision pandas matplotlib scikit-image tqdm lpips"""
    cells.append(make_cell("code", c2))

    # Step 3: Reuse Existing Helpers & Losses
    cells.append(make_cell("markdown", "## Step 3 — Existing Canonical Data Helpers & Loss Modules"))
    c3 = """import rasterio
import numpy as np
from torch.nn.utils import spectral_norm

SEN2NAIP_LR_ORDER = [0, 1, 2, 3]
SEN2NAIP_HR_ORDER = [0, 1, 2, 3]
WS_LR_ORDER       = [3, 2, 1, 7]
WS_HR_ORDER       = [0, 1, 2, 3]

SCALE_S2    = 10000.0
SCALE_NAIP  = 255.0
SCALE_WS_HR = 4095.0

def read_raster(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

def match_radiometry_patch(hr_patch, lr_patch):
    matched = np.zeros_like(hr_patch)
    for c in range(hr_patch.shape[0]):
        src = hr_patch[c].ravel()
        tpl = lr_patch[c].ravel()
        s_quantiles = (np.argsort(np.argsort(src)) + 0.5) / len(src)
        matched[c] = np.quantile(tpl, s_quantiles).reshape(hr_patch[c].shape)
    return np.clip(matched, 0.0, 1.0)

def align_hr_to_4x(lr, hr):
    expected_h = lr.shape[1] * 4
    expected_w = lr.shape[2] * 4
    if hr.shape[1] != expected_h or hr.shape[2] != expected_w:
        hr_t = torch.from_numpy(hr).unsqueeze(0).float()
        hr = F.interpolate(hr_t, size=(expected_h, expected_w), mode='bilinear', align_corners=False).squeeze(0).numpy()
    return hr

def load_canonical_pair(lr_path, hr_path, is_worldstrat=False):
    lr_raw = read_raster(lr_path)
    hr_raw = read_raster(hr_path)
    if is_worldstrat:
        lr_4b = lr_raw[WS_LR_ORDER]
        hr_4b = hr_raw[WS_HR_ORDER]
        lr_scaled = np.clip(lr_4b, 0.0, 1.0)
        hr_scaled = np.clip(hr_4b / SCALE_WS_HR, 0.0, 1.0)
    else:
        lr_4b = lr_raw[SEN2NAIP_LR_ORDER]
        hr_4b = hr_raw[SEN2NAIP_HR_ORDER]
        lr_scaled = np.clip(lr_4b / SCALE_S2, 0.0, 1.0)
        hr_scaled = np.clip(hr_4b / SCALE_NAIP, 0.0, 1.0)
    hr_scaled = align_hr_to_4x(lr_scaled, hr_scaled)
    assert lr_scaled.shape[0] == 4 and hr_scaled.shape[0] == 4
    return lr_scaled, hr_scaled

class ShiftTolerantL1Loss(nn.Module):
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
    def __init__(self, urban_boost=1.0):
        super().__init__()
        self.urban_boost = urban_boost
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def forward(self, pred, target):
        # Weight non-vegetated pixels (NDVI < 0.25: buildings, roads, concrete, bare soil) higher
        with torch.no_grad():
            ndvi_t = (target[:, 3:4] - target[:, 0:1]) / (target[:, 3:4] + target[:, 0:1] + 1e-6)
            w_map = 1.0 + self.urban_boost * (ndvi_t < 0.25).float()
        loss = 0.0
        for c in range(pred.shape[1]):
            p, t = pred[:, c:c+1], target[:, c:c+1]
            gx_p, gy_p = F.conv2d(p, self.sx, padding=1), F.conv2d(p, self.sy, padding=1)
            gx_t, gy_t = F.conv2d(t, self.sx, padding=1), F.conv2d(t, self.sy, padding=1)
            loss += (w_map * (torch.abs(gx_p - gx_t) + torch.abs(gy_p - gy_t))).mean()
        return loss / pred.shape[1]

class MultiIndexSpectralLoss(nn.Module):
    \"\"\"
    4-band [R,G,B,NIR] spectral loss: NDVI + NDWI + Red-Green Built-up/Bare-Soil Proxy.
    Note: True SWIR NDBI requires B11 (20m), which is absent in 4-band RGB+NIR data.
    \"\"\"
    def __init__(self, eps=1e-6, weight_ndwi=0.5, weight_urban=0.25):
        super().__init__()
        self.eps = eps
        self.weight_ndwi = weight_ndwi
        self.weight_urban = weight_urban

    def forward(self, pred, target):
        red_p, green_p, nir_p = pred[:, 0], pred[:, 1], pred[:, 3]
        red_t, green_t, nir_t = target[:, 0], target[:, 1], target[:, 3]
        ndvi_p = (nir_p - red_p) / (nir_p + red_p + self.eps)
        ndvi_t = (nir_t - red_t) / (nir_t + red_t + self.eps)
        l_ndvi = F.l1_loss(ndvi_p, ndvi_t)
        ndwi_p = (green_p - nir_p) / (green_p + nir_p + self.eps)
        ndwi_t = (green_t - nir_t) / (green_t + nir_t + self.eps)
        l_ndwi = F.l1_loss(ndwi_p, ndwi_t)
        urb_p = (red_p - green_p) / (red_p + green_p + self.eps)
        urb_t = (red_t - green_t) / (red_t + green_t + self.eps)
        l_urb = F.l1_loss(urb_p, urb_t)
        return l_ndvi + (self.weight_ndwi * l_ndwi) + (self.weight_urban * l_urb)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        return self.net(x)

l1_loss_fn    = ShiftTolerantL1Loss(max_shift=1).to(device)
edge_loss_fn  = SobelEdgeLoss().to(device)
spec_loss_fn  = MultiIndexSpectralLoss().to(device)
gan_criterion = nn.MSELoss().to(device)
print("Canonical helpers and base loss modules ready.")"""
    cells.append(make_cell("code", c3))

    # Step 4: Scene Discovery & Split (Never tune on test_scenes!)
    cells.append(make_cell("markdown", "## Step 4 — Scene-Stratified Split (Read-Only `test_scenes.json`)"))
    c4 = """sen2naip_lr_files = sorted(glob.glob(os.path.join(DATASETS_DIR, 'sentinaiplow', '*.tif')))
all_scenes = []
for lr_p in sen2naip_lr_files:
    base = os.path.basename(lr_p)
    hr_p = os.path.join(DATASETS_DIR, 'sentinaiphigh', base.replace('_lr.tif', '_hr.tif'))
    if os.path.exists(hr_p):
        all_scenes.append({'id': base.replace('_lr.tif', ''), 'lr': lr_p, 'hr': hr_p, 'is_ws': False})

assert len(all_scenes) >= 5, f"Insufficient SEN2NAIP pairs in {DATASETS_DIR}!"

if os.path.exists(SPLIT_FILE):
    with open(SPLIT_FILE, 'r', encoding='utf-8') as f:
        split_meta = json.load(f)
    test_ids = set(split_meta.get('test_ids', []))
    test_scenes = [s for s in all_scenes if s['id'] in test_ids]
    rem = [s for s in all_scenes if s['id'] not in test_ids]
else:
    rng = random.Random(42)
    shuffled = list(all_scenes)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * 0.10))
    test_scenes = shuffled[-n_test:]
    rem = shuffled[:-n_test]

n_val = max(1, int(len(rem) * 0.10))
val_scenes   = rem[:n_val]
train_scenes = rem[n_val:]
print(f"Total SEN2NAIP: {len(all_scenes)} | Train={len(train_scenes)}, Val={len(val_scenes)}, Test={len(test_scenes)} (frozen)")"""
    cells.append(make_cell("code", c4))

    # Step 5: Cell B1 & B2 — Memory-Mapped Patch Cache
    cells.append(make_cell("markdown", "## Step 5 — Memory-Mapped Patch Cache on `/content/cache` (Cells B1 & B2)"))
    cb1 = """from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

CACHE_DIR = '/content/cache'
DRIVE_CACHE_DIR = os.path.join(PROJECT_DIR, 'Cache lite')
os.makedirs(CACHE_DIR, exist_ok=True)

# Fast-restore prebuilt cache from Google Drive if present (avoids rebuilding)
if os.path.isdir(DRIVE_CACHE_DIR):
    for fn in ['train_lr.npy', 'train_hr.npy', 'train_meta.json', 'val_lr.npy', 'val_hr.npy', 'val_meta.json']:
        src_f = os.path.join(DRIVE_CACHE_DIR, fn)
        dst_f = os.path.join(CACHE_DIR, fn)
        if os.path.exists(src_f) and not os.path.exists(dst_f):
            print(f"Restoring {fn} from Drive 'Cache lite'...")
            shutil.copy2(src_f, dst_f)

def build_cache(scenes, name, crops=8, p=64, seed=0):
    meta_path = f'{CACHE_DIR}/{name}_meta.json'
    lr_path   = f'{CACHE_DIR}/{name}_lr.npy'
    hr_path   = f'{CACHE_DIR}/{name}_hr.npy'
    if os.path.exists(meta_path) and os.path.exists(lr_path) and os.path.exists(hr_path):
        cnt = json.load(open(meta_path))['count']
        print(f"Reusing existing /content/cache/{name} ({cnt} patches).")
        return cnt

    rng = np.random.RandomState(seed)
    n = len(scenes) * crops
    lr_mm = np.lib.format.open_memmap(lr_path, mode='w+', dtype=np.float16, shape=(n, 4, p, p))
    hr_mm = np.lib.format.open_memmap(hr_path, mode='w+', dtype=np.float16, shape=(n, 4, p*4, p*4))
    meta, k = [], 0
    for s in tqdm(scenes, desc=f"Caching {name}"):
        try:
            lr, hr = load_canonical_pair(s['lr'], s['hr'], is_worldstrat=s['is_ws'])
            hr = align_hr_to_4x(lr, hr)
        except Exception:
            continue
        _, h, w = lr.shape
        if h < p or w < p:
            ph, pw = max(0, p-h), max(0, p-w)
            lr = np.pad(lr, ((0,0),(0,ph),(0,pw)), mode='reflect')
            hr = np.pad(hr, ((0,0),(0,ph*4),(0,pw*4)), mode='reflect')
            _, h, w = lr.shape
        for _ in range(crops):
            y = rng.randint(0, h-p+1); x = rng.randint(0, w-p+1)
            l = lr[:, y:y+p, x:x+p]
            hh = hr[:, y*4:(y+p)*4, x*4:(x+p)*4]
            hh = match_radiometry_patch(hh, l)
            bic = F.interpolate(torch.from_numpy(l.copy())[None], scale_factor=4, mode='bicubic', align_corners=False)[0].clamp(0,1).numpy()
            mse = float(((bic - hh)**2).mean()) + 1e-12
            q_psnr = 10 * math.log10(1.0 / mse)
            edge_e = float(np.abs(np.diff(hh, axis=2)).mean() + np.abs(np.diff(hh, axis=1)).mean())
            ndvi_hh = (hh[3] - hh[0]) / (hh[3] + hh[0] + 1e-6)
            builtup_frac = float(np.mean(ndvi_hh < 0.25))
            lr_mm[k] = l; hr_mm[k] = hh
            meta.append((s['id'], int(y), int(x), q_psnr, edge_e, builtup_frac)); k += 1
    lr_mm.flush(); hr_mm.flush()
    json.dump({'count': k, 'meta': meta}, open(meta_path, 'w'))
    return k

n_train = build_cache(train_scenes, 'train', crops=8, seed=0)
n_val   = build_cache(val_scenes,   'val',   crops=2, seed=1)
print(f"Cached patches -> Train: {n_train} | Val: {n_val}")

# Plot q_psnr pair-quality distribution so MIN_Q can be chosen from actual data
train_meta = json.load(open(f'{CACHE_DIR}/train_meta.json'))['meta']
q_vals = np.array([m[3] for m in train_meta])
print(f"Pair-quality q_psnr percentiles [5, 10, 20, 50, 80, 95]: {np.percentile(q_vals, [5, 10, 20, 50, 80, 95]).round(2)}")
plt.figure(figsize=(8, 3.5))
plt.hist(q_vals, bins=50, color='steelblue', edgecolor='white')
plt.title('SEN2NAIP Pair-Quality Distribution (Bicubic S2 vs HR q_psnr)')
plt.xlabel('q_psnr (dB)'); plt.ylabel('Patch Count'); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'pair_quality_q_psnr_hist.png'), dpi=150)
plt.show()"""
    cells.append(make_cell("code", cb1))

    cb2 = """from torch.utils.data import DataLoader, WeightedRandomSampler

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, name, is_train=True, min_q=None):
        meta = json.load(open(f'{CACHE_DIR}/{name}_meta.json'))
        self.k = meta['count']
        self.lr = np.load(f'{CACHE_DIR}/{name}_lr.npy', mmap_mode='r')
        self.hr = np.load(f'{CACHE_DIR}/{name}_hr.npy', mmap_mode='r')
        rows = meta['meta'][:self.k]
        q = np.array([m[3] for m in rows])
        self.idx = np.arange(self.k) if min_q is None else np.where(q >= min_q)[0]
        self.edge_e = np.array([m[4] if len(m) > 4 else 0.0 for m in rows])[self.idx]
        self.builtup = np.array([m[5] if len(m) > 5 else 0.0 for m in rows])[self.idx]
        self.is_train = is_train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        l = torch.from_numpy(np.array(self.lr[j], dtype=np.float32))
        h = torch.from_numpy(np.array(self.hr[j], dtype=np.float32))
        if self.is_train:
            if np.random.rand() > .5: l, h = l.flip(2), h.flip(2)
            if np.random.rand() > .5: l, h = l.flip(1), h.flip(1)
            k = np.random.randint(0, 4)
            if k: l, h = torch.rot90(l, k, (1, 2)), torch.rot90(h, k, (1, 2))
        return l.contiguous(), h.contiguous()

MIN_Q = None               # Set after inspecting q_psnr histogram (e.g. 20th percentile); None = all pairs
USE_URBAN_SAMPLER = False  # Disabled to prevent patch-repetition overfitting; Sobel urban_boost=1.0 still active
BATCH_SIZE = 16            # Fall back to 8 if T4 hits OOM
train_ds = CachedDataset('train', True, MIN_Q)
val_ds   = CachedDataset('val', False)

if USE_URBAN_SAMPLER:
    norm_edge = train_ds.edge_e / max(float(np.percentile(train_ds.edge_e, 95)), 1e-6)
    sample_w = np.clip(1.0 + 3.0 * norm_edge + 1.5 * train_ds.builtup, 0.5, 6.0)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w).double(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    print(f"Urban/Edge WeightedRandomSampler active | Weight range: [{sample_w.min():.2f}, {sample_w.max():.2f}], Mean: {sample_w.mean():.2f}")
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                              persistent_workers=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)} | Steps/epoch: {len(train_loader)}")"""
    cells.append(make_cell("code", cb2))

    # Step 6: Cell C — Generator, Multi-Layer VGG, Metrics
    cells.append(make_cell("markdown", "## Step 6 — `ResidualSR`, Multi-Layer RGB+NIR `VGGMulti`, & Validation Metrics (Cell C)"))
    cc = """import copy
import torchvision.models as models

class RDB(nn.Module):
    def __init__(s, nf=48, gc=24):
        super().__init__()
        s.c1 = nn.Conv2d(nf, gc, 3, 1, 1); s.c2 = nn.Conv2d(nf+gc, gc, 3, 1, 1)
        s.c3 = nn.Conv2d(nf+2*gc, gc, 3, 1, 1); s.c4 = nn.Conv2d(nf+3*gc, gc, 3, 1, 1)
        s.c5 = nn.Conv2d(nf+4*gc, nf, 3, 1, 1); s.a = nn.LeakyReLU(0.2, True)
    def forward(s, x):
        x1 = s.a(s.c1(x)); x2 = s.a(s.c2(torch.cat([x, x1], 1)))
        x3 = s.a(s.c3(torch.cat([x, x1, x2], 1))); x4 = s.a(s.c4(torch.cat([x, x1, x2, x3], 1)))
        return x + 0.2 * s.c5(torch.cat([x, x1, x2, x3, x4], 1))

class RRDB(nn.Module):
    def __init__(s, nf=48, gc=24):
        super().__init__(); s.b = nn.Sequential(RDB(nf, gc), RDB(nf, gc), RDB(nf, gc))
    def forward(s, x): return x + 0.2 * s.b(x)

class ResidualSR(nn.Module):
    \"\"\"out = bicubic(x) + learned residual. Zero-init output conv => starts exactly at bicubic. No PixelShuffle.\"\"\"
    def __init__(s, in_ch=4, nf=48, gc=24, nb=4):
        super().__init__()
        s.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
        s.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        s.body_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        s.up1 = nn.Conv2d(nf, nf, 3, 1, 1); s.up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        s.hr  = nn.Conv2d(nf, nf, 3, 1, 1); s.out = nn.Conv2d(nf, in_ch, 3, 1, 1)
        s.a = nn.LeakyReLU(0.2, True)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s, x):
        base = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        f = s.head(x); f = f + s.body_conv(s.body(f))
        f = s.a(s.up1(F.interpolate(f, scale_factor=2, mode='nearest')))
        f = s.a(s.up2(F.interpolate(f, scale_factor=2, mode='nearest')))
        return base + s.out(s.a(s.hr(f)))

class VGGMulti(nn.Module):
    \"\"\"Multi-layer VGG19 loss on stretched true colour [R,G,B] and NIR false colour [NIR,R,G].\"\"\"
    def __init__(s):
        super().__init__()
        v = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:27].eval()
        for p in v.parameters(): p.requires_grad = False
        s.v = v; s.ids = {3: 0.1, 8: 0.1, 17: 1.0, 26: 1.0}
        s.register_buffer('m', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer('sd', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def feats(s, x):
        x = ((x / 0.3).clamp(0, 1) - s.m) / s.sd; out = {}
        for i, l in enumerate(s.v):
            x = l(x)
            if i in s.ids: out[i] = x
        return out
    def forward(s, p, t):
        loss = 0.
        for sel in ([0, 1, 2], [3, 0, 1]):
            fp, ft = s.feats(p[:, sel]), s.feats(t[:, sel])
            loss = loss + sum(w * F.l1_loss(fp[i], ft[i]) for i, w in s.ids.items())
        return loss / 2

def sam_deg_t(p, t, eps=1e-6):
    d = (p * t).sum(1); n = p.norm(dim=1) * t.norm(dim=1) + eps
    return torch.rad2deg(torch.acos((d / n).clamp(-1, 1))).mean()

def grad_energy(x):
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:] - x[:, :, :-1]).abs().mean()

vgg_ml = VGGMulti().to(device)
lrcons = lambda sr, lr: F.l1_loss(F.avg_pool2d(sr, 4), lr)

@torch.no_grad()
def validate(G):
    G.eval(); l1 = sam = hf = n = 0.0
    for lr_v, hr_v in val_loader:
        lr_v, hr_v = lr_v.to(device), hr_v.to(device)
        sr = G(lr_v).clamp(0, 1)
        b = lr_v.size(0); n += b
        l1  += F.l1_loss(sr, hr_v).item() * b
        sam += sam_deg_t(sr, hr_v).item() * b
        hf  += (grad_energy(sr) / grad_energy(hr_v)).item() * b
    return l1 / n, sam / n, hf / n

print("ResidualSR, VGGMulti, and validation metrics compiled.")"""
    cells.append(make_cell("code", cc))

    # Week 1 Diagnostic Cell (Official Baseline vs Compound vs Bicubic + Loss Magnitudes)
    cells.append(make_cell("markdown", "## Step 6b — Week 1 Diagnostic Check (Official `mlstac` Baseline vs Raw `CNNSR` & Loss Magnitudes)"))
    cdiag = """import mlstac, safetensors.torch, urllib.request
from sen2sr.models.opensr_baseline.cnn import CNNSR

# 1. Load existing Compound model (read-only) for diagnostic comparison
BEST_COMPOUND = os.path.join(RESULTS_DIR, 'sen2sr_lite_compound_best.pth')
compound_model = CNNSR(in_channels=4, out_channels=4, feature_channels=24, upscale=4, bias=True, train_mode=False, num_blocks=6).to(device)
if os.path.exists(BEST_COMPOUND):
    ck = torch.load(BEST_COMPOUND, map_location=device)
    compound_model.load_state_dict(ck.get('model_state_dict', ck))
compound_model.eval()

# 2. Load raw CNNSR baseline
weights_path = os.path.join(MODEL_DIR, 'sr_model.safetensor')
if not os.path.exists(weights_path):
    try:
        mlstac.download(file='https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/main/mlm.json', output_dir=MODEL_DIR)
    except Exception:
        urllib.request.urlretrieve('https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/main/sr_model.safetensor', weights_path)

baseline_model = CNNSR(in_channels=4, out_channels=4, feature_channels=24, upscale=4, bias=True, train_mode=False, num_blocks=6).to(device)
baseline_model.load_state_dict(safetensors.torch.load_file(weights_path))
baseline_model.eval()

# 3. Log GPU metrics & Compound loss magnitudes FIRST (before touching compiled_model)
lr_v, hr_v = next(iter(val_loader))
lr_v, hr_v = lr_v.to(device), hr_v.to(device)
with torch.no_grad():
    bic_v = F.interpolate(lr_v, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
    sr_c  = compound_model(lr_v).clamp(0, 1)
    sr_b  = baseline_model(lr_v).clamp(0, 1)
    print("Val batch L1/SAM -> Bicubic:", round(F.l1_loss(bic_v, hr_v).item(), 4), round(sam_deg_t(bic_v, hr_v).item(), 3),
          "| Raw CNNSR:", round(F.l1_loss(sr_b, hr_v).item(), 4), round(sam_deg_t(sr_b, hr_v).item(), 3),
          "| Compound:", round(F.l1_loss(sr_c, hr_v).item(), 4), round(sam_deg_t(sr_c, hr_v).item(), 3))

    print("Compound loss term magnitudes:", {k: round(float(v), 5) for k, v in dict(
        l1=l1_loss_fn(sr_c, hr_v), edge=edge_loss_fn(sr_c, hr_v), spec=spec_loss_fn(sr_c, hr_v),
        cons=lrcons(sr_c, lr_v), vgg=vgg_ml(sr_c, hr_v)).items()})

# 4. Safely test official mlstac compiled_model on CPU (never poisons CUDA if shape asserts!)
try:
    ml_obj = mlstac.load(os.path.join(MODEL_DIR, 'mlm.json'))
    official_cpu = ml_obj.compiled_model(device='cpu')
    lr_cpu, hr_cpu = lr_v[:2].cpu(), hr_v[:2].cpu()
    with torch.no_grad():
        try:
            sr_off = official_cpu(lr_cpu).clamp(0, 1)
            print("Official compiled_model (64x64) -> L1:", round(F.l1_loss(sr_off, hr_cpu).item(), 4), "SAM:", round(sam_deg_t(sr_off, hr_cpu).item(), 3))
        except Exception as e64:
            print(f"Official compiled_model 64x64 note: {e64}")
except Exception as e:
    print(f"Official mlstac.compiled_model check note: {e}")"""
    cells.append(make_cell("code", cdiag))

    # Step 7: Cell D1 — Phase 1 Pixel/Spectral Pretraining (Resumable)
    cells.append(make_cell("markdown", "## Step 7 — Phase 1: `ResidualSR` Pixel & Spectral Pretraining (Cell D1, Resumable)"))
    cd1 = """SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

PH1_EPOCHS = 6
PH2_EPOCHS = 12
SAM_MAX    = 1.70
LR_P1      = 5e-5
LR_P2      = 5e-5
LAM = dict(l1=1.0, edge=0.2, spec=0.15, cons=0.5, vgg=0.04, adv=0.015)

# Save full experiment config (Rule 6)
exp_config = dict(seed=SEED, min_q=MIN_Q, use_urban_sampler=USE_URBAN_SAMPLER,
                  urban_boost=edge_loss_fn.urban_boost, weight_urban=spec_loss_fn.weight_urban,
                  batch_size=BATCH_SIZE, ph1_epochs=PH1_EPOCHS, ph2_epochs=PH2_EPOCHS,
                  lr_p1=LR_P1, lr_p2=LR_P2, sam_max=SAM_MAX, lam=LAM)
with open(os.path.join(EXP_DIR, 'exp_config.json'), 'w') as f:
    json.dump(exp_config, f, indent=2)

PRE   = os.path.join(EXP_DIR, 'resSR_pre_best.pth')
P1_CK = os.path.join(EXP_DIR, 'resSR_p1_v2_state.pth')

G = ResidualSR().to(device)
ema = copy.deepcopy(G).eval()
print(f"ResidualSR params: {sum(p.numel() for p in G.parameters()):,}")

def ema_update(d=0.999):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), G.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)

def pix_loss(sr, lr, hr):
    t = dict(l1=l1_loss_fn(sr, hr), edge=edge_loss_fn(sr, hr), spec=spec_loss_fn(sr, hr), cons=lrcons(sr, lr))
    return sum(LAM[k] * v for k, v in t.items()), t

optG = torch.optim.AdamW(G.parameters(), lr=LR_P1, betas=(0.9, 0.99), weight_decay=1e-4)
schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, PH1_EPOCHS * len(train_loader), eta_min=5e-6)

start, best1 = 1, 1e9
if os.path.exists(P1_CK):
    ck = torch.load(P1_CK, map_location=device)
    G.load_state_dict(ck['G']); ema.load_state_dict(ck['ema'])
    optG.load_state_dict(ck['optG']); schG.load_state_dict(ck['schG'])
    start, best1 = ck['ep'] + 1, ck['best1']
    print('Resumed Phase 1 at epoch', start)
else:
    l1_0, sam_0, hf_0 = validate(ema)
    best1 = l1_0
    torch.save(ema.state_dict(), PRE)
    print(f"[P1 00/{PH1_EPOCHS:02d} Bicubic Init] val L1 {l1_0:.4f} SAM {sam_0:.3f} HF {hf_0:.3f}")

for ep in range(start, PH1_EPOCHS + 1):
    G.train(); agg = {}
    for lr_b, hr_b in train_loader:
        lr_b, hr_b = lr_b.to(device), hr_b.to(device)
        loss, t = pix_loss(G(lr_b), lr_b, hr_b)
        optG.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        optG.step(); schG.step(); ema_update()
        for k, v in t.items():
            agg[k] = agg.get(k, 0.0) + v.item() / len(train_loader)
    l1v, samv, hfv = validate(ema)
    tag = ''
    if l1v < best1:
        best1 = l1v
        torch.save(ema.state_dict(), PRE)
        tag = '[BEST]'
    print(f"[P1 {ep:02d}/{PH1_EPOCHS:02d}] " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()) +
          f" | val L1 {l1v:.4f} SAM {samv:.3f} HF {hfv:.3f} {tag}")
    torch.save({'G': G.state_dict(), 'ema': ema.state_dict(), 'optG': optG.state_dict(),
                'schG': schG.state_dict(), 'ep': ep, 'best1': best1}, P1_CK)"""
    cells.append(make_cell("code", cd1))

    # Step 8: Cell D2 — Phase 2 GAN Fine-Tune (Resumable)
    cells.append(make_cell("markdown", "## Step 8 — Phase 2: Sharpness-Guarded LSGAN Fine-Tuning (Cell D2, Resumable)"))
    cd2 = """BEST_GAN = os.path.join(EXP_DIR, 'resSR_gan_best.pth')
FALLBACK = os.path.join(EXP_DIR, 'resSR_gan_lowsam.pth')
P2_CK    = os.path.join(EXP_DIR, 'resSR_p2_v2_state.pth')

net_D = PatchGANDiscriminator(4, 64).to(device)
optG = torch.optim.AdamW(G.parameters(), lr=LR_P2, betas=(0.9, 0.99), weight_decay=1e-4)
optD = torch.optim.AdamW(net_D.parameters(), lr=LR_P2, betas=(0.9, 0.99), weight_decay=1e-4)
tot = PH2_EPOCHS * len(train_loader)
schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, tot, eta_min=1e-6)
schD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, tot, eta_min=1e-6)

start, step, best_score, best_sam = 1, 0, 1e9, 1e9
if os.path.exists(P2_CK):
    ck = torch.load(P2_CK, map_location=device)
    G.load_state_dict(ck['G']); ema.load_state_dict(ck['ema']); net_D.load_state_dict(ck['D'])
    optG.load_state_dict(ck['optG']); optD.load_state_dict(ck['optD'])
    schG.load_state_dict(ck['schG']); schD.load_state_dict(ck['schD'])
    start, step, best_score, best_sam = ck['ep'] + 1, ck['step'], ck['best_score'], ck['best_sam']
    print('Resumed Phase 2 at epoch', start)
else:
    G.load_state_dict(torch.load(PRE, map_location=device))
    ema.load_state_dict(torch.load(PRE, map_location=device))

warm = len(train_loader) * 2
for ep in range(start, PH2_EPOCHS + 1):
    G.train(); net_D.train(); agg = {}
    for lr_b, hr_b in train_loader:
        lr_b, hr_b = lr_b.to(device), hr_b.to(device); step += 1
        with torch.no_grad():
            fake = G(lr_b)
        dr, df = net_D(hr_b), net_D(fake)
        lossD = 0.5 * (gan_criterion(dr, torch.ones_like(dr)) + gan_criterion(df, torch.zeros_like(df)))
        optD.zero_grad(); lossD.backward()
        nn.utils.clip_grad_norm_(net_D.parameters(), 1.0)
        optD.step(); schD.step()

        sr = G(lr_b); dg = net_D(sr)
        lp, t = pix_loss(sr, lr_b, hr_b)
        t['vgg'] = vgg_ml(sr, hr_b)
        t['adv'] = gan_criterion(dg, torch.ones_like(dg))
        ramp = min(1.0, step / warm)
        lossG = lp + ramp * (LAM['vgg'] * t['vgg'] + LAM['adv'] * t['adv'])
        optG.zero_grad(); lossG.backward()
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        optG.step(); schG.step(); ema_update()
        for k, v in t.items():
            agg[k] = agg.get(k, 0.0) + v.item() / len(train_loader)

    l1v, samv, hfv = validate(ema)
    score = abs(hfv - 1.0) if samv <= SAM_MAX else 1e9
    tag = ''
    if samv < best_sam:
        best_sam = samv
        torch.save({'ema': ema.state_dict()}, FALLBACK)
    if score < best_score:
        best_score = score
        torch.save({'ema': ema.state_dict(), 'ep': ep, 'sam': samv, 'hf': hfv}, BEST_GAN)
        tag = '[BEST]'
    print(f"[P2 {ep:02d}/{PH2_EPOCHS:02d}] " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()) +
          f" | val L1 {l1v:.4f} SAM {samv:.3f} HF {hfv:.3f} {tag}")
    torch.save({'G': G.state_dict(), 'ema': ema.state_dict(), 'D': net_D.state_dict(),
                'optG': optG.state_dict(), 'optD': optD.state_dict(), 'schG': schG.state_dict(),
                'schD': schD.state_dict(), 'ep': ep, 'step': step, 'best_score': best_score, 'best_sam': best_sam}, P2_CK)

if not os.path.exists(BEST_GAN):
    import shutil
    shutil.copy(FALLBACK, BEST_GAN)
    print("No epoch met the SAM guard; saved lowest-SAM epoch. Lower LAM['adv']/LAM['vgg'].")

def load_fidelity_model():
    m = ResidualSR().to(device)
    m.load_state_dict(torch.load(PRE, map_location=device))
    return m.eval()

def load_gan_model():
    m = ResidualSR().to(device)
    m.load_state_dict(torch.load(BEST_GAN, map_location=device)['ema'])
    return m.eval()

fidelity_model = load_fidelity_model()
gan_model      = load_gan_model()
print("Loaded ResidualSR fidelity & sharpness-guarded GAN models.")"""
    cells.append(make_cell("code", cd2))

    # Step 9: Cells E & F — Evaluation (HF-ratio, LPIPS, Alpha-Blend Trade-Off)
    cells.append(make_cell("markdown", "## Step 9 — Evaluation (`HF-ratio`, `LPIPS`, Alpha-Blend Trade-Off & Frozen Test Gate) (Cells E & F)"))
    cef = """import pandas as pd
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

def compute_all_metrics(pred_4b, gt_4b, eps=1e-6):
    psnr_val = float(np.mean([psnr_fn(gt_4b[c], pred_4b[c], data_range=1.0) for c in range(4)]))
    ssim_val = float(np.mean([ssim_fn(gt_4b[c], pred_4b[c], data_range=1.0) for c in range(4)]))
    rmse_val = float(np.sqrt(np.mean((gt_4b - pred_4b) ** 2)))
    mae_val  = float(np.mean(np.abs(gt_4b - pred_4b)))
    gx = lambda a: (np.abs(np.diff(a, axis=2)).mean() + np.abs(np.diff(a, axis=1)).mean())
    gx_p, gy_p = np.diff(pred_4b, axis=2), np.diff(pred_4b, axis=1)
    gx_g, gy_g = np.diff(gt_4b, axis=2), np.diff(gt_4b, axis=1)
    grad_mae = float(0.5 * (np.mean(np.abs(gx_p - gx_g)) + np.mean(np.abs(gy_p - gy_g))))
    hf_ratio = float(gx(pred_4b) / max(gx(gt_4b), eps))

    dot = np.sum(pred_4b * gt_4b, axis=0)
    sam_deg = float(np.mean(np.degrees(np.arccos(np.clip(dot / (np.linalg.norm(pred_4b, axis=0) * np.linalg.norm(gt_4b, axis=0) + eps), -1.0, 1.0)))))
    ndvi_mae = float(np.mean(np.abs((pred_4b[3]-pred_4b[0])/(pred_4b[3]+pred_4b[0]+eps) - (gt_4b[3]-gt_4b[0])/(gt_4b[3]+gt_4b[0]+eps))))
    ndwi_mae = float(np.mean(np.abs((pred_4b[1]-pred_4b[3])/(pred_4b[1]+pred_4b[3]+eps) - (gt_4b[1]-gt_4b[3])/(gt_4b[1]+gt_4b[3]+eps))))
    urb_mae  = float(np.mean(np.abs((pred_4b[0]-pred_4b[1])/(pred_4b[0]+pred_4b[1]+eps) - (gt_4b[0]-gt_4b[1])/(gt_4b[0]+gt_4b[1]+eps))))

    # LPIPS on stretched RGB rescaled to [-1, 1]
    p_rgb = np.clip(pred_4b[:3] / 0.3, 0.0, 1.0) * 2.0 - 1.0
    g_rgb = np.clip(gt_4b[:3] / 0.3, 0.0, 1.0) * 2.0 - 1.0
    with torch.no_grad():
        lp_val = float(lpips_fn(torch.from_numpy(p_rgb)[None].float().to(device),
                                torch.from_numpy(g_rgb)[None].float().to(device)).item())

    return {
        'PSNR (dB)': round(psnr_val, 3), 'SSIM': round(ssim_val, 4), 'RMSE': round(rmse_val, 4),
        'L1 MAE': round(mae_val, 4), 'Edge MAE': round(grad_mae, 4), 'HF-ratio (->1)': round(hf_ratio, 3),
        'LPIPS (↓)': round(lp_val, 4), 'SAM (deg)': round(sam_deg, 3),
        'NDVI MAE': round(ndvi_mae, 4), 'NDWI MAE': round(ndwi_mae, 4), 'BuiltUp Proxy MAE': round(urb_mae, 4),
    }

# Cell F — Fidelity / Sharpness Dial (on validation split)
def blended_pred(x, alpha=0.5):
    with torch.no_grad():
        return ((1.0 - alpha) * fidelity_model(x) + alpha * gan_model(x)).clamp(0, 1)

val_eval_rows = []
N_VAL_BLEND = min(100, len(val_ds))
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    recs = []
    for i in range(N_VAL_BLEND):
        lr_i, hr_i = val_ds[i]
        sr_i = blended_pred(lr_i[None].to(device), alpha=alpha).squeeze(0).cpu().numpy()
        recs.append(compute_all_metrics(sr_i, hr_i.numpy()))
    m = pd.DataFrame(recs).mean().to_dict()
    val_eval_rows.append({'Model / Alpha': f"Blend alpha={alpha:.2f}", **{k: round(v, 4) for k, v in m.items()}})

blend_df = pd.DataFrame(val_eval_rows)
blend_csv = os.path.join(EXP_DIR, 'val_alpha_blend_tradeoff.csv')
blend_df.to_csv(blend_csv, index=False)
print("=== Validation Fidelity-vs-Sharpness Trade-off Curve ===")
display(blend_df)

# Frozen Test Split Evaluation (Set True ONLY once at the end per Section 12 & 16!)
RUN_FINAL_TEST_EVAL = False
if RUN_FINAL_TEST_EVAL:
    test_rows = []
    models_eval = {
        'Bicubic 4x': lambda x: F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1),
        'Pretrained CNNSR': lambda x: baseline_model(x).clamp(0, 1),
        'Compound CNNSR': lambda x: compound_model(x).clamp(0, 1),
        'ResidualSR Fidelity (Ours)': lambda x: fidelity_model(x).clamp(0, 1),
        'ResidualSR LSGAN (Ours)': lambda x: gan_model(x).clamp(0, 1),
    }
    agg = {k: [] for k in models_eval}
    for sc in test_scenes:
        lr_f, hr_f = load_canonical_pair(sc['lr'], sc['hr'], is_worldstrat=sc['is_ws'])
        cy, cx = lr_f.shape[1] // 2, lr_f.shape[2] // 2
        if cy < 32 or cx < 32: continue
        lr_p = lr_f[:, cy-32:cy+32, cx-32:cx+32]
        hr_p = match_radiometry_patch(hr_f[:, (cy-32)*4:(cy+32)*4, (cx-32)*4:(cx+32)*4], lr_p)
        inp = torch.from_numpy(lr_p)[None].to(device)
        with torch.no_grad():
            for name, fn in models_eval.items():
                agg[name].append(compute_all_metrics(fn(inp).squeeze(0).cpu().numpy(), hr_p))
    for name, recs in agg.items():
        df_m = pd.DataFrame(recs)
        test_rows.append({'Model': name, **{c: f"{df_m[c].mean():.4f} ± {df_m[c].std():.4f}" for c in df_m.columns}})
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(os.path.join(EXP_DIR, 'final_test_281_summary.csv'), index=False)
    display(test_df)"""
    cells.append(make_cell("code", cef))

    # Step 10: India No-Reference Checks
    cells.append(make_cell("markdown", "## Step 10 — Indian Copernicus No-Reference Checks (LR-Consistency, NDVI & SAM Stability)"))
    cind = """import zipfile, io, re
from PIL import Image

def check_copernicus_noref(zip_name):
    cands = [os.path.join(d, zip_name) for d in [DATASETS_DIR, PROJECT_DIR, RESULTS_DIR]]
    zp = next((p for p in cands if os.path.exists(p)), None)
    if zp is None:
        print(f"Skipping {zip_name} (not found).")
        return
    raw = {}
    with zipfile.ZipFile(zp, 'r') as z:
        for nm in z.namelist():
            for b in ['B04', 'B03', 'B02', 'B08']:
                if re.search(r'(^|[-_.])' + b + r'([-._]|$)', os.path.basename(nm).upper()):
                    with z.open(nm) as f:
                        raw[b] = np.array(Image.open(io.BytesIO(f.read())), dtype=np.float32)
    cube = np.stack([raw['B04'], raw['B03'], raw['B02'], raw['B08']], axis=0)
    cal = np.zeros_like(cube, dtype=np.float32)
    for c in range(4):
        ch = cube[c]
        r = (np.maximum(0.0, ch - 1000.0) / 10000.0) if (ch.max() > 255.0 and ch.min() >= 800.0) else (ch / 10000.0 if ch.max() > 255.0 else (ch / 255.0 if ch.max() > 2.0 else ch))
        cal[c] = np.clip(r, 0.0, 1.0)
    inp = torch.from_numpy(cal)[None].to(device)
    with torch.no_grad():
        for name, m in [('ResidualSR Fidelity', fidelity_model), ('ResidualSR LSGAN', gan_model)]:
            sr = m(inp).clamp(0, 1)
            lr_down = F.avg_pool2d(sr, 4)
            lrc = float(F.l1_loss(lr_down, inp).item())
            sam = float(sam_deg_t(lr_down, inp).item())
            ndvi_in = (inp[:, 3] - inp[:, 0]) / (inp[:, 3] + inp[:, 0] + 1e-6)
            ndvi_dn = (lr_down[:, 3] - lr_down[:, 0]) / (lr_down[:, 3] + lr_down[:, 0] + 1e-6)
            ndvi_err = float(F.l1_loss(ndvi_dn, ndvi_in).item())
            print(f"[{zip_name}] {name:20s} -> LR-cons L1: {lrc:.4f} | Downsampled SAM: {sam:.3f} deg | NDVI drift: {ndvi_err:.4f}")

for zf in ['Browser_images_new1.zip', 'Parade ground.zip', 'Jollygrant.zip']:
    check_copernicus_noref(zf)"""
    cells.append(make_cell("code", cind))

    nb = create_notebook(cells)
    for idx, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            code_str = "".join(c['source'])
            clean_lines = [l for l in code_str.split("\n") if not l.strip().startswith(('%pip', '!', '%'))]
            ast.parse("\n".join(clean_lines))

    out_path = 'notebooks/04_finetuning_experiments/04c_Lite_ResidualSR_Cache_And_LSGAN.ipynb'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Validated and saved: {out_path} ({len(nb['cells'])} cells)")

    gdrive_path = 'G:/My Drive/Project_Folder/Notebooks/04_finetuning_experiments/04c_Lite_ResidualSR_Cache_And_LSGAN.ipynb'
    if os.path.exists(os.path.dirname(gdrive_path)):
        with open(gdrive_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Synced to Google Drive: {gdrive_path}")

if __name__ == '__main__':
    build_04c_notebook()
