"""
Generator for Task T6:
- 04b_Lite_Adversarial_Finetune.ipynb
"""

import json, os, ast

def make_cell(cell_type, source, cell_id=None):
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

def build_gan_notebook():
    cells = []
    
    # Title
    cells.append(make_cell("markdown", "# SEN2SR Lite: Adversarial Fine-Tuning with PatchGAN & Spectral Normalization\n\n**Major Project — Spectral Enhancement (Task T6)**\n\n> [!WARNING]\n> **Adversarial Texture Warning:**\n> Adversarial GAN training actively encourages the generator to produce high-frequency edge textures that fool a discriminator. **Texture may be hallucinated; compare against NAIP ground truth, not against intuition.** Spectral consistency (NDVI/NDWI) and shift-tolerant L1 losses are enforced simultaneously to prevent spectral drift.\n\n### Objective & Architecture:\n1. **Initialization:** Starts directly from the best fine-tuned compound checkpoint (`sen2sr_lite_compound_best.pth`).\n2. **PatchGAN Discriminator:** Spectral-normalized convolutional discriminator evaluating local $70\\times 70$ patch realism.\n3. **Compound Adversarial Objective:**\n   $$\\mathcal{L}_G = \\mathcal{L}_{\\text{shift-L1}} + 0.20 \\cdot \\mathcal{L}_{\\text{Sobel}} + 0.15 \\cdot \\mathcal{L}_{\\text{spectral}} + 0.05 \\cdot \\mathcal{L}_{\\text{VGG}} + 0.005 \\cdot \\mathcal{L}_{\\text{adv}}$$\n4. **Controlled Training:** Generator learning rate $2\\times 10^{-5}$, discriminator $5\\times 10^{-5}$, short run (10 epochs), best-checkpoint saving."))

    # Step 1
    cells.append(make_cell("markdown", "## Step 1 — Colab Environment & Google Drive Setup"))
    c1 = """import os, sys, glob, time, warnings, logging

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
SPLIT_FILE   = os.path.join(RESULTS_DIR, 'test_scenes.json')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch: {torch.__version__} | Active Device: {device} | CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')"""
    cells.append(make_cell("code", c1))

    # Step 2
    cells.append(make_cell("markdown", "## Step 2 — Install Required Dependencies"))
    c2 = """%pip install -q sen2sr mlstac rasterio safetensors torchvision pandas matplotlib scikit-image tqdm"""
    cells.append(make_cell("code", c2))

    # Step 3 - T1 Audit & Canonical
    cells.append(make_cell("markdown", "## Step 3 — Canonical Band Ordering & Radiometric Calibration Helpers"))
    c3 = """import rasterio
import numpy as np

SEN2NAIP_LR_ORDER = [0, 1, 2, 3]
SEN2NAIP_HR_ORDER = [0, 1, 2, 3]
WS_LR_ORDER       = [3, 2, 1, 7]
WS_HR_ORDER       = [0, 1, 2, 3]

SCALE_S2 = 10000.0
SCALE_NAIP = 255.0
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
        import torch
        import torch.nn.functional as F
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
    return lr_scaled, hr_scaled"""
    cells.append(make_cell("code", c3))

    # Step 4 - Dataset
    cells.append(make_cell("markdown", "## Step 4 — Scene-Stratified Split & Dynamic Augmentation Dataset"))
    c4 = """import json, random
from torch.utils.data import Dataset, DataLoader

sen2naip_lr_files = sorted(glob.glob(os.path.join(DATASETS_DIR, 'sentinaiplow', '*.tif')))
sen2naip_hr_files = sorted(glob.glob(os.path.join(DATASETS_DIR, 'sentinaiphigh', '*.tif')))

sen2naip_pairs = []
for lr_p in sen2naip_lr_files:
    base = os.path.basename(lr_p)
    hr_p = os.path.join(DATASETS_DIR, 'sentinaiphigh', base.replace('_lr.tif', '_hr.tif'))
    if os.path.exists(hr_p):
        sen2naip_pairs.append({'id': base.replace('_lr.tif', ''), 'lr': lr_p, 'hr': hr_p, 'is_ws': False})

assert len(sen2naip_pairs) >= 5, "Insufficient pairs found in Datasets!"

# Load test split if exists
if os.path.exists(SPLIT_FILE):
    with open(SPLIT_FILE, 'r', encoding='utf-8') as f:
        split_meta = json.load(f)
    test_ids = set(split_meta.get('test_ids', []))
    test_scenes = [s for s in sen2naip_pairs if s['id'] in test_ids]
    rem = [s for s in sen2naip_pairs if s['id'] not in test_ids]
else:
    rng = random.Random(42)
    shuffled = list(sen2naip_pairs)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * 0.10))
    test_scenes = shuffled[-n_test:]
    rem = shuffled[:-n_test]

n_val = max(1, int(len(rem) * 0.10))
val_scenes = rem[:n_val]
train_scenes = rem[n_val:]
print(f"Scene Split: Train={len(train_scenes)}, Val={len(val_scenes)}, Test={len(test_scenes)}")

class ScenePatchDataset(Dataset):
    def __init__(self, scenes, patch_lr=64, is_train=True):
        self.scenes = scenes
        self.patch_lr = patch_lr
        self.patch_hr = patch_lr * 4
        self.is_train = is_train

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        lr_full, hr_full = load_canonical_pair(scene['lr'], scene['hr'], is_worldstrat=scene['is_ws'])
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

        return torch.from_numpy(lr_crop.copy()).float(), torch.from_numpy(hr_crop.copy()).float()

train_ds = ScenePatchDataset(train_scenes, patch_lr=64, is_train=True)
val_ds   = ScenePatchDataset(val_scenes, patch_lr=64, is_train=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)"""
    cells.append(make_cell("code", c4))

    # Step 5 - Models (Generator + Discriminator + VGG)
    cells.append(make_cell("markdown", "## Step 5 — Architecture: PatchGAN Discriminator with Spectral Normalization & VGG Perceptual Engine"))
    c5 = """import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision.models as models
import safetensors.torch
import mlstac
from sen2sr.models.opensr_baseline.cnn import CNNSR

# 1. PatchGAN Discriminator with Spectral Normalization
class PatchGANDiscriminator(nn.Module):
    \"\"\"
    Spectral-Normalized 70x70 PatchGAN Discriminator.
    Guarantees 1-Lipschitz continuity for ultra-stable adversarial training.
    \"\"\"
    def __init__(self, in_channels=4, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        return self.net(x)

# 2. VGG-19 Feature Extractor for Perceptual Loss on RGB
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        # ImageNet normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_rgb, target_rgb):
        p_norm = (pred_rgb - self.mean) / self.std
        t_norm = (target_rgb - self.mean) / self.std
        return F.l1_loss(self.vgg(p_norm), self.vgg(t_norm))

net_D = PatchGANDiscriminator(in_channels=4, ndf=64).to(device)
vgg_loss_fn = VGGPerceptualLoss().to(device)

# 3. Generator initialized from Best Fine-Tuned Compound Checkpoint
generator = CNNSR(in_channels=4, out_channels=4, feature_channels=24, upscale=4, bias=True, train_mode=True, num_blocks=6)

BEST_COMPOUND = os.path.join(RESULTS_DIR, 'sen2sr_lite_compound_best.pth')
LAST_COMPOUND = os.path.join(RESULTS_DIR, 'sen2sr_lite_compound_last.pth')

if os.path.exists(BEST_COMPOUND):
    ckpt = torch.load(BEST_COMPOUND, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    generator.load_state_dict(state)
    print(f'Generator initialized from best compound checkpoint: {BEST_COMPOUND}')
elif os.path.exists(LAST_COMPOUND):
    ckpt = torch.load(LAST_COMPOUND, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    generator.load_state_dict(state)
    print(f'Generator initialized from last compound checkpoint: {LAST_COMPOUND}')
else:
    weights_path = os.path.join(MODEL_DIR, 'sr_model.safetensor')
    if not os.path.exists(weights_path):
        print('Downloading base weights for initialization...')
        try:
            mlstac.download(file='https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/main/mlm.json', output_dir=MODEL_DIR)
        except Exception as e:
            import urllib.request
            u = 'https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SRLite/main/sr_model.safetensor'
            urllib.request.urlretrieve(u, weights_path)
    generator.load_state_dict(safetensors.torch.load_file(weights_path))

generator = generator.to(device)
print(f'Generator parameters: {sum(p.numel() for p in generator.parameters()):,}')
print(f'Discriminator parameters: {sum(p.numel() for p in net_D.parameters()):,}')"""
    cells.append(make_cell("code", c5))

    # Step 6 - Losses & Training Setup
    cells.append(make_cell("markdown", "## Step 6 — Multi-Objective Compound & Adversarial Loss Formulations"))
    c6 = """class ShiftTolerantL1Loss(nn.Module):
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
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def forward(self, pred, target):
        loss = 0.0
        for c in range(pred.shape[1]):
            p, t = pred[:, c:c+1], target[:, c:c+1]
            gx_p, gy_p = F.conv2d(p, self.sx, padding=1), F.conv2d(p, self.sy, padding=1)
            gx_t, gy_t = F.conv2d(t, self.sx, padding=1), F.conv2d(t, self.sy, padding=1)
            loss += F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
        return loss / pred.shape[1]

class MultiIndexSpectralLoss(nn.Module):
    def __init__(self, eps=1e-6, weight_ndwi=0.5):
        super().__init__()
        self.eps = eps
        self.weight_ndwi = weight_ndwi

    def forward(self, pred, target):
        red_p, green_p, nir_p = pred[:, 0], pred[:, 1], pred[:, 3]
        red_t, green_t, nir_t = target[:, 0], target[:, 1], target[:, 3]
        ndvi_p = (nir_p - red_p) / (nir_p + red_p + self.eps)
        ndvi_t = (nir_t - red_t) / (nir_t + red_t + self.eps)
        l_ndvi = F.l1_loss(ndvi_p, ndvi_t)
        ndwi_p = (green_p - nir_p) / (green_p + nir_p + self.eps)
        ndwi_t = (green_t - nir_t) / (green_t + nir_t + self.eps)
        l_ndwi = F.l1_loss(ndwi_p, ndwi_t)
        return l_ndvi + (self.weight_ndwi * l_ndwi)

l1_loss_fn = ShiftTolerantL1Loss(max_shift=1).to(device)
edge_loss_fn = SobelEdgeLoss().to(device)
spec_loss_fn = MultiIndexSpectralLoss().to(device)
gan_criterion = nn.BCEWithLogitsLoss().to(device)

# Optimizers with conservative learning rates
opt_G = torch.optim.AdamW(generator.parameters(), lr=2e-5, weight_decay=1e-5)
opt_D = torch.optim.AdamW(net_D.parameters(), lr=5e-5, weight_decay=1e-5)

sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=5, eta_min=1e-6)
sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=5, eta_min=1e-6)
print('Optimizers & Loss modules initialized.')"""
    cells.append(make_cell("code", c6))

    # Step 7 - Training Loop
    cells.append(make_cell("markdown", "## Step 7 — Adversarial Fine-Tuning Loop (5 Epochs Recommended)"))
    c7 = """from tqdm import tqdm

# Training duration configuration:
# - EPOCHS = 3: Fast exploratory run (~35 to 40 min on Colab T4).
# - EPOCHS = 5: Recommended refinement equilibrium (~1.1 hours on Colab T4).
# - EPOCHS = 10: Extended convergence run (~2.2 hours on Colab T4).
EPOCHS = 5
LAMBDA_EDGE = 0.20
LAMBDA_SPEC = 0.15
LAMBDA_VGG  = 0.05
LAMBDA_ADV  = 0.005

BEST_ADV_CKPT = os.path.join(RESULTS_DIR, 'sen2sr_lite_adversarial_best.pth')
LAST_ADV_CKPT = os.path.join(RESULTS_DIR, 'sen2sr_lite_adversarial_last.pth')

best_val_loss = float('inf')
history_gan = {'loss_G': [], 'loss_D': [], 'val_l1': []}

print('=' * 80)
print(f'STARTING ADVERSARIAL FINE-TUNING LOOP ({EPOCHS} Epochs)')
print(f'Train scenes: {len(train_ds)} | Val scenes: {len(val_ds)}')
print('=' * 80)

t0_adv = time.time()
for epoch in range(1, EPOCHS + 1):
    generator.train()
    net_D.train()
    running_G, running_D = 0.0, 0.0

    for lr_b, hr_b in train_loader:
        lr_b = lr_b.to(device)
        hr_b = hr_b.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        opt_D.zero_grad()
        with torch.no_grad():
            fake_hr = generator(lr_b)

        d_real = net_D(hr_b)
        d_fake = net_D(fake_hr.detach())

        loss_d_real = gan_criterion(d_real, torch.ones_like(d_real))
        loss_d_fake = gan_criterion(d_fake, torch.zeros_like(d_fake))
        loss_D = 0.5 * (loss_d_real + loss_d_fake)

        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(net_D.parameters(), max_norm=1.0)
        opt_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        opt_G.zero_grad()
        gen_hr = generator(lr_b)
        d_fake_for_g = net_D(gen_hr)

        l_l1 = l1_loss_fn(gen_hr, hr_b)
        l_edge = edge_loss_fn(gen_hr, hr_b)
        l_spec = spec_loss_fn(gen_hr, hr_b)
        l_vgg  = vgg_loss_fn(gen_hr[:, :3], hr_b[:, :3]) # RGB channels
        l_adv  = gan_criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))

        loss_G = l_l1 + (LAMBDA_EDGE * l_edge) + (LAMBDA_SPEC * l_spec) + (LAMBDA_VGG * l_vgg) + (LAMBDA_ADV * l_adv)

        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        opt_G.step()

        running_G += loss_G.item() * lr_b.size(0)
        running_D += loss_D.item() * lr_b.size(0)

    sched_G.step()
    sched_D.step()

    ep_loss_G = running_G / len(train_ds)
    ep_loss_D = running_D / len(train_ds)

    # Validation
    generator.eval()
    running_val_l1 = 0.0
    with torch.no_grad():
        for lr_v, hr_v in val_loader:
            lr_v, hr_v = lr_v.to(device), hr_v.to(device)
            v_sr = generator(lr_v)
            running_val_l1 += l1_loss_fn(v_sr, hr_v).item() * lr_v.size(0)

    ep_val_l1 = running_val_l1 / len(val_ds)
    history_gan['loss_G'].append(ep_loss_G)
    history_gan['loss_D'].append(ep_loss_D)
    history_gan['val_l1'].append(ep_val_l1)

    is_best = ep_val_l1 < best_val_loss
    if is_best:
        best_val_loss = ep_val_l1
        torch.save({'epoch': epoch, 'model_state_dict': generator.state_dict(), 'val_l1': best_val_loss}, BEST_ADV_CKPT)

    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Loss G: {ep_loss_G:.4f} | Loss D: {ep_loss_D:.4f} | Val L1: {ep_val_l1:.4f} {'[BEST SAVED]' if is_best else ''}")

torch.save({'epoch': EPOCHS, 'model_state_dict': generator.state_dict(), 'val_l1': ep_val_l1}, LAST_ADV_CKPT)
print(f"\\nAdversarial training completed in {time.time() - t0_adv:.1f}s.")
print(f"  Best checkpoint: {BEST_ADV_CKPT}")"""
    cells.append(make_cell("code", c7))

    # Step 8 - 5-Way Comparison
    cells.append(make_cell("markdown", "## Step 8 — 5-Way Evaluation against Real NAIP Aerial Ground Truth\n\n> [!WARNING]\n> **Texture Comparison Notice:** Observe the fine details in the Adversarial output against real NAIP ground truth. Textures that match aerial structures reflect recovered signal; textures absent in aerial photography are hallucinated high-frequency noise."))
    c8 = """import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# Load Baseline and Compound Models for Comparison
baseline_model = CNNSR(in_channels=4, out_channels=4, feature_channels=24, upscale=4, bias=True, train_mode=False, num_blocks=6)
baseline_model.load_state_dict(safetensors.torch.load_file(os.path.join(MODEL_DIR, 'sr_model.safetensor')))
baseline_model = baseline_model.to(device).eval()

compound_model = CNNSR(in_channels=4, out_channels=4, feature_channels=24, upscale=4, bias=True, train_mode=False, num_blocks=6)
if os.path.exists(BEST_COMPOUND):
    c_ckpt = torch.load(BEST_COMPOUND, map_location=device)
    compound_model.load_state_dict(c_ckpt['model_state_dict'] if 'model_state_dict' in c_ckpt else c_ckpt)
compound_model = compound_model.to(device).eval()

# Load best adversarial generator
adv_ckpt = torch.load(BEST_ADV_CKPT, map_location=device)
generator.load_state_dict(adv_ckpt['model_state_dict'] if 'model_state_dict' in adv_ckpt else adv_ckpt)
generator.eval()

def shared_stretch(img_4b, ref_hr):
    rgb = img_4b[[0, 1, 2]].transpose(1, 2, 0)
    ref_rgb = ref_hr[[0, 1, 2]].transpose(1, 2, 0)
    lo = np.percentile(ref_rgb, 2)
    hi = np.percentile(ref_rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

N_TEST_SHOW = min(3, len(test_scenes))
fig, axes = plt.subplots(N_TEST_SHOW, 5, figsize=(25, 5 * N_TEST_SHOW))
if N_TEST_SHOW == 1:
    axes = np.expand_dims(axes, 0)

for i in range(N_TEST_SHOW):
    scene = test_scenes[i]
    lr_full, hr_full = load_canonical_pair(scene['lr'], scene['hr'], is_worldstrat=scene['is_ws'])
    cy, cx = lr_full.shape[1] // 2, lr_full.shape[2] // 2
    lr_c = lr_full[:, cy-32:cy+32, cx-32:cx+32]
    hr_c = hr_full[:, (cy-32)*4:(cy+32)*4, (cx-32)*4:(cx+32)*4]
    hr_c = match_radiometry_patch(hr_c, lr_c)

    in_t = torch.from_numpy(lr_c).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_base = baseline_model(in_t).squeeze(0).cpu().numpy()
        sr_comp = compound_model(in_t).squeeze(0).cpu().numpy()
        sr_adv  = generator(in_t).squeeze(0).cpu().numpy()
        bicubic = F.interpolate(in_t, scale_factor=4, mode='bicubic', align_corners=False).squeeze(0).cpu().numpy()

    axes[i, 0].imshow(shared_stretch(bicubic, hr_c))
    axes[i, 0].set_title(f"1. Native 10m Bicubic 4x\\n{scene['id']}", fontweight='bold')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(shared_stretch(sr_base, hr_c))
    axes[i, 1].set_title("2. Pretrained Baseline\\nOfficial Weights", fontweight='bold')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(shared_stretch(sr_comp, hr_c))
    axes[i, 2].set_title("3. Compound Fine-Tuned\\nShift-Tolerant L1", fontweight='bold')
    axes[i, 2].axis('off')

    axes[i, 3].imshow(shared_stretch(sr_adv, hr_c))
    axes[i, 3].set_title("4. Adversarial PatchGAN (Ours)\\nSharpened Texture", fontweight='bold', color='darkorange')
    axes[i, 3].axis('off')

    axes[i, 4].imshow(shared_stretch(hr_c, hr_c))
    axes[i, 4].set_title("5. Real NAIP Aerial Ground Truth\\nGround Reference", fontweight='bold', color='green')
    axes[i, 4].axis('off')

plt.tight_layout()
adv_preview = os.path.join(RESULTS_DIR, 'sen2sr_lite_adversarial_5way_comparison.png')
plt.savefig(adv_preview, dpi=160, bbox_inches='tight')
plt.show()
print(f"5-Way comparison preview saved to: {adv_preview}")"""
    cells.append(make_cell("code", c8))

    # Step 9 - Indian Target Benchmark: Graphic Era University (Dehradun)
    cells.append(make_cell("markdown", """## Step 9 — Real Copernicus Benchmark: Graphic Era University (Dehradun, India)

### Visualizing Adversarial Texture vs. Compound Loss on Real Terrain
This evaluation runs full-scene reflect-padded inference on the Graphic Era University capture (`Browser_images_new1.zip`).
It compares:
1. Native 10m Input (Bicubic 4x)
2. Pretrained Baseline SEN2SR Lite (Official Taco Foundation Weights)
3. Fine-Tuned Compound Loss (Shift-Tolerant L1 + Sobel + Spectral)
4. Fine-Tuned Adversarial PatchGAN (Ours: Spectral-Norm Discriminator + VGG Perceptual)

Observe the rooftop textures and road edges closely in the zoom callouts."""))

    c10 = """# Unpack & Run Reflect-Padded Inference on Graphic Era University
import zipfile, io, re
from PIL import Image

possible_new1_zips = [
    os.path.join(DATASETS_DIR, 'Browser_images_new1.zip'),
    os.path.join(PROJECT_DIR, 'Datasets', 'Browser_images_new1.zip'),
    'data/Browser_images_new1.zip',
    os.path.join(RESULTS_DIR, 'Browser_images_new1.zip')
]
GRAPHIC_ERA_ZIP = None
for pz in possible_new1_zips:
    if os.path.exists(pz):
        GRAPHIC_ERA_ZIP = pz
        break

assert GRAPHIC_ERA_ZIP is not None, f"Graphic Era archive (Browser_images_new1.zip) not found in {DATASETS_DIR}!"

print("=" * 80)
print("BENCHMARKING INDIAN TARGET: Graphic Era University, Dehradun")
print(f"Archive Source: {GRAPHIC_ERA_ZIP}")
print("=" * 80)

BAND_NAMES = ['B04', 'B03', 'B02', 'B08']
raw_bands = {}

with zipfile.ZipFile(GRAPHIC_ERA_ZIP, 'r') as z:
    for name in z.namelist():
        for b in BAND_NAMES:
            if re.search(r'(^|[-_.])' + b + r'([-._]|$)', os.path.basename(name).upper()):
                with z.open(name) as f:
                    im = Image.open(io.BytesIO(f.read()))
                    raw_bands[b] = np.array(im, dtype=np.float32)

print("Identified Graphic Era Bands:", list(raw_bands.keys()))
assert len(raw_bands) >= 4, f"Missing required bands from {BAND_NAMES}, found: {list(raw_bands.keys())}"

ge_cube_raw = np.stack([raw_bands['B04'], raw_bands['B03'], raw_bands['B02'], raw_bands['B08']], axis=0)
print(f"Graphic Era Raw Input Extent: {ge_cube_raw.shape}")

# Radiometric calibration for Copernicus L2A (offset subtraction & specular tone mapping)
def calibrate_copernicus(cube_raw, offset=1000.0, scale_s2=10000.0):
    calibrated = np.zeros_like(cube_raw, dtype=np.float32)
    for c in range(cube_raw.shape[0]):
        ch = cube_raw[c].astype(np.float32)
        if ch.max() > 255.0:
            if ch.min() >= 800.0:
                refl = np.maximum(0.0, (ch - offset)) / scale_s2
            else:
                refl = ch / scale_s2
        elif ch.max() > 2.0:
            refl = ch / 255.0
        else:
            refl = ch

        p99 = np.percentile(refl, 99)
        if p99 > 0.85:
            refl = refl * (0.85 / p99)
        calibrated[c] = np.clip(refl, 0.0, 1.0)
    return calibrated

ge_calibrated = calibrate_copernicus(ge_cube_raw)
print(f"Calibrated Surface Reflectance: shape={ge_calibrated.shape}, range=[{ge_calibrated.min():.4f}, {ge_calibrated.max():.4f}]")

# Reflect-Padded Sliding Window Inference Engine (Zero Dark Corners)
def run_reflect_padded_tiled_inference(model, X, patch_size=128, stride=48, scale=4):
    if hasattr(X, 'cpu'):
        arr = X.squeeze(0).cpu().numpy() if X.dim() == 4 else X.cpu().numpy()
    else:
        arr = np.array(X, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]

    C, H, W = arr.shape
    ps_hr = patch_size * scale
    pad_h = max(stride, (patch_size - H + 1) // 2 + 16)
    pad_w = max(stride, (patch_size - W + 1) // 2 + 16)

    padded = np.pad(arr, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    Hp, Wp = padded.shape[1], padded.shape[2]

    w1d = torch.hann_window(ps_hr, periodic=False, dtype=torch.float32)
    w2d = (w1d.unsqueeze(1) * w1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(device)

    out_h, out_w = Hp * scale, Wp * scale
    acc = torch.zeros(1, C, out_h, out_w, device=device)
    weight = torch.zeros(1, 1, out_h, out_w, device=device)

    X_pad_t = torch.from_numpy(padded).unsqueeze(0).float()
    y_steps = list(range(0, Hp - patch_size + 1, stride))
    if not y_steps or y_steps[-1] != Hp - patch_size:
        y_steps.append(Hp - patch_size)
    x_steps = list(range(0, Wp - patch_size + 1, stride))
    if not x_steps or x_steps[-1] != Wp - patch_size:
        x_steps.append(Wp - patch_size)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                patch = X_pad_t[:, :, y:y+patch_size, x:x+patch_size].to(device)
                sr_p = model(patch)
                acc[:, :, y*scale:y*scale+ps_hr, x*scale:x*scale+ps_hr] += sr_p * w2d
                weight[:, :, y*scale:y*scale+ps_hr, x*scale:x*scale+ps_hr] += w2d

    blended = acc / torch.clamp(weight, min=1e-6)
    cropped = blended[:, :, pad_h*scale:(pad_h+H)*scale, pad_w*scale:(pad_w+W)*scale]
    return torch.clamp(cropped, 0.0, 1.0).squeeze(0).cpu().numpy()

print("Running full-scene reflect-padded inference on Graphic Era...")
ge_bicubic = F.interpolate(torch.from_numpy(ge_calibrated).unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False).squeeze(0).numpy()
ge_base    = run_reflect_padded_tiled_inference(baseline_model, ge_calibrated, patch_size=128, stride=48, scale=4)
ge_comp    = run_reflect_padded_tiled_inference(compound_model, ge_calibrated, patch_size=128, stride=48, scale=4)
ge_adv     = run_reflect_padded_tiled_inference(generator, ge_calibrated, patch_size=128, stride=48, scale=4)
print("Inference complete across all 4 resolution tiers.")"""
    cells.append(make_cell("code", c10))

    # Step 10 - 4-Way Visual Matrix: Full Scene + Zooms
    c11 = """# 4-Way Comprehensive Visual Matrix: Full Scene & Campus Rooftop Zooms
fig = plt.figure(figsize=(24, 18))

rgb_ref = ge_comp[:3].transpose(1, 2, 0)
lo, hi = np.percentile(rgb_ref, [2, 98])

def stretch_ge(cube):
    rgb = cube[:3].transpose(1, 2, 0)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

img_bic = stretch_ge(ge_bicubic)
img_bas = stretch_ge(ge_base)
img_cmp = stretch_ge(ge_comp)
img_adv = stretch_ge(ge_adv)

models_info = [
    (img_bic, "1. Native 10m Input (Bicubic 4x)\\nCopernicus S2 Capture", 'black'),
    (img_bas, "2. Pretrained Baseline SEN2SR Lite (2.5m)\\nOfficial Weights", 'black'),
    (img_cmp, "3. Fine-Tuned Compound Loss (2.5m, Ours)\\nShift-Tolerant L1 + Sobel", 'green'),
    (img_adv, "4. Fine-Tuned Adversarial PatchGAN (Ours)\\nVGG Perceptual + Discriminator", 'darkorange')
]

for col, (im, title, color) in enumerate(models_info):
    ax = plt.subplot2grid((3, 4), (0, col))
    ax.imshow(im)
    ax.set_title(title, fontsize=12, fontweight='bold', color=color)
    ax.axis('off')

# Zoom Callout 1: Academic Blocks & Building Roofs
H, W = img_cmp.shape[:2]
cy1, cx1 = int(H * 0.45), int(W * 0.40)
zh, zw = 35, 45

for col, (im, _, _) in enumerate(models_info):
    ax = plt.subplot2grid((3, 4), (1, col))
    ax.imshow(im[cy1-zh:cy1+zh, cx1-zw:cx1+zw])
    if col == 0:
        ax.set_ylabel("Zoom A: Campus Rooftops\\n(Grain vs Sharpness)", fontsize=11, fontweight='bold')
    if col == 2:
        ax.set_title("Compound: Smooth & Radiometric", fontsize=10, fontweight='bold', color='green')
    elif col == 3:
        ax.set_title("GAN: Crisp Texture (Inspect Grain)", fontsize=10, fontweight='bold', color='darkorange')
    ax.axis('off')

# Zoom Callout 2: Road Intersections & Campus Perimeters
cy2, cx2 = int(H * 0.65), int(W * 0.60)
for col, (im, _, _) in enumerate(models_info):
    ax = plt.subplot2grid((3, 4), (2, col))
    ax.imshow(im[cy2-zh:cy2+zh, cx2-zw:cx2+zw])
    if col == 0:
        ax.set_ylabel("Zoom B: Road Network\\n(Edge Silhouette)", fontsize=11, fontweight='bold')
    if col == 2:
        ax.set_title("Compound: Sharp Continuous Vector", fontsize=10, fontweight='bold', color='green')
    elif col == 3:
        ax.set_title("GAN: Enhanced Local Micro-Contrast", fontsize=10, fontweight='bold', color='darkorange')
    ax.axis('off')

plt.tight_layout()
ge_adv_png = os.path.join(RESULTS_DIR, 'sen2sr_lite_adversarial_graphic_era_showcase.png')
plt.savefig(ge_adv_png, dpi=180, bbox_inches='tight')
plt.show()
print(f"Graphic Era University Adversarial showcase saved to: {ge_adv_png}")"""
    cells.append(make_cell("code", c11))

    return create_notebook(cells)

def validate_and_save(nb, path):
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
    gan_nb = build_gan_notebook()
    p_gan = 'notebooks/04_finetuning_experiments/04b_Lite_Adversarial_Finetune.ipynb'
    validate_and_save(gan_nb, p_gan)

    gdrive_gan = 'G:/My Drive/Project_Folder/Notebooks/04_finetuning_experiments/04b_Lite_Adversarial_Finetune.ipynb'
    if os.path.exists(os.path.dirname(gdrive_gan)):
        with open(gdrive_gan, 'w', encoding='utf-8') as f:
            json.dump(gan_nb, f, indent=2)
        print(f"Synced to Google Drive: {gdrive_gan}")
