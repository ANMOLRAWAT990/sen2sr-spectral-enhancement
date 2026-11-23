import json
import os
import ast

def make_cell(cell_type, source, cell_id=None):
    lines = source.split('\n')
    src = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src
    }
    if cell_id:
        cell["id"] = cell_id
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def build_full_adversarial_notebook():
    cells = []

    # =========================================================================
    # Cell 0: Markdown Header & Overview
    # =========================================================================
    c0 = """# SEN2SR Full (`MambaSR`): Adversarial Fine-Tuning with PatchGAN & Spectral Normalization

**Major Project — Spectral Enhancement**

> [!WARNING]
> **Adversarial Texture Notice (The Perception-Distortion Frontier):**
> Adversarial GAN training (Generator vs. Discriminator) forces the MambaSR network to recover sharp, high-frequency boundaries and micro-contrasts along building footprints and road margins.
> However, pure GAN losses risk hallucinating synthetic textures (e.g. artificial rooftop grain or vegetation noise).
> This notebook trains the **SEN2SR Full Adversarial Model** and evaluates it against:
> 1. Real coincident NAIP aerial photography (Ground Truth reference)
> 2. Native Sentinel-2 Copernicus imagery of an Indian campus: **Graphic Era University (Dehradun, India)**

---

### Key Architectural Components
1. **Generator**: SEN2SR Full `MambaSR` (Vision State-Space Model / selective-scan sequence architecture).
2. **Discriminator**: $70\\times 70$ PatchGAN with Spectral Normalization (enforces 1-Lipschitz continuity for stable minimax convergence).
3. **Loss Objective**: $L_{\\text{total}} = L_{1,\\text{shift=1}} + 0.20 \\cdot L_{\\text{sobel}} + 0.15 \\cdot L_{\\text{spectral}} + 0.05 \\cdot L_{\\text{VGG}} + 0.005 \\cdot L_{\\text{adv}}$.
4. **Estimated Training Time (Google Colab T4 GPU)**:
   - **`EPOCHS = 3`**: ~45 to 50 minutes (fast exploration).
   - **`EPOCHS = 5`**: ~1.2 to 1.4 hours (recommended default for stable equilibrium).
   - **`EPOCHS = 10`**: ~2.4 to 2.8 hours."""
    cells.append(make_cell("markdown", c0))

    # =========================================================================
    # Cell 1: Step 1 - Colab Environment Setup
    # =========================================================================
    cells.append(make_cell("markdown", "## Step 1 — Colab Environment & Google Drive Setup"))
    c1 = """import os, sys, glob, time, warnings, logging

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CPL_LOG_ERRORS'] = 'OFF'
os.environ['CPL_LOG'] = 'OFF'
os.environ['GDAL_PAM_ENABLED'] = 'NO'

warnings.filterwarnings('ignore')
for logger_name in ['rasterio', 'rasterio._env', 'rasterio.env', 'rasterio._base']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    IN_COLAB = True
    print("Google Drive mounted successfully.")
except Exception:
    IN_COLAB = False
    print("Running in local/offline environment.")

if IN_COLAB:
    PROJECT_DIR  = '/content/drive/My Drive/Project_Folder'
    DATASETS_DIR = os.path.join(PROJECT_DIR, 'Datasets')
    RESULTS_DIR  = os.path.join(PROJECT_DIR, 'Results')
    MODEL_DIR    = '/content/model/SEN2SR_FULL'
else:
    PROJECT_DIR  = 'G:/My Drive/Project_Folder'
    DATASETS_DIR = os.path.join(PROJECT_DIR, 'Datasets') if os.path.exists('G:/My Drive/Project_Folder/Datasets') else 'data'
    RESULTS_DIR  = os.path.join(PROJECT_DIR, 'Results') if os.path.exists('G:/My Drive/Project_Folder/Results') else 'results'
    MODEL_DIR    = 'models/SEN2SR_FULL'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Version: {torch.__version__} | Active Device: {device}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")"""
    cells.append(make_cell("code", c1))

    # =========================================================================
    # Cell 2: Step 2 - Dependencies & Mamba Fallback Engine
    # =========================================================================
    cells.append(make_cell("markdown", "## Step 2 — Install Dependencies & Initialize Mamba Engine"))
    c2 = """# Install required packages
if IN_COLAB:
    import subprocess
    pkgs = ['sen2sr', 'mlstac', 'rasterio', 'safetensors', 'einops', 'timm', 'torchvision', 'pandas', 'matplotlib', 'scikit-image', 'tqdm']
    for p in pkgs:
        try:
            __import__(p)
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', p], check=True)
    print("Dependencies verified.")

import types
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
    print("Native compiled Mamba CUDA engine verified successfully.")
except Exception as e:
    print(f"Notice: Native CUDA selective_scan unavailable ({e}). Activating pure-PyTorch selective scan fallback...")

    def pytorch_selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        batch, dim, dstate = u.shape[0], u.shape[1], A.shape[1]
        length = u.shape[2]

        if delta_bias is not None:
            delta = delta + delta_bias.unsqueeze(-1)
        if delta_softplus:
            delta = torch.nn.functional.softplus(delta)

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
    print("Pure-PyTorch selective scan fallback initialized successfully.")

from sen2sr.models.mambasr.mambasr import MambaSR
print("MambaSR architecture successfully loaded.")"""
    cells.append(make_cell("code", c2))

    # =========================================================================
    # Cell 3: Step 3 - Canonical Band Ordering & Radiometric Helpers
    # =========================================================================
    cells.append(make_cell("markdown", "## Step 3 — Canonical Band Ordering & Radiometric Calibration"))
    c3 = """import rasterio
import numpy as np

# Canonical Band Ordering: [Red, Green, Blue, NIR] = [B04, B03, B02, B08]
SEN2NAIP_LR_ORDER = [0, 1, 2, 3]
SEN2NAIP_HR_ORDER = [0, 1, 2, 3]
WS_LR_ORDER       = [3, 2, 1, 7]
WS_HR_ORDER       = [0, 1, 2, 3]

SCALE_S2    = 10000.0
SCALE_NAIP  = 255.0
SCALE_WS_HR = 4095.0

def read_raster_audit(path):
    with rasterio.open(path) as src:
        arr = src.read()
        return arr.astype(np.float32)

def match_radiometry_patch(hr_patch, lr_patch):
    \"\"\"Quantile-matches HR aerial radiometry to satellite LR patch per channel.\"\"\"
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
        import torch.nn.functional as F
        hr_t = torch.from_numpy(hr).unsqueeze(0).float()
        hr = F.interpolate(hr_t, size=(expected_h, expected_w), mode='bilinear', align_corners=False).squeeze(0).numpy()
    return hr

def load_canonical_pair(lr_path, hr_path, is_worldstrat=False):
    lr_raw = read_raster_audit(lr_path)
    hr_raw = read_raster_audit(hr_path)

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
    assert lr_scaled.shape[0] == 4, f"Invalid LR channels: {lr_scaled.shape}"
    assert hr_scaled.shape[0] == 4, f"Invalid HR channels: {hr_scaled.shape}"
    return lr_scaled, hr_scaled

print("Canonical band ordering and radiometric helpers compiled.")"""
    cells.append(make_cell("code", c3))

    # =========================================================================
    # Cell 4: Step 4 - Scene-Stratified Split & Dynamic Dataset
    # =========================================================================
    cells.append(make_cell("markdown", "## Step 4 — Scene-Stratified Split & Dynamic Augmentation Dataset"))
    c4 = """from torch.utils.data import Dataset, DataLoader
import random, json

TEST_SCENES_JSON = os.path.join(RESULTS_DIR, 'test_scenes.json')

all_scenes = []

# Discover SEN2NAIP pairs
sen2naip_lr = sorted(glob.glob(os.path.join(DATASETS_DIR, 'sentinaiplow', '*.tif')))
sen2naip_hr = sorted(glob.glob(os.path.join(DATASETS_DIR, 'sentinaiphigh', '*.tif')))
hr_dict = {os.path.basename(f).replace('_hr.tif', ''): f for f in sen2naip_hr}
for lr_p in sen2naip_lr:
    sid = os.path.basename(lr_p).replace('_lr.tif', '')
    if sid in hr_dict:
        all_scenes.append({'id': sid, 'lr': lr_p, 'hr': hr_dict[sid], 'is_ws': False})

# Discover WorldStrat pairs
ws_lr = sorted(glob.glob(os.path.join(DATASETS_DIR, 'worldstrat_low', '**/L2A/*-L2A_data.tiff'), recursive=True))
ws_hr = sorted(glob.glob(os.path.join(DATASETS_DIR, 'worldstrat_high', 'extracted/**/*_rgbn.tiff'), recursive=True))
ws_hr_map = {os.path.basename(h).replace('_rgbn.tiff', '').replace('-rgbn.tiff', ''): h for h in ws_hr}

for l in ws_lr:
    base = os.path.basename(l).split('-L2A')[0]
    if base in ws_hr_map:
        all_scenes.append({'id': f"WS_{base}", 'lr': l, 'hr': ws_hr_map[base], 'is_ws': True})

print(f"Total Discovered Scenes: {len(all_scenes)}")
assert len(all_scenes) > 0, "No dataset scenes found! Verify Datasets directory."

# Load canonical held-out test split
if os.path.exists(TEST_SCENES_JSON):
    with open(TEST_SCENES_JSON, 'r') as f:
        saved_split = json.load(f)
    test_ids = set(saved_split.get('test_ids', []))
    print(f"Loaded existing stratified test split from {TEST_SCENES_JSON} ({len(test_ids)} test scenes).")
else:
    random.seed(42)
    shuffled = list(all_scenes)
    random.shuffle(shuffled)
    n_test = max(30, int(len(shuffled) * 0.10))
    test_ids = set([s['id'] for s in shuffled[:n_test]])
    test_scenes_saved = [s for s in shuffled if s['id'] in test_ids]
    with open(TEST_SCENES_JSON, 'w') as f:
        json.dump({'test_ids': list(test_ids), 'test_scenes': test_scenes_saved}, f, indent=2)
    print(f"Created new stratified split with {len(test_ids)} test scenes.")

test_scenes  = [s for s in all_scenes if s['id'] in test_ids]
train_val    = [s for s in all_scenes if s['id'] not in test_ids]
val_count    = max(20, int(len(train_val) * 0.12))
train_scenes = train_val[:-val_count]
val_scenes   = train_val[-val_count:]

print(f"Split breakdown -> Train: {len(train_scenes)} | Val: {len(val_scenes)} | Test: {len(test_scenes)}")

class FullMambaPatchDataset(Dataset):
    \"\"\"
    Dynamic Patch Dataset for MambaSR Super-Resolution:
    - 64x64 LR -> 256x256 HR random crop per epoch.
    - Patch quantile radiometry matching.
    - Synchronous geometric flips/rotations.
    - Photometric jitter so bright urban roofs are within training range.
    \"\"\"
    def __init__(self, scenes, patch_lr=64, scale=4, is_train=True):
        self.scenes = scenes
        self.patch_lr = patch_lr
        self.scale = scale
        self.patch_hr = patch_lr * scale
        self.is_train = is_train

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        lr_full, hr_full = load_canonical_pair(scene['lr'], scene['hr'], is_worldstrat=scene['is_ws'])

        H_lr, W_lr = lr_full.shape[1], lr_full.shape[2]
        if H_lr <= self.patch_lr or W_lr <= self.patch_lr:
            pad_h = max(0, self.patch_lr - H_lr + 1)
            pad_w = max(0, self.patch_lr - W_lr + 1)
            lr_full = np.pad(lr_full, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            hr_full = np.pad(hr_full, ((0,0), (0, pad_h * self.scale), (0, pad_w * self.scale)), mode='reflect')
            H_lr, W_lr = lr_full.shape[1], lr_full.shape[2]

        if self.is_train:
            y_lr = random.randint(0, H_lr - self.patch_lr)
            x_lr = random.randint(0, W_lr - self.patch_lr)
        else:
            y_lr = (H_lr - self.patch_lr) // 2
            x_lr = (W_lr - self.patch_lr) // 2

        y_hr, x_hr = y_lr * self.scale, x_lr * self.scale
        lr_crop = lr_full[:, y_lr:y_lr+self.patch_lr, x_lr:x_lr+self.patch_lr]
        hr_crop = hr_full[:, y_hr:y_hr+self.patch_hr, x_hr:x_hr+self.patch_hr]

        hr_crop = match_radiometry_patch(hr_crop, lr_crop)

        if self.is_train:
            if random.random() > 0.5:
                lr_crop = np.flip(lr_crop, axis=2)
                hr_crop = np.flip(hr_crop, axis=2)
            if random.random() > 0.5:
                lr_crop = np.flip(lr_crop, axis=1)
                hr_crop = np.flip(hr_crop, axis=1)
            k = random.randint(0, 3)
            if k > 0:
                lr_crop = np.rot90(lr_crop, k, axes=(1, 2))
                hr_crop = np.rot90(hr_crop, k, axes=(1, 2))

            # Photometric jitter
            b_scale = random.uniform(0.85, 1.25)
            c_scale = random.uniform(0.90, 1.15)
            lr_mean = lr_crop.mean(axis=(1, 2), keepdims=True)
            hr_mean = hr_crop.mean(axis=(1, 2), keepdims=True)
            lr_crop = np.clip((lr_crop - lr_mean) * c_scale + lr_mean * b_scale, 0.0, 1.0)
            hr_crop = np.clip((hr_crop - hr_mean) * c_scale + hr_mean * b_scale, 0.0, 1.0)

        lr_t = torch.from_numpy(np.ascontiguousarray(lr_crop)).float()
        hr_t = torch.from_numpy(np.ascontiguousarray(hr_crop)).float()

        if lr_t.shape != (4, self.patch_lr, self.patch_lr):
            import torch.nn.functional as F
            lr_t = F.interpolate(lr_t.unsqueeze(0), size=(self.patch_lr, self.patch_lr), mode='bilinear').squeeze(0)
        if hr_t.shape != (4, self.patch_hr, self.patch_hr):
            import torch.nn.functional as F
            hr_t = F.interpolate(hr_t.unsqueeze(0), size=(self.patch_hr, self.patch_hr), mode='bilinear').squeeze(0)

        return lr_t, hr_t

train_ds = FullMambaPatchDataset(train_scenes, patch_lr=64, is_train=True)
val_ds   = FullMambaPatchDataset(val_scenes, patch_lr=64, is_train=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
print(f"DataLoaders initialized: {len(train_loader)} train batches | {len(val_loader)} val batches.")"""
    cells.append(make_cell("code", c4))

    # =========================================================================
    # Cell 5: Step 5 - PatchGAN Discriminator & MambaSR Generator
    # =========================================================================
    cells.append(make_cell("markdown", "## Step 5 — Architecture: MambaSR Generator & Spectral-Normalized PatchGAN"))
    c5 = """import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision.models as models
import safetensors.torch
import mlstac

# 1. 70x70 PatchGAN Discriminator with Spectral Normalization
class PatchGANDiscriminator(nn.Module):
    \"\"\"
    Spectral-Normalized 70x70 PatchGAN Discriminator.
    Enforces 1-Lipschitz continuity for stable minimax convergence.
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

# 2. VGG-19 Perceptual Loss Module
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_rgb, target_rgb):
        p_norm = (pred_rgb - self.mean) / self.std
        t_norm = (target_rgb - self.mean) / self.std
        return F.l1_loss(self.vgg(p_norm), self.vgg(t_norm))

net_D = PatchGANDiscriminator(in_channels=4, ndf=64).to(device)
vgg_loss_fn = VGGPerceptualLoss().to(device)

# 3. MambaSR Generator
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

generator = MambaSR(**sr_parameters)

# Initialize generator weights: check for fine-tuned checkpoint first, then baseline
BEST_COMPOUND = os.path.join(RESULTS_DIR, 'sen2sr_full_mamba_compound_finetuned.pth')
LAST_COMPOUND = os.path.join(RESULTS_DIR, 'sen2sr_full_compound_best.pth')
weights_path  = os.path.join(MODEL_DIR, 'sr_model.safetensor')

if os.path.exists(BEST_COMPOUND):
    ckpt = torch.load(BEST_COMPOUND, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    generator.load_state_dict(state, strict=False)
    print(f"Generator initialized from fine-tuned compound checkpoint: {BEST_COMPOUND}")
elif os.path.exists(LAST_COMPOUND):
    ckpt = torch.load(LAST_COMPOUND, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    generator.load_state_dict(state, strict=False)
    print(f"Generator initialized from compound checkpoint: {LAST_COMPOUND}")
else:
    if not os.path.exists(weights_path):
        print("Downloading official Taco Foundation SEN2SR Full baseline weights...")
        try:
            mlstac.download(file='https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SR/main/mlm.json', output_dir=MODEL_DIR)
        except Exception as e:
            import urllib.request
            u = 'https://huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SR/main/sr_model.safetensor'
            urllib.request.urlretrieve(u, weights_path)
    generator.load_state_dict(safetensors.torch.load_file(weights_path), strict=False)
    print("Generator initialized from official baseline weights.")

generator = generator.to(device)
print(f"MambaSR Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"PatchGAN Discriminator parameters: {sum(p.numel() for p in net_D.parameters()):,}")"""
    cells.append(make_cell("code", c5))

    # =========================================================================
    # Cell 6: Step 6 - Loss Formulations & Optimizers
    # =========================================================================
    cells.append(make_cell("markdown", "## Step 6 — Multi-Objective Compound & Adversarial Loss Formulations"))
    c6 = """class ShiftTolerantL1Loss(nn.Module):
    \"\"\"Tightened Shift-Tolerant L1 (+/- 1 px) to eliminate rooftop checkerboard grain.\"\"\"
    def __init__(self, max_shift=1):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, pred, target):
        if self.max_shift == 0:
            return F.l1_loss(pred, target)
        min_loss = None
        for dy in range(-self.max_shift, self.max_shift + 1):
            for dx in range(-self.max_shift, self.max_shift + 1):
                if dy > 0:
                    p = pred[:, :, dy:, :]
                    t = target[:, :, :-dy, :]
                elif dy < 0:
                    p = pred[:, :, :dy, :]
                    t = target[:, :, -dy:, :]
                else:
                    p = pred
                    t = target

                if dx > 0:
                    p = p[:, :, :, dx:]
                    t = t[:, :, :, :-dx]
                elif dx < 0:
                    p = p[:, :, :, :dx]
                    t = t[:, :, :, -dx:]

                loss = F.l1_loss(p, t)
                if min_loss is None or loss < min_loss:
                    min_loss = loss
        return min_loss

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
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

l1_loss_fn    = ShiftTolerantL1Loss(max_shift=1).to(device)
edge_loss_fn  = SobelEdgeLoss().to(device)
spec_loss_fn  = MultiIndexSpectralLoss().to(device)
gan_criterion = nn.BCEWithLogitsLoss().to(device)

# Optimizers with conservative learning rates
trainable_g = (
    list(generator.conv_after_body.parameters()) +
    list(generator.conv_before_upsample.parameters()) +
    list(generator.upsample.parameters()) +
    list(generator.conv_last.parameters())
)

opt_G = torch.optim.AdamW(trainable_g, lr=2e-5, weight_decay=1e-5)
opt_D = torch.optim.AdamW(net_D.parameters(), lr=5e-5, weight_decay=1e-5)

EPOCHS = 5
sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=EPOCHS, eta_min=1e-6)
sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=EPOCHS, eta_min=1e-6)
print("Optimizers & Loss modules initialized.")"""
    cells.append(make_cell("code", c6))

    # =========================================================================
    # Cell 7: Step 7 - Adversarial Training Loop
    # =========================================================================
    cells.append(make_cell("markdown", "## Step 7 — Adversarial Fine-Tuning Loop (5 Epochs Recommended)"))
    c7 = """from tqdm import tqdm

# Training duration configuration:
# - EPOCHS = 3: Fast exploratory run (~45 to 50 min on Colab T4).
# - EPOCHS = 5: Recommended refinement equilibrium (~1.2 to 1.4 hours on Colab T4).
# - EPOCHS = 10: Extended convergence run (~2.4 to 2.8 hours on Colab T4).
EPOCHS = 5
LAMBDA_EDGE = 0.20
LAMBDA_SPEC = 0.15
LAMBDA_VGG  = 0.05
LAMBDA_ADV  = 0.005

BEST_ADV_CKPT = os.path.join(RESULTS_DIR, 'sen2sr_full_adversarial_best.pth')
LAST_ADV_CKPT = os.path.join(RESULTS_DIR, 'sen2sr_full_adversarial_last.pth')

best_val_loss = float('inf')
history_gan = {'loss_G': [], 'loss_D': [], 'val_l1': []}

print('=' * 80)
print(f'STARTING FULL MAMBASR ADVERSARIAL FINE-TUNING LOOP ({EPOCHS} Epochs)')
print(f'Train scenes: {len(train_ds)} | Val scenes: {len(val_ds)}')
print('=' * 80)

t0_adv = time.time()
for epoch in range(1, EPOCHS + 1):
    generator.train()
    net_D.train()
    running_loss_G, running_loss_D = 0.0, 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Full Adversarial]")
    for lr_batch, hr_batch in pbar:
        lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

        # -------------------------------------------------------------
        # 1. Train Discriminator D
        # -------------------------------------------------------------
        opt_D.zero_grad()
        with torch.no_grad():
            fake_hr = generator(lr_batch)

        pred_real = net_D(hr_batch)
        loss_D_real = gan_criterion(pred_real, torch.ones_like(pred_real) * 0.9)  # label smoothing

        pred_fake = net_D(fake_hr.detach())
        loss_D_fake = gan_criterion(pred_fake, torch.zeros_like(pred_fake))

        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        opt_D.step()

        # -------------------------------------------------------------
        # 2. Train Generator G
        # -------------------------------------------------------------
        opt_G.zero_grad()
        fake_hr = generator(lr_batch)

        l_l1 = l1_loss_fn(fake_hr, hr_batch)
        l_edge = edge_loss_fn(fake_hr, hr_batch)
        l_spec = spec_loss_fn(fake_hr, hr_batch)
        l_vgg  = vgg_loss_fn(fake_hr[:, :3], hr_batch[:, :3])

        pred_fake_for_G = net_D(fake_hr)
        l_adv = gan_criterion(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

        loss_G = l_l1 + (LAMBDA_EDGE * l_edge) + (LAMBDA_SPEC * l_spec) + (LAMBDA_VGG * l_vgg) + (LAMBDA_ADV * l_adv)
        loss_G.backward()
        opt_G.step()

        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()
        pbar.set_postfix({
            'loss_G': f"{loss_G.item():.4f}",
            'loss_D': f"{loss_D.item():.4f}",
            'l1': f"{l_l1.item():.4f}"
        })

    sched_G.step()
    sched_D.step()

    # Validation
    generator.eval()
    val_l1_acc = 0.0
    with torch.no_grad():
        for lr_b, hr_b in val_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            out = generator(lr_b)
            val_l1_acc += l1_loss_fn(out, hr_b).item()

    val_l1 = val_l1_acc / len(val_loader)
    avg_G = running_loss_G / len(train_loader)
    avg_D = running_loss_D / len(train_loader)

    history_gan['loss_G'].append(avg_G)
    history_gan['loss_D'].append(avg_D)
    history_gan['val_l1'].append(val_l1)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} Summary: Loss_G={avg_G:.4f} | Loss_D={avg_D:.4f} | Val L1={val_l1:.4f}")

    if val_l1 < best_val_loss:
        best_val_loss = val_l1
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'val_l1': val_l1
        }, BEST_ADV_CKPT)
        print(f"  [+] Saved new best full adversarial checkpoint: {BEST_ADV_CKPT}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'val_l1': val_l1
    }, LAST_ADV_CKPT)

print(f"Adversarial fine-tuning complete in {(time.time() - t0_adv)/60:.1f} minutes.")"""
    cells.append(make_cell("code", c7))

    # =========================================================================
    # Cell 8: Step 8 - 5-Way Evaluation against Real NAIP Ground Truth
    # =========================================================================
    cells.append(make_cell("markdown", """## Step 8 — 5-Way Evaluation against Real NAIP Aerial Ground Truth

> [!WARNING]
> **Texture Comparison Notice:** Observe the fine details in the Adversarial output against real NAIP ground truth. Textures that match aerial structures reflect recovered signal; textures absent in aerial photography are hallucinated high-frequency noise."""))

    c8 = """import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# Load Baseline and Compound Models for Comparison
baseline_model = MambaSR(**sr_parameters).to(device)
baseline_model.load_state_dict(safetensors.torch.load_file(weights_path), strict=False)
baseline_model.eval()

compound_model = MambaSR(**sr_parameters).to(device)
if os.path.exists(BEST_COMPOUND):
    c_ckpt = torch.load(BEST_COMPOUND, map_location=device)
    compound_model.load_state_dict(c_ckpt.get('model_state_dict', c_ckpt), strict=False)
compound_model.eval()

# Select 3 distinct held-out test scenes
eval_scenes = test_scenes[:3]
fig, axes = plt.subplots(len(eval_scenes), 5, figsize=(25, 5 * len(eval_scenes)))

def shared_stretch(img_4b, ref_hr):
    rgb = img_4b[:3].transpose(1, 2, 0)
    ref = ref_hr[:3].transpose(1, 2, 0)
    lo, hi = np.percentile(ref, (2, 98))
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

for i, scene in enumerate(eval_scenes):
    lr, hr = load_canonical_pair(scene['lr'], scene['hr'], is_worldstrat=scene['is_ws'])
    cy, cx = lr.shape[1] // 2, lr.shape[2] // 2
    lr_c = lr[:, cy-32:cy+32, cx-32:cx+32]
    hr_c = hr[:, (cy-32)*4:(cy+32)*4, (cx-32)*4:(cx+32)*4]
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
    axes[i, 1].set_title("2. Pretrained MambaSR Baseline\\nOfficial Weights", fontweight='bold')
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
adv_preview = os.path.join(RESULTS_DIR, 'sen2sr_full_adversarial_5way_comparison.png')
plt.savefig(adv_preview, dpi=160, bbox_inches='tight')
plt.show()
print(f"5-Way comparison preview saved to: {adv_preview}")"""
    cells.append(make_cell("code", c8))

    # =========================================================================
    # Cell 9: Step 9 - Graphic Era University Copernicus Benchmark
    # =========================================================================
    cells.append(make_cell("markdown", """## Step 9 — Real Copernicus Benchmark: Graphic Era University (Dehradun, India)

### Visualizing Adversarial Texture vs. Compound Loss on Real Terrain
This evaluation runs full-scene reflect-padded inference on the Graphic Era University capture (`Browser_images_new1.zip`).
It compares:
1. Native 10m Input (Bicubic 4x)
2. Pretrained Baseline MambaSR (Official Taco Foundation Weights)
3. Fine-Tuned Compound Loss MambaSR (Shift-Tolerant L1 + Sobel + Spectral)
4. Fine-Tuned Adversarial MambaSR PatchGAN (Ours: Spectral-Norm Discriminator + VGG Perceptual)"""))

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

# Radiometric calibration for Copernicus L2A
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

# Reflect-Padded Sliding Window Inference Engine for MambaSR (Zero Dark Corners)
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

print("Running full-scene reflect-padded inference on Graphic Era with MambaSR...")
ge_bicubic = F.interpolate(torch.from_numpy(ge_calibrated).unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False).squeeze(0).numpy()
ge_base    = run_reflect_padded_tiled_inference(baseline_model, ge_calibrated, patch_size=128, stride=48, scale=4)
ge_comp    = run_reflect_padded_tiled_inference(compound_model, ge_calibrated, patch_size=128, stride=48, scale=4)
ge_adv     = run_reflect_padded_tiled_inference(generator, ge_calibrated, patch_size=128, stride=48, scale=4)
print("Inference complete across all 4 resolution tiers.")"""
    cells.append(make_cell("code", c10))

    # =========================================================================
    # Cell 10: Step 10 - 4-Way Full-Scene Overview & Rooftop Zoom Comparison
    # =========================================================================
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
    (img_bas, "2. Pretrained Baseline MambaSR (2.5m)\\nOfficial Weights", 'black'),
    (img_cmp, "3. Fine-Tuned Compound MambaSR (2.5m)\\nShift-Tolerant L1 + Sobel", 'green'),
    (img_adv, "4. Fine-Tuned Adversarial MambaSR (Ours)\\nVGG Perceptual + Discriminator", 'darkorange')
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
ge_adv_png = os.path.join(RESULTS_DIR, 'sen2sr_full_adversarial_graphic_era_showcase.png')
plt.savefig(ge_adv_png, dpi=180, bbox_inches='tight')
plt.show()
print(f"Graphic Era University Full Adversarial showcase saved to: {ge_adv_png}")"""
    cells.append(make_cell("code", c11))

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_path = 'notebooks/06_full_model_experiments/06b_Full_Adversarial_Finetune.ipynb'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated: {out_path} ({len(cells)} cells)")

    # Validate AST
    for idx, c in enumerate(cells):
        if c['cell_type'] == 'code':
            code_str = "".join(c['source'])
            clean_lines = [l for l in code_str.split("\n") if not l.strip().startswith(('%pip', '!', '%'))]
            clean_code = "\n".join(clean_lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                raise SyntaxError(f"AST SyntaxError in cell {idx}: {e}")

    print("All cells in 06b passed AST validation!")

    gdrive_nb = "G:/My Drive/Project_Folder/Notebooks/06_full_model_experiments/06b_Full_Adversarial_Finetune.ipynb"
    if os.path.exists(os.path.dirname(gdrive_nb)):
        with open(gdrive_nb, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Synced to Google Drive: {gdrive_nb}")

    return True

if __name__ == '__main__':
    build_full_adversarial_notebook()
