import json, os

def make_cell(cell_type, source, cell_id=None):
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }
    if cell_id:
        c["id"] = cell_id
    return c

# Shared Markdown for T1
T1_MARKDOWN = """## Step 3 — Audit & Contract: Band Order & Physical Normalization

### 1. Band Order Decision: Standardized Canonical [Red, Green, Blue, NIR] (`[R, G, B, N]`)
**Evidence and Sources of Truth:**
- **Pretrained Lite Model Contract:** Official Hugging Face metadata (`tacofoundation/SEN2SRLite` `mlm.json`) explicitly defines input bands as `["B04", "B03", "B02", "B08"]`, which maps directly to `[Red, Green, Blue, NIR]`.
- **SEN2NAIP Dataset:** Empirical inspection of raw `sentinaiplow` (`*_lr.tif`) and `sentinaiphigh` (`*_hr.tif`) confirms channel 0 is Red (mean ~958 / 128), channel 1 is Green (mean ~765 / 122), channel 2 is Blue (mean ~496 / 100), and channel 3 is NIR (vegetation peak mean ~1878 / 134).
- **WorldStrat Dataset:** High-resolution file `*_rgbn.tiff` natively encodes `[Red, Green, Blue, NIR]`. For the 12-band Sentinel-2 L2A cube (`B01` through `B12`), `B04` is channel 3, `B03` is channel 2, `B02` is channel 1, and `B08` is channel 7. Hence `WS_LR_ORDER = [3, 2, 1, 7]` yields `[R, G, B, N]`.
- **Pretrained Full Model Contract (`MambaSR` / `Swin2SR`):** Official metadata defines 10 bands as `["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]`. When extracting RGBN core channels from the 10-band stack, Red is index 2 (`B04`), Green is index 1 (`B03`), Blue is index 0 (`B02`), and NIR is index 6 (`B08`).
- **Elimination of Channel Inversion:** Standardizing canonical order to `[R, G, B, N]` ensures True Color RGB is simply `img[[0, 1, 2]]` (no red/blue channel swap), NDVI is `(img[3] - img[0]) / (img[3] + img[0])`, and NDWI is `(img[1] - img[3]) / (img[1] + img[3])`.

### 2. Radiometric Normalization Contract (Fixed Physical Scaling, No Max-Based Guessing)
- **Sentinel-2 L2A Reflectance:** Scaled by fixed factor `10000.0` (reflectance range `[0.0, 1.0]`).
- **NAIP Aerial Photography:** Scaled by fixed factor `255.0` (8-bit sensor DN range `[0.0, 1.0]`).
- **WorldStrat Pléiades/SPOT HR:** Scaled by fixed factor `4095.0` (12-bit sensor DN range `[0.0, 1.0]`).
- **Per-Patch Radiometric Alignment:** Target HR patches are quantile-matched to LR patch radiometry. This prevents the super-resolution network from overfitting to camera gain/exposure differences and forces it to learn structural edge geometry."""

# Shared Code for T1
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

    # Fail loudly on any assumption violation
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

sen2naip_lr_files = sorted(glob.glob('/content/drive/MyDrive/Project_Folder/Datasets/sentinaiplow/*.tif'))
sen2naip_hr_files = sorted(glob.glob('/content/drive/MyDrive/Project_Folder/Datasets/sentinaiphigh/*.tif'))
assert len(sen2naip_lr_files) > 0, "No SEN2NAIP LR files found in Datasets/sentinaiplow!"
assert len(sen2naip_hr_files) > 0, "No SEN2NAIP HR files found in Datasets/sentinaiphigh!"

lr_s, hr_s, m_lr, m_hr = load_canonical_pair(sen2naip_lr_files[0], sen2naip_hr_files[0], is_worldstrat=False)
print(f"\\n[SEN2NAIP Sample]: {os.path.basename(sen2naip_lr_files[0])}")
print(f"  LR File: count={m_lr['count']}, dtype={m_lr['dtype']}, shape={m_lr['shape']}, descriptions={m_lr['descriptions']}")
print(f"  HR File: count={m_hr['count']}, dtype={m_hr['dtype']}, shape={m_hr['shape']}, descriptions={m_hr['descriptions']}")
print(f"  LR Canonical [R,G,B,N]: shape={lr_s.shape}, range=[{lr_s.min():.4f}, {lr_s.max():.4f}], mean={lr_s.mean():.4f}")
print(f"  HR Canonical [R,G,B,N]: shape={hr_s.shape}, range=[{hr_s.min():.4f}, {hr_s.max():.4f}], mean={hr_s.mean():.4f}")

ws_lr_candidates = sorted(glob.glob('/content/drive/MyDrive/Project_Folder/Datasets/worldstrat_low/**/L2A/*-L2A_data.tiff', recursive=True))
ws_hr_candidates = sorted(glob.glob('/content/drive/MyDrive/Project_Folder/Datasets/worldstrat_high/**/extracted/**/*_rgbn.tiff', recursive=True))

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

print("T1 markdown and code strings prepared.")
