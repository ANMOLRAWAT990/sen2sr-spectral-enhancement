import os, glob, numpy as np
import tifffile

SEN2NAIP_DIR_LR = 'G:/My Drive/Project_Folder/Datasets/sentinaiplow'
SEN2NAIP_DIR_HR = 'G:/My Drive/Project_Folder/Datasets/sentinaiphigh'
WS_DIR_LR = 'G:/My Drive/Project_Folder/Datasets/worldstrat_low'
WS_DIR_HR = 'G:/My Drive/Project_Folder/Datasets/worldstrat_high/extracted'

SEN2NAIP_LR_ORDER = [0, 1, 2, 3] # [Red, Green, Blue, NIR]
SEN2NAIP_HR_ORDER = [0, 1, 2, 3] # [Red, Green, Blue, NIR]
WS_LR_ORDER = [3, 2, 1, 7]       # [B04=Red, B03=Green, B02=Blue, B08=NIR]
WS_HR_ORDER = [0, 1, 2, 3]       # [Red, Green, Blue, NIR]

def read_raster_robust(path):
    # Try rasterio first, fallback to tifffile
    try:
        import rasterio
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
    except ImportError:
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] in [4, 12]:
            arr = arr.transpose(2, 0, 1) # to (C, H, W)
        meta = {
            'driver': 'tifffile',
            'count': arr.shape[0],
            'dtype': str(arr.dtype),
            'shape': (arr.shape[1], arr.shape[2]),
            'crs': 'EPSG (embedded)',
            'descriptions': None
        }
        return arr, meta

def load_canonical_sample(lr_path, hr_path, is_worldstrat=False):
    lr_raw, lr_meta = read_raster_robust(lr_path)
    hr_raw, hr_meta = read_raster_robust(hr_path)
    
    # 1. Band Reordering to [Red, Green, Blue, NIR]
    if is_worldstrat:
        lr_4b = lr_raw[WS_LR_ORDER]
        hr_4b = hr_raw[WS_HR_ORDER]
        # WorldStrat LR is already BOA reflectance [0.0, 1.0]
        lr_scaled = np.clip(lr_4b, 0.0, 1.0)
        # WorldStrat HR is SPOT/Pléiades 12-bit DN (divide by 4095.0)
        hr_scaled = np.clip(hr_4b / 4095.0, 0.0, 1.0)
    else:
        lr_4b = lr_raw[SEN2NAIP_LR_ORDER]
        hr_4b = hr_raw[SEN2NAIP_HR_ORDER]
        # SEN2NAIP LR is Sentinel-2 L2A surface reflectance scaled by 10,000
        lr_scaled = np.clip(lr_4b / 10000.0, 0.0, 1.0)
        # SEN2NAIP HR is NAIP 8-bit digital numbers (0-255)
        hr_scaled = np.clip(hr_4b / 255.0, 0.0, 1.0)
        
    # Shape and range assertions
    assert lr_scaled.shape[0] == 4, f"Expected 4 channels for LR, got {lr_scaled.shape}"
    assert hr_scaled.shape[0] == 4, f"Expected 4 channels for HR, got {hr_scaled.shape}"
    assert not np.isnan(lr_scaled).any(), "NaN found in LR sample"
    assert not np.isnan(hr_scaled).any(), "NaN found in HR sample"
    assert 0.0 <= lr_scaled.min() and lr_scaled.max() <= 1.0, f"LR value out of range [0, 1]: [{lr_scaled.min()}, {lr_scaled.max()}]"
    assert 0.0 <= hr_scaled.min() and hr_scaled.max() <= 1.0, f"HR value out of range [0, 1]: [{hr_scaled.min()}, {hr_scaled.max()}]"
    
    return lr_scaled, hr_scaled, lr_meta, hr_meta

# Test with real files
lr_f = sorted(glob.glob(os.path.join(SEN2NAIP_DIR_LR, '*.tif')))[0]
hr_f = sorted(glob.glob(os.path.join(SEN2NAIP_DIR_HR, '*.tif')))[0]
lr_s, hr_s, m_lr, m_hr = load_canonical_sample(lr_f, hr_f, is_worldstrat=False)
print("SEN2NAIP Canonical Sample Verified:")
print(f"  LR shape: {lr_s.shape}, range: [{lr_s.min():.4f}, {lr_s.max():.4f}], mean: {lr_s.mean():.4f}")
print(f"  HR shape: {hr_s.shape}, range: [{hr_s.min():.4f}, {hr_s.max():.4f}], mean: {hr_s.mean():.4f}")

ws_lr_candidates = sorted(glob.glob(os.path.join(WS_DIR_LR, '**/L2A/*-L2A_data.tiff'), recursive=True))
ws_hr_candidates = sorted(glob.glob(os.path.join(WS_DIR_HR, '**/*_rgbn.tiff'), recursive=True))
if ws_lr_candidates and ws_hr_candidates:
    lr_ws, hr_ws, m_lws, m_hws = load_canonical_sample(ws_lr_candidates[0], ws_hr_candidates[0], is_worldstrat=True)
    print("WorldStrat Canonical Sample Verified:")
    print(f"  LR shape: {lr_ws.shape}, range: [{lr_ws.min():.4f}, {lr_ws.max():.4f}], mean: {lr_ws.mean():.4f}")
    print(f"  HR shape: {hr_ws.shape}, range: [{hr_ws.min():.4f}, {hr_ws.max():.4f}], mean: {hr_ws.mean():.4f}")
