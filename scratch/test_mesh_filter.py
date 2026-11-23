import tifffile, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load ROI_0000
lr_raw = tifffile.imread('G:/My Drive/Project_Folder/Datasets/sentinaiplow/ROI_0000_lr.tif').transpose(2, 0, 1)
hr_raw = tifffile.imread('G:/My Drive/Project_Folder/Datasets/sentinaiphigh/ROI_0000_hr.tif').transpose(2, 0, 1)

def norm_s(arr):
    p99 = np.percentile(arr[np.isfinite(arr)], 99)
    div = 10000.0 if p99 > 255.0 else (255.0 if p99 > 2.0 else 1.0)
    return np.clip(np.nan_to_num(arr / div, nan=0.0), 0.0, 1.0).astype(np.float32)

lr_4b = norm_s(lr_raw[:4])
hr_4b = norm_s(hr_raw[:4])

# Let's inspect the crop
c_lr = 32
cy = (121 - c_lr) // 2
cx = (121 - c_lr) // 2

# Scale HR 2x to match 8x target
hr_t = torch.from_numpy(hr_4b).unsqueeze(0).float()
hr_8x = F.interpolate(hr_t, size=(121 * 8, 121 * 8), mode='bicubic', align_corners=False).squeeze(0)
hr_crop = hr_8x[:, cy*8:(cy+c_lr)*8, cx*8:(cx+c_lr)*8].numpy()

# Load Stage 1 checkpoint to generate real Stage 1 output
import types, sys
# Register mamba fallback if needed
device = torch.device('cpu')

print('Testing edge-preserving anti-mesh filter on checkerboard pattern...')

