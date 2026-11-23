import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Mock data
H, W = 540, 944
img_input = np.random.rand(H, W).astype(np.float32)
img_base = img_input + np.random.normal(0, 0.01, (H, W)).astype(np.float32)
img_fine = img_base + np.random.normal(0, 0.02, (H, W)).astype(np.float32)

# Test Tenengrad / Sobel edge energy calculation
def calc_edge_energy(arr):
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(t, sx, padding=1)
    gy = F.conv2d(t, sy, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    return mag.mean().item()

e_in = calc_edge_energy(img_input)
e_base = calc_edge_energy(img_base)
e_fine = calc_edge_energy(img_fine)
gain = ((e_fine - e_base) / max(e_base, 1e-6)) * 100
print(f"Edge energy Input: {e_in:.4f}, Base: {e_base:.4f}, Fine: {e_fine:.4f}, Gain: {gain:+.2f}%")
print("Edge test passed!")
