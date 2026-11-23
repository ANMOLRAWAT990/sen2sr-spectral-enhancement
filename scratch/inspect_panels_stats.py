from PIL import Image
import numpy as np

im = Image.open('scratch/photographic_clarity_showcase.png')
arr = np.array(im)[:, :, :3]
w = arr.shape[1] // 4
for i, name in enumerate(["1. Blown-out & Blurred", "2. Balanced Gamma 1.4", "3. High-Clarity VHR", "4. Ultra-Crisp VHR"]):
    p = arr[100:-50, i*w+50:(i+1)*w-50]
    print(f"{name}: mean={p.mean():.1f}, std={p.std():.1f}, min={p.min()}, max={p.max()}")

