from PIL import Image
import numpy as np

img_path = r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\.user_uploaded\media_1794218471715_5e1104f1.jpg'
im = Image.open(img_path)
print("Image size:", im.size)
arr = np.array(im)
print("Image shape:", arr.shape)

# Let's inspect the 4 panels across the width
w = arr.shape[1]
h = arr.shape[0]
pw = w // 4

for i in range(4):
    panel = arr[:, i*pw:(i+1)*pw, :3]
    print(f"Panel {i+1}: mean={panel.mean(axis=(0,1))}, std={panel.std(axis=(0,1))}, min={panel.min()}, max={panel.max()}")

