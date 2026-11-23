from PIL import Image
import numpy as np

img = Image.open(r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\.user_uploaded\media_1794218471715_5e1104f1.jpg')
arr = np.array(img)[:, :, :3].astype(np.float32)
h, w, _ = arr.shape
pw = w // 4

p1 = arr[:, :pw]
p2 = arr[:, pw:2*pw]
p3 = arr[:, 2*pw:3*pw]
p4 = arr[:, 3*pw:]

print("Diff p1 vs p2 MSE:", np.mean((p1 - p2)**2))
print("Diff p2 vs p3 MSE:", np.mean((p2 - p3)**2))
print("Diff p3 vs p4 MSE:", np.mean((p3 - p4)**2))
print("Diff p1 vs p4 MSE:", np.mean((p1 - p4)**2))

