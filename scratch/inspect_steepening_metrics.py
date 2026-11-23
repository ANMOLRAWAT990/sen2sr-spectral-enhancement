from PIL import Image
import numpy as np
import cv2

im = Image.open('scratch/edge_steepening_test.png')
print("Image size:", im.size)
arr = np.array(im)[:, :, :3]
w = arr.shape[1] // 3

for i, name in enumerate(["Regression 'Enhanced Blur'", "PDE Shock Filter", "Morphological Boundary Engine"]):
    p = arr[100:-50, i*w+50:(i+1)*w-50]
    gray = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Also calculate Sobel gradient magnitude (edge outline sharpness)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.mean(np.sqrt(sx**2 + sy**2))
    print(f"{name:32s}: Edge Gradient Mag={grad_mag:.2f}, Laplacian Sharpness={lap:.1f}")

