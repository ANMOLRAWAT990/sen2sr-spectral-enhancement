from PIL import Image
import numpy as np
import cv2

im = Image.open('scratch/tone_and_clarity_test.png')
print("Image size:", im.size)
arr = np.array(im)
w = arr.shape[1] // 4
for i, name in enumerate(["Current Blown-out", "Gamma 1.6", "CLAHE LAB", "Photographic VHR"]):
    p = arr[100:-50, i*w+50:(i+1)*w-50, :3]
    gray = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"{name:20s}: Mean={gray.mean():.1f}, Contrast(Std)={gray.std():.1f}, Sharpness={lap_var:.1f}")
