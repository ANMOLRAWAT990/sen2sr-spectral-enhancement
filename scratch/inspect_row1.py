from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

im = Image.open('scratch/row1_im1.png')
print("Row 1 image size:", im.size)
arr = np.array(im)[:, :, :3]
h, w, _ = arr.shape
pw = w // 4

p1_native = arr[:, :pw]
p2_base = arr[:, pw:2*pw]
p3_s1 = arr[:, 2*pw:3*pw]
p4_vhr = arr[:, 3*pw:]

print("Panels extracted:", p1_native.shape, p2_base.shape, p3_s1.shape, p4_vhr.shape)
