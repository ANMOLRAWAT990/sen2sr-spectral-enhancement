from PIL import Image
import numpy as np

im = Image.open('scratch/cell11_output_from_gdrive_nb.png')
arr = np.array(im)[:, :, :3]
w = arr.shape[1]
p2 = arr[100:500, w//4+50:w//2-50].astype(np.float32)
p3 = arr[100:500, w//2+50:3*w//4-50].astype(np.float32)

# Check FFT 2D peak power at Nyquist / pixel-shuffle frequencies
for name, p in [('Stage 1', p2), ('Stage 2', p3)]:
    gray = p.mean(axis=2)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    print(f"{name}: mean={gray.mean():.2f}, std={gray.std():.2f}, max_f={mag.max():.2e}")

