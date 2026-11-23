from PIL import Image
import numpy as np

im = Image.open('scratch/photographic_clarity_showcase.png')
print("Showcase size:", im.size)
arr = np.array(im)
print("Showcase shape:", arr.shape)

