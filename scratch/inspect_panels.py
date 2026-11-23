from PIL import Image
import numpy as np

im = Image.open('scratch/cell11_output_from_gdrive_nb.png')
arr = np.array(im)
print('Shape:', arr.shape)
# Top 40 rows: title area
print('Mean RGB:', arr.mean(axis=(0,1)))
# Check middle slice of 4 panels
w = arr.shape[1]
panel1 = arr[:, :w//4]
panel2 = arr[:, w//4:w//2]
panel3 = arr[:, w//2:3*w//4]
panel4 = arr[:, 3*w//4:]
print('Panel 1 mean RGB:', panel1.mean(axis=(0,1)))
print('Panel 2 mean RGB:', panel2.mean(axis=(0,1)))
print('Panel 3 mean RGB:', panel3.mean(axis=(0,1)))
print('Panel 4 mean RGB:', panel4.mean(axis=(0,1)))

