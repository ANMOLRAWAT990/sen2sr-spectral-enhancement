from PIL import Image
import numpy as np

im1 = Image.open('scratch/cell11_output_from_gdrive_nb.png')
print('GDrive nb image size:', im1.size)

# Let's inspect user uploaded images
for name in ['media_1793863216854_7beae066.jpg', 'media_1793865037981_2c532f52.jpg']:
    p = f'C:/Users/AnmolRawat/.gemini/antigravity/brain/354ddd0a-1e4c-4bf6-934a-07dcb8863431/.user_uploaded/{name}'
    im = Image.open(p)
    print(name, im.size)

