import zipfile
from PIL import Image

with zipfile.ZipFile('data/Browser_images_new1.zip') as z:
    for name in z.namelist():
        if name.endswith('.tif') or name.endswith('.tiff'):
            with z.open(name) as f:
                im = Image.open(f)
                print('Browser_images_new1:', name, im.size) # width, height
            break

with zipfile.ZipFile('G:/My Drive/Project_Folder/Datasets/Browser_images.zip') as z:
    for name in z.namelist():
        if name.endswith('.tif') or name.endswith('.tiff'):
            with z.open(name) as f:
                im = Image.open(f)
                print('Browser_images:', name, im.size) # width, height
            break

