import zipfile, rasterio, io

with zipfile.ZipFile('data/Browser_images_new1.zip') as z:
    for name in z.namelist():
        if name.endswith('.tif') or name.endswith('.tiff'):
            with z.open(name) as f:
                with rasterio.open(f) as src:
                    print('Browser_images_new1:', name, src.shape)
            break

with zipfile.ZipFile('G:/My Drive/Project_Folder/Datasets/Browser_images.zip') as z:
    for name in z.namelist():
        if name.endswith('.tif') or name.endswith('.tiff'):
            with z.open(name) as f:
                with rasterio.open(f) as src:
                    print('Browser_images:', name, src.shape)
            break

