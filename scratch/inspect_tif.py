import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Let's inspect the raw VHR output from the pipeline before all those blurring passes!
# We can load the saved GeoTIFF or rerun on the raw data.
import rasterio

tif_path = r'G:\My Drive\Project_Folder\Results\sen2sr_cascaded_1.5m_vhr.tif'
with rasterio.open(tif_path) as src:
    vhr_tif = src.read()
    print("VHR GeoTIFF shape:", vhr_tif.shape, "range:", vhr_tif.min(), vhr_tif.max())

