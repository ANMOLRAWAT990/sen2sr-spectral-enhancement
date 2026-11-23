import tifffile
import numpy as np

arr = tifffile.imread(r'G:\My Drive\Project_Folder\Results\sen2sr_cascaded_1.5m_vhr.tif')
print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("Min:", arr.min(), "Max:", arr.max(), "Mean:", arr.mean())
for c in range(min(4, arr.shape[0])):
    print(f"Band {c}: min={arr[c].min():.4f}, max={arr[c].max():.4f}, mean={arr[c].mean():.4f}, p2={np.percentile(arr[c], 2):.4f}, p98={np.percentile(arr[c], 98):.4f}")

