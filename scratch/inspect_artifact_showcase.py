from PIL import Image
import numpy as np

im = Image.open(r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\SEN2SR_1.5M_New1_GroundTruth_Showcase.png')
print('Size:', im.size)
arr = np.array(im)
print('Shape:', arr.shape)
print('Mean:', arr.mean())

