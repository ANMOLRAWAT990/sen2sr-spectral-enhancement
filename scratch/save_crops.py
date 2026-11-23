from PIL import Image
import numpy as np

user_img = Image.open(r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\.user_uploaded\media_1794218471715_5e1104f1.jpg')
p1 = r'G:\My Drive\Project_Folder\Results\SEN2SR_1.5M_4Way_Comparison.png'
im1 = Image.open(p1)
p2 = r'G:\My Drive\Project_Folder\Results\SEN2SR_1.5M_New1_4Way_Comparison.png'
im2 = Image.open(p2)

# Save smaller downscaled crops to check
im1.crop((0, 0, im1.width, int(im1.height * 0.18))).save('scratch/row1_im1.png')
im2.crop((0, 0, im2.width, int(im2.height * 0.18))).save('scratch/row1_im2.png')
print("Saved row1 crops for comparison")

