from PIL import Image
import numpy as np

user_img = Image.open(r'C:\Users\AnmolRawat\.gemini\antigravity\brain\354ddd0a-1e4c-4bf6-934a-07dcb8863431\.user_uploaded\media_1794218471715_5e1104f1.jpg')
user_arr = np.array(user_img)

# Let's check G:/My Drive/Project_Folder/Results/SEN2SR_1.5M_4Way_Comparison.png
p1 = r'G:\My Drive\Project_Folder\Results\SEN2SR_1.5M_4Way_Comparison.png'
im1 = Image.open(p1)
print("SEN2SR_1.5M_4Way_Comparison size:", im1.size)

# Let's check G:/My Drive/Project_Folder/Results/SEN2SR_1.5M_New1_4Way_Comparison.png
p2 = r'G:\My Drive\Project_Folder\Results\SEN2SR_1.5M_New1_4Way_Comparison.png'
im2 = Image.open(p2)
print("SEN2SR_1.5M_New1_4Way_Comparison size:", im2.size)

# Compare top row of im1 and im2 with user_arr
row1_im1 = im1.crop((0, 0, im1.width, im1.height // 6))
row1_im2 = im2.crop((0, 0, im2.width, im2.height // 6))
print("Row 1 im1 size:", row1_im1.size)
print("Row 1 im2 size:", row1_im2.size)

