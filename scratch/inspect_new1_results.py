from PIL import Image

im = Image.open(r'G:\My Drive\Project_Folder\Results\SEN2SR_1.5M_New1_GroundTruth_Showcase.png')
print('Showcase image size:', im.size)
im2 = Image.open(r'G:\My Drive\Project_Folder\Results\SEN2SR_1.5M_New1_4Way_Comparison.png')
print('4Way comparison size:', im2.size)

