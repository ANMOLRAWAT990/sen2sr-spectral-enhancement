import os, json

# Search all ipynb files in G: Drive and check their size, mtime, and whether they have cell1_code with Browser_images_new1
import glob

matches = glob.glob('G:/My Drive/**/*.ipynb', recursive=True)
print(f"Total ipynb in G: Drive: {len(matches)}")
for m in matches:
    if '07_cascaded' in m or '1.5M' in m or '1.5Meter' in m or 'Colab' in m:
        try:
            nb = json.load(open(m, encoding='utf-8'))
            s = str(nb)
            has_new1 = 'Browser_images_new1.zip' in s
            has_cv2_cell7 = 'import cv2' in s
            print(f"{m}:\n  Size: {os.path.getsize(m)}, Mtime: {os.path.getmtime(m)}\n  has_new1: {has_new1}, has_cv2: {has_cv2_cell7}")
        except Exception as e:
            print(f"{m}: Error {e}")

