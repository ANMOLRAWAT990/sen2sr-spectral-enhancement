with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = [
    ("axes[0, 0].set_ylabel('Overall Multispectral\\n(10-Band Mean)'", "axes[0, 0].set_ylabel('Overall Multispectral\\\\n(10-Band Mean)'"),
    ("axes[1, 0].set_ylabel('True Color RGB\\n(B04, B03, B02)'", "axes[1, 0].set_ylabel('True Color RGB\\\\n(B04, B03, B02)'"),
    ("axes[2, 0].set_ylabel('Color Infrared (CIR)\\n(B08, B04, B03)'", "axes[2, 0].set_ylabel('Color Infrared (CIR)\\\\n(B08, B04, B03)'"),
    ("axes[3, 0].set_ylabel('NDVI\\n(Vegetation)'", "axes[3, 0].set_ylabel('NDVI\\\\n(Vegetation)'"),
    ("axes[4, 0].set_ylabel('NDWI\\n(Water / Moisture)'", "axes[4, 0].set_ylabel('NDWI\\\\n(Water / Moisture)'"),
    ("axes[5, 0].set_ylabel('NDBI\\n(Built-up / Buildings)'", "axes[5, 0].set_ylabel('NDBI\\\\n(Built-up / Buildings)'"),
    ("Blurry mixed pixels | Edge: {edge_in:.4f}'", "Blurry mixed pixels | Edge: {edge_in:.4f}'"),
    ("Baseline | Edge: {edge_s1:.4f}'", "Baseline | Edge: {edge_s1:.4f}'"),
    ("Zero Mesh, Calibrated | Edge: {edge_vhr:.4f}'", "Zero Mesh, Calibrated | Edge: {edge_vhr:.4f}'"),
    ("Reference | Edge: {edge_gt:.4f}'", "Reference | Edge: {edge_gt:.4f}'"),
]

for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        print(f"Replaced {old[:30]}...")

with open('scratch/build_1_5meter_experiments.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Saved scratch/build_1_5meter_experiments.py")

