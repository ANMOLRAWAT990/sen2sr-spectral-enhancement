"""
Script for Task T5:
- Prepend deprecation banner to all notebooks in 05 and 07
- Update READMEs in 05 and 07 to document physical limitations and remove inflated claims
- Sync to Google Drive
"""

import json, os

BANNER_MD = [
    "# > [!WARNING]\n",
    "# > **DEPRECATED: Stage-2 target is the 2.5m NAIP interpolated 2x, so no real sub-2.5m detail exists. Do not run unless real 1m aerial pairs are supplied.**\n",
    "\n",
    "Physical Sentinel-2 10m pixels cannot mathematically resolve structures smaller than ~2.5m. Cascaded upsampling beyond 2.5m using 2x-interpolated targets produces smoothed or hallucinated textures rather than genuine physical resolution recovery. Refer to `04_finetuning_experiments` and `06_full_model_experiments` for physically grounded 10m -> 2.5m super-resolution."
]

notebook_paths = [
    'notebooks/05_ultra_highres_1m_experiments/Train_1Meter_Cascaded_Refiner.ipynb',
    'notebooks/05_ultra_highres_1m_experiments/SEN2SR_1Meter_4Way_Comparison.ipynb',
    'notebooks/07_cascaded_1.5m_vhr_experiments/Train_SEN2SR_Cascaded_1.5m_VHR.ipynb',
    'notebooks/07_cascaded_1.5m_vhr_experiments/SEN2SR_1.5Meter_MultiStage_Comparison.ipynb',
]

def add_deprecation_banner(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    first_cell_src = "".join(nb['cells'][0].get('source', [])) if nb['cells'] else ""
    if "DEPRECATED: Stage-2 target is the 2.5m NAIP" not in first_cell_src:
        banner_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": BANNER_MD
        }
        nb['cells'].insert(0, banner_cell)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Added deprecation banner to: {path}")
    else:
        print(f"Deprecation banner already present in: {path}")

    # Mirror to G: drive if exists
    gdrive_path = path.replace('notebooks/', 'G:/My Drive/Project_Folder/Notebooks/')
    if os.path.exists(os.path.dirname(gdrive_path)):
        with open(gdrive_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Synced deprecated notebook to G Drive: {gdrive_path}")

for p in notebook_paths:
    add_deprecation_banner(p)

# Update README in 05
README_05 = """# [ARCHIVED / DEPRECATED] Stage 2: Cascaded 1.25m Refiner Experiments

> [!WARNING]
> **DEPRECATED / ARCHIVED EXPERIMENTAL CODE**
> **Scientific Limitation:** The Stage-2 training target in these experiments was standard 2.5m NAIP aerial photography interpolated 2x (bicubic/bilinear), rather than real sub-meter sensor captures. Consequently, no genuine sub-2.5m physical detail exists in the training data, and the network predominantly learned smoothed approximations.
> 
> A single native 10m Sentinel-2 pixel cannot physically resolve ground structures smaller than ~2.5m without hallucinated textures. For the physically grounded, peer-reviewed model, refer to **`04_finetuning_experiments`** and **`06_full_model_experiments`** (10m -> 2.5m).

---

## Retrospective Summary of Experiment

This folder preserves the historical code used to explore multi-stage cascaded refinement:
1. **Stage 1 (4x Reconstruction):** 10m to 2.5m multispectral upscaling.
2. **Stage 2 (2x Spatial Refiner):** 2.5m to ~1.25m residual dense refinement.

### Experimental Files Preserved:
- **`Train_1Meter_Cascaded_Refiner.ipynb`**: Archived training script.
- **`SEN2SR_1Meter_4Way_Comparison.ipynb`**: Archived comparative evaluation.
"""

with open('notebooks/05_ultra_highres_1m_experiments/README.md', 'w', encoding='utf-8') as f:
    f.write(README_05)
print("Updated notebooks/05_ultra_highres_1m_experiments/README.md")

# Update README in 07
README_07 = """# [ARCHIVED / DEPRECATED] Folder 07: Cascaded 1.5m VHR Refinement Experiments

> [!WARNING]
> **DEPRECATED / ARCHIVED EXPERIMENTAL CODE**
> **Scientific Limitation:** The Stage-2 target used in these experiments was interpolated from 2.5m NAIP aerial imagery. Without true sub-meter aerial imagery, cascaded upscaling past 2.5m yields interpolated edge contrast rather than verified physical resolution. Claims of "ground-truth level clarity" or "+87% acuity gains" reflect contrast-inflated gradient metrics, not genuine geometric accuracy.
> 
> The core validated objective of this project is **10m -> 2.5m super-resolution**, fully documented and benchmarked against real NAIP aerial truth in **`04_finetuning_experiments`** and **`06_full_model_experiments`**.

---

## Retrospective Summary of Experiment

This folder contains experimental exploration of perceptual feature loss (VGG-19) and Laplacian curvature constraints on cascaded refinement:
- Stage 1: 10m -> 2.5m via Full SEN2SR (`MambaSR`).
- Stage 2: 2.5m -> 1.5m via deep residual refiner.

### Preserved Notebooks:
- **`Train_SEN2SR_Cascaded_1.5m_VHR.ipynb`**: Archived training notebook.
- **`SEN2SR_1.5Meter_MultiStage_Comparison.ipynb`**: Archived multi-stage comparison notebook.
"""

with open('notebooks/07_cascaded_1.5m_vhr_experiments/README.md', 'w', encoding='utf-8') as f:
    f.write(README_07)
print("Updated notebooks/07_cascaded_1.5m_vhr_experiments/README.md")
