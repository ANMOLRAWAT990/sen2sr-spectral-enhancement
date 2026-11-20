# Sentinel-2 Spectral Enhancement and Super-Resolution (SEN2SR)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Major Project — Spectral Enhancement**  
Fine-tuning and rigorous ground-truth evaluation of the **SEN2SR** (Sentinel-2 Super-Resolution) architecture for **10m $\rightarrow$ 2.5m (4×)** spatial upscaling, validated side-by-side against real coincident aerial photography (NAIP).

---

## 1. Project Goal & Scientific Scope

Sentinel-2 captures free, global multispectral imagery at **10m ground sampling distance (GSD)** for visible/NIR bands. At 10m per pixel, rural road corridors, field boundaries, riverbanks, and individual building footprints are heavily blended into mixed pixels.

### The Objective:
Reconstruct **10m Sentinel-2 bands to physically accurate 2.5m (4× GSD)** so that outputs reflect real high-resolution aerial imagery of the same terrain, while strictly preserving physical surface reflectance and multispectral indices (NDVI, NDWI).

### Resolution Boundary Notice:
> [!IMPORTANT]
> **Sub-2m (1m / 1.25m / 1.5m) is explicitly NOT a physical goal.**  
> A single 10m satellite pixel cannot mathematically or physically determine structures smaller than ~2.5m without hallucinating invented textures. Cascaded upsampling beyond 2.5m using interpolated targets results in smoothed or hallucinatory detail. This repository focuses strictly on physically grounded **10m $\rightarrow$ 2.5m super-resolution**, evaluated directly against high-resolution aerial ground truth.

---

## 2. Key Architectural Innovations & Audit Fixes

Following a comprehensive audit of the baseline pipeline, the following core corrections and enhancements are implemented:

1. **Standardized Canonical Band Ordering (`[Red, Green, Blue, NIR]`):**
   - Eliminates channel swaps between datasets and official pretrained weights (`tacofoundation/SEN2SRLite` and `tacofoundation/SEN2SR`).
   - True Color RGB is canonical `img[[0, 1, 2]]` without red/blue inversion.
   - Eliminates vegetation misclassification where farmland previously appeared blue and NDVI collapsed.
2. **Fixed Physical Radiometric Normalization:**
   - Sentinel-2 L2A surface reflectance scaled by `10000.0`.
   - NAIP 8-bit aerial photography scaled by `255.0`.
   - WorldStrat 12-bit sensor DN scaled by `4095.0`.
   - Target HR patches are quantile-matched to LR radiometry per patch, eliminating camera gain/exposure offsets and forcing the model to learn structural edge geometry.
3. **Geographical Scene-Stratified Split & Dynamic Cropping:**
   - Scene-level split (80% Train, 10% Val, 10% Test) by geographical scene ID prevents spatial data leakage.
   - Identical held-out test scenes saved to `Results/test_scenes.json` for shared evaluation across Lite and Full models.
   - Dynamic per-epoch random cropping and synchronized geometric augmentations (flips, 90° rotations) eliminate static patch memorization.
4. **Shift-Tolerant Multi-Objective Compound Loss:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{shift-L1}} + 0.20 \cdot \mathcal{L}_{\text{Sobel}} + 0.15 \cdot \left(\mathcal{L}_{\text{NDVI}} + 0.5 \cdot \mathcal{L}_{\text{NDWI}}\right)$$
   - Evaluates L1 reconstruction error over $\pm 2$ pixel sub-grid shifts to avoid false penalties from sensor misregistration.
   - Sobel edge gradient constraints sharpen building edges and road corridors.
   - NDVI and NDWI multi-index loss eliminates spectral bleeding and color distortion.
5. **PatchGAN Adversarial Fine-Tuning (Task T6):**
   - Spectral-normalized PatchGAN discriminator (`04b_Lite_Adversarial_Finetune.ipynb`) combined with VGG-19 perceptual feature supervision for crisp edge gradients.
6. **Pure-PyTorch Selective Scan Fallback for Mamba:**
   - Ensures `MambaSR` runs reliably in Google Colab (T4) without C++ ABI symbol crashes.
7. **2D Hann Spatial Window Blending:**
   - Seamless tiled inference with boundary-step correction completely eliminates border artifacts and tile seams.

---

## 3. Repository Structure

```text
sen2sr-spectral-enhancement/
├── CHANGELOG.md                                   # Comprehensive audit & fix history
├── README.md                                      # Project overview & documentation
├── requirements.txt                               # Dependencies
├── docs/
│   ├── COPERNICUS_DOWNLOAD_GUIDE.md               # Guide to exporting Copernicus scenes
│   └── LITERATURE_REVIEW.md                       # Theoretical & mathematical foundations
├── notebooks/
│   ├── 01_data_preparation/                       # Download & extraction of SEN2NAIP & WorldStrat
│   ├── 02_baseline_inference/                     # Pretrained SEN2SR Lite & Full verification
│   ├── 03_methodology_docs/                       # Loss formulation & methodology guides
│   ├── 04_finetuning_experiments/
│   │   ├── Train_SEN2SR_Lite_Compound_Loss.ipynb  # Lite fine-tuning with shift-tolerant loss
│   │   ├── SEN2SR_Baseline_vs_Finetuned_Comparison.ipynb # Dual-section benchmark & Copernicus eval
│   │   └── 04b_Lite_Adversarial_Finetune.ipynb    # PatchGAN adversarial fine-tuning
│   ├── 05_ultra_highres_1m_experiments/           # [ARCHIVED/DEPRECATED] Sub-2m experiments
│   │   └── README.md
│   ├── 06_full_model_experiments/
│   │   ├── Train_SEN2SR_Full_Compound_Loss.ipynb  # Full MambaSR fine-tuning with PyTorch fallback
│   │   ├── SEN2SR_Full_Baseline_vs_Finetuned_Comparison.ipynb # 10-band benchmark & Copernicus eval
│   │   └── README.md
│   └── 07_cascaded_1.5m_vhr_experiments/          # [ARCHIVED/DEPRECATED] 1.5m VHR experiments
│       └── README.md
└── results/
    ├── README.md                                  # Figure descriptions & metric summaries
    └── test_scenes.json                           # Canonical held-out test scene split
```

---

## 4. Model Tiers

| Feature | SEN2SR Lite (`04`) | Full Model `MambaSR` (`06`) | Adversarial `04b` (`04b`) |
| :--- | :--- | :--- | :--- |
| **Architecture** | `CNNSR` (6 ResBlocks, 24 ch) | `MambaSR` (SSM, 6 stages × 8 blocks) | `CNNSR` + Spectral PatchGAN |
| **Target Scale** | 10m $\rightarrow$ 2.5m (4×) | 10m $\rightarrow$ 2.5m (4×, 10 bands) | 10m $\rightarrow$ 2.5m (4×) |
| **Loss Function** | Shift-Tolerant Compound Loss | Shift-Tolerant Compound Loss | Compound + VGG + PatchGAN |
| **Colab Compatibility**| T4 GPU (Fast, ~4 min) | T4 GPU (Pure-PyTorch SSM Fallback) | T4 GPU (10 Epochs, ~8 min) |
| **Recommended Use**| General deployment & edge devices | Maximum spectral representation | Sharpest visual edge definition |

---

## 5. Execution Guide (Google Colab)

Every notebook is designed to run **top-to-bottom with zero manual code edits** and fail loudly on contract violations.

### Recommended Workflow:
1. **Train SEN2SR Lite:**
   Open and execute `notebooks/04_finetuning_experiments/Train_SEN2SR_Lite_Compound_Loss.ipynb`.
   - Runs audit cell (T1), creates scene-level split (T2), trains for 15 epochs with shift-tolerant compound loss (T3).
   - Saves `Results/sen2sr_lite_compound_best.pth` and `Results/test_scenes.json`.
2. **Benchmark Lite vs Baseline:**
   Open and execute `notebooks/04_finetuning_experiments/SEN2SR_Baseline_vs_Finetuned_Comparison.ipynb`.
   - Evaluates $N=6$ held-out patches with shared contrast stretch against real NAIP ground truth.
   - Displays 3× zooms, spatial error difference maps, and sanity benchmark table (PSNR/SSIM/NDVI-MAE).
   - Runs full-scene Copernicus Hann-window inference in Section 2.
3. **(Optional) Run Adversarial Step:**
   Open and execute `notebooks/04_finetuning_experiments/04b_Lite_Adversarial_Finetune.ipynb` to train the PatchGAN refiner.
4. **Train & Evaluate Full Model:**
   Execute `notebooks/06_full_model_experiments/Train_SEN2SR_Full_Compound_Loss.ipynb` and `SEN2SR_Full_Baseline_vs_Finetuned_Comparison.ipynb`.

---

## 6. Citation & Acknowledgments

- **SEN2SR:** Taco Foundation ([Hugging Face Model Hub](https://huggingface.co/tacofoundation/SEN2SR)).
- **SEN2NAIP Dataset:** Coincident Sentinel-2 and USDA NAIP aerial photography.
- **WorldStrat Dataset:** Global paired Sentinel-2 and high-resolution SPOT/Pléiades imagery.
