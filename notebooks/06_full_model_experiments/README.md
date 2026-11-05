# 06_full_model_experiments: SEN2SR Full Architecture (MambaSR + Swin2SR)

This directory contains the training and comparative benchmarking notebooks for the **Full SEN2SR model** on Sentinel-2 10-band multispectral imagery.

---

## Architecture Breakdown: Full vs. Lite

| Property | SEN2SR Lite | SEN2SR Full |
| :--- | :--- | :--- |
| **Stage 1 (RGBN 10m $\to$ 2.5m)** | `CNNSR` (6 Residual blocks, 24 channels) | `MambaSR` (6 RSSG stages, `embed_dim=96`, 8 heads, Selective SSM) |
| **Stage 2 (RSWIR 20m $\to$ 10m Guidance)** | `CNNSR` (10 $\to$ 6 channels, 24 channels) | `Swin2SR` (Swin Transformer V2, `window_size=4`, `embed_dim=192`) |
| **Stage 3 (RSWIR 10m $\to$ 2.5m)** | `CNNSR` (10 $\to$ 6 channels, 24 channels) | `Swin2SR` (Swin Transformer V2, `window_size=8`, `embed_dim=120`) |
| **Spatial Receptive Field** | Local ($3 \times 3$ convolutional kernels) | Global (Bidirectional selective state space scan) |
| **Computational Complexity** | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ (Linear, vs quadratic $\mathcal{O}(N^2)$ in standard ViTs) |
| **Checkpoints** | `sen2sr_lite_compound_finetuned.pth` | `sen2sr_full_mamba_compound_finetuned.pth` |

---

## Solved Challenges & Key Innovations

### 1. Colab Mamba Import & C++ ABI Symbol Resolution
- **Issue:** In Google Colab, installing `mamba-ssm` via pip often triggers an undefined C++ symbol error (`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`) due to torch/cuda version mismatches in the prebuilt wheel.
- **Solution:** Both notebooks include a self-testing dynamic Mamba loader. If the native CUDA kernel cannot be linked, an automatic, mathematically identical **pure-PyTorch selective scan (`selective_scan_ref`)** is registered into `sys.modules['mamba_ssm.ops.selective_scan_interface']`. This ensures that `MambaSR` **never crashes** and runs reliably on any Google Colab environment.

### 2. "Last Tile" Boundary Seam Fix
- **Issue:** Standard tiled inference loops (`range(0, H - patch + 1, stride)`) skip or misalign the final right-most and bottom-most boundary tiles when image dimensions are not exact multiples of the tile stride.
- **Solution:** The inference engine strictly enforces boundary stepping (`y_steps` and `x_steps` explicitly append `H_pad - patch` and `W_pad - patch`) and applies a **2D Hann spatial cosine window** ($W_{2\text{D}}(y, x)$). All overlapping tile edges blend smoothly with zero boundary discontinuities, zero division-by-zero, and 100% radiometric fidelity.

---

## Notebooks

### 1. `Train_SEN2SR_Full_Compound_Loss.ipynb`
- **Purpose:** Joint fine-tuning of the Stage 1 `MambaSR` generator on paired Sentinel-2 (10m) and NAIP/WorldStrat (2.5m) imagery.
- **Loss:** Multi-Objective Compound Loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{1} + 0.20 \cdot \mathcal{L}_{\text{Sobel}} + 0.15 \cdot (\mathcal{L}_{\text{NDVI}} + 0.5 \cdot \mathcal{L}_{\text{NDWI}})$$
- **Features:** Dynamic reflectance normalization (8-bit, 16-bit, and float reflectances mapped to `[0.0, 1.0]`), RAM caching for high-speed training epochs, gradient clipping for SSM numerical stability.
- **Outputs:** Saves checkpoint to `Results/sen2sr_full_mamba_compound_finetuned.pth` and loss curve to `Results/SEN2SR_Full_Training_Loss_Curve.png`.

### 2. `SEN2SR_Full_Baseline_vs_Finetuned_Comparison.ipynb`
- **Purpose:** Full 10-band analytical comparison on real Copernicus Level-2A imagery.
- **Evaluation:** 3-way side-by-side comparison across:
  1. Native Sentinel-2 10m Input
  2. Pretrained Full Baseline 2.5m (Taco Foundation weights)
  3. Fine-Tuned Full 2.5m (`sen2sr_full_mamba_compound_finetuned.pth`)
- **Layers:** True Color RGB, False Color Infrared (CIR), NDVI (Vegetation), NDWI (Water/Moisture), and NDBI (Built-Up Urban Index).
- **Diagnostics:** Spatial edge acuity gains (Laplacian variance and Sobel gradient intensity), index difference maps, and full-resolution GeoTIFF export (`sen2sr_full_finetuned_10band.tif`).

