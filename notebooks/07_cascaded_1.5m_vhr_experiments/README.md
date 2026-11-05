# Folder 07: Cascaded 1.5m VHR (Very High Resolution) Super-Resolution Experiments
### Major Project: Progressive Two-Stage Satellite Spatial-Spectral Refinement (10m → 2.5m → 1.5m GSD)

---

## 1. Overview & Motivation
Standard single-step super-resolution directly from 10m to sub-2m introduces blurring and hallucination artifacts because the scaling factor ($>6\\times$) exceeds the physical Nyquist sampling limit.

In this experiment folder, we implement a **Progressive Two-Stage Cascaded Pipeline**:
$$\\text{Stage 1: } 10\\text{m} \\xrightarrow{\\text{4x Full SEN2SR (MambaSR)}} 2.5\\text{m} \\quad \\longrightarrow \\quad \\text{Stage 2: } 2.5\\text{m} \\xrightarrow{\\text{2x Deep VHR Refiner}} \\mathbf{1.25\\text{m} - 1.5\\text{m} \\text{ VHR}}$$

---

## 2. Key Architectural Innovations
1. **Zero-Initialized Residual Head (Anti-Brightening Guarantee):**
   - The final convolution layer of the Stage 2 Refiner is initialized to zero (`nn.init.zeros_`).
   - The refiner begins at the exact identity mapping ($\Delta = 0$) and only learns zero-mean high-frequency edge corrections, completely eliminating artificial brightening or washed-out pixels.
2. **Luminance Conservation Loss:**
   - $\\mathcal{L}_{\\text{luminance}} = |\\bar{y}_{\\text{pred}} - \\bar{y}_{\\text{true}}|$ strictly penalizes radiometric drift.
3. **Urban Building Delineation:**
   - Evaluates urban building footprints, rooftop geometry, and street networks against native aerial NAIP ground truth.
4. **Pure FP32 Stable Tiled Inference:**
   - Single-tile fast-path (~20s execution time) and multi-tile Hann window blending, free from FP16 recurrence overflows.

---

## 3. Notebooks in this Folder
1. **`Train_SEN2SR_Cascaded_1.5m_VHR.ipynb`**:
   - Fast Stage 2 Refiner training with Stage 1 feature caching (completes in ~2–3 minutes on Colab GPU).
   - Saves checkpoint: `sen2sr_cascaded_1.5m_refiner.pth`.
2. **`SEN2SR_1.5Meter_MultiStage_Comparison.ipynb`**:
   - 4-Way Comprehensive Multi-Modal Evaluation:
     1. Native Sentinel-2 (10m Input)
     2. Stage 1 Baseline Full (2.5m)
     3. Stage 1 Fine-Tuned (2.5m, Ours)
     4. Stage 2 Cascaded VHR (1.5m/1.25m, Ours)
   - Visual matrix across all 6 layers (Overall Multispectral, RGB, CIR, NDVI, NDWI, NDBI).
   - Building footprint extraction, 1D spatial transect profile, and quantitative acuity table.
   - Exports 1.5m GIS-ready GeoTIFF: `sen2sr_cascaded_1.5m_vhr.tif`.
