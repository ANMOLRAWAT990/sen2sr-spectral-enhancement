# Folder 07: Cascaded 1.5m VHR (Very High Resolution) Super-Resolution Experiments
### Major Project: Ground-Truth Level Clarity via Perceptual Feature Loss & Multi-Scale Spectrum Supervision

---

## 1. Overview & Scientific Rationale
Standard single-stage super-resolution relying exclusively on $L_1$ pixel regression suffers from the **"Regression-to-the-Mean"** problem: when high-frequency textures (individual tree crowns, building roof edges, crop furrow stripes) are ambiguous, $L_1$ forces the network to output a blurry average.

In this experiment, we deploy:
$$\text{Stage 1: } 10\text{m} \xrightarrow{\text{4x Full SEN2SR (MambaSR)}} 2.5\text{m} \quad \longrightarrow \quad \text{Stage 2: } 2.5\text{m} \xrightarrow{\text{Deep VHR Refiner}} \mathbf{1.25\text{m} - 1.5\text{m} \text{ VHR}}$$

---

## 2. Core Breakthroughs in this Folder
1. **Perceptual Feature Supervision (VGG-19 Deep Textures):**
   - Incorporates deep VGG-19 feature loss ($\text{relu3\_3}$) to penalize blurry averages and force realistic spatial textures.
2. **2nd-Order Laplacian Curvature Loss:**
   - Supervises surface edge curvature directly to recover razor-sharp building corners and road corridors.
3. **Dual-Stage Anti-Checkerboard PixelShuffle:**
   - Double LeakyReLU + Conv3x3 after PixelShuffle eliminates periodic sub-pixel mesh artifacts.
4. **Strict Zero-Brightening Guarantee:**
   - Zero-initialized residual head (`nn.init.zeros_`) and luminance conservation loss $|\bar{y}_{\text{pred}} - \bar{y}_{\text{true}}|$ strictly guarantee $0.00\%$ brightness shift.
5. **Scaled Dataset (250+ Patches) with 8× Spatial Augmentations:**
   - Trained across residential buildings, smallholder farms, canals, and roads with random flips and $90^\circ$ rotations.

---

## 3. Notebooks in this Folder
1. **`Train_SEN2SR_Cascaded_1.5m_VHR.ipynb`**:
   - Fast VHR Refiner training with Stage 1 foundation feature caching (completes in ~3 minutes on Colab GPU).
   - Saves checkpoint: `sen2sr_cascaded_1.5m_refiner.pth`.
2. **`SEN2SR_1.5Meter_MultiStage_Comparison.ipynb`**:
   - 4-Way Comprehensive Multi-Modal Evaluation:
     1. Native Sentinel-2 (10m Input)
     2. Stage 1 Baseline Full (2.5m)
     3. Stage 1 Fine-Tuned (2.5m, Ours)
     4. Stage 2 Cascaded VHR (1.5m/1.25m, Ours)
   - Visual matrix across all 6 layers (Overall Multispectral, RGB, CIR, NDVI, NDWI, NDBI).
   - Ground-Truth Aerial Showcase comparing directly against NAIP optical ground truth.
   - Exports 1.5m GIS-ready GeoTIFF: `sen2sr_cascaded_1.5m_vhr.tif`.
