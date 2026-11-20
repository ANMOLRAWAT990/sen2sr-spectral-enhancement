# Changelog: Audit-Driven Overhaul of SEN2SR Repository

This document lists all modifications, bug fixes, architectural enhancements, and documentation revisions made to the repository.

---

## [2.0.0] - Audit-Driven Overhaul (Tasks T1 through T7)

### Task T1: Band Order & Physical Normalization Contract
- **Root Cause & Evidence:**
  - Official Hugging Face metadata (`tacofoundation/SEN2SRLite` `mlm.json`) explicitly defines input bands as `["B04", "B03", "B02", "B08"]` = `[Red, Green, Blue, NIR]`.
  - SEN2NAIP raw files (`*_lr.tif` and `*_hr.tif`) have 4 channels: ch0=Red, ch1=Green, ch2=Blue, ch3=NIR.
  - Previous code assumed `[B, G, R, N]` and loaded `img[:4]`, causing Red and Blue channels to be swapped. In display, `img[[2, 1, 0]]` swapped them again, turning vegetation blue and collapsing NDVI to ~0.15.
  - WorldStrat 12-band L2A stack has `B04` at index 3, `B03` at index 2, `B02` at index 1, and `B08` at index 7.
  - Previous normalization guessed divisors dynamically based on image max (`max>255 -> /10000; max>2 -> /255`), which produced erratic training loss and ignored sensor exposure differences between aerial cameras and satellite reflectance.
- **Modifications:**
  - Established canonical constants across all notebooks:
    ```python
    SEN2NAIP_LR_ORDER = [0, 1, 2, 3]  # [Red, Green, Blue, NIR]
    SEN2NAIP_HR_ORDER = [0, 1, 2, 3]  # [Red, Green, Blue, NIR]
    WS_LR_ORDER       = [3, 2, 1, 7]  # Extracts [B04, B03, B02, B08] from 12-band L2A
    WS_HR_ORDER       = [0, 1, 2, 3]  # [Red, Green, Blue, NIR]
    ```
  - True Color RGB is now simply `img[[0, 1, 2]]` without channel reversal.
  - Fixed physical scaling factors: Sentinel-2 L2A `/10000.0`, NAIP `/255.0`, WorldStrat `/4095.0`.
  - Added per-patch radiometric quantile matching (`match_radiometry_patch`) to align target HR radiometry to LR satellite radiometry, forcing the network to optimize structural edge recovery rather than sensor gain offsets.
  - Added strict runtime assertions: shapes `(4, H, W)`, values in `[0.0, 1.0]`, zero NaN values.

---

### Task T2: Geographical Scene-Stratified Split & Dynamic Patch Augmentation
- **Root Cause & Evidence:**
  - Previous training cached a single random crop per scene in memory (`self.cache[idx]`), locking the network into training on 429 static 64×64 patches for all epochs with zero spatial augmentation.
  - Random patch-level splits (`random_split`) over patches from the same scene caused severe data leakage between training and validation.
  - Hardcoded sample caps (`max_samples=400` in Lite, `max_samples=60` in Full) artificially constrained dataset size, while patch size 32×32 in Full was too small for Mamba receptive fields.
- **Modifications:**
  - Replaced random splitting with geographical scene-level split: 80% Train, 10% Val, 10% Test, grouped strictly by geographical scene ID.
  - Saved held-out test scenes to `Results/test_scenes.json` so Lite and Full models are evaluated on the exact same test scenes.
  - Implemented dynamic per-epoch random cropping in `ScenePatchDataset.__getitem__`, drawing new random crop coordinates on every sample access.
  - Added synchronized data augmentations (horizontal flips, vertical flips, random 90-degree rotations) applied identically to LR and HR pairs.
  - Removed sample caps (`max_samples=None` by default), increased Full model LR patch size to 64×64 (HR 256×256).

---

### Task T3: Multi-Objective Shift-Tolerant Loss & Best Checkpointing
- **Root Cause & Evidence:**
  - Satellite-to-aerial pixel misregistration (typically 1–2 pixels) caused standard L1 pixel loss to heavily penalize sharp reconstructed edges that were sub-pixel shifted relative to NAIP.
  - Band indices in compound loss were hardcoded numbers.
  - Only the final epoch checkpoint was saved; validation loss plateaued after epoch ~5, leading to sub-optimal saved weights.
- **Modifications:**
  - Implemented `ShiftTolerantL1Loss(max_shift=2)` which minimizes L1 reconstruction error across $\pm 2$ pixel spatial shifts.
  - Refactored `MultiIndexSpectralLoss` to use canonical constants (`BAND_RED=0`, `BAND_GREEN=1`, `BAND_BLUE=2`, `BAND_NIR=3`) to compute NDVI and NDWI.
  - Implemented best-validation checkpointing (`sen2sr_*_compound_best.pth`) alongside the final checkpoint (`sen2sr_*_compound_last.pth`), with full epoch-by-epoch JSON training history tracking.

---

### Task T4: Dual-Section Comparison with Ground-Truth Side-by-Side & Shared Contrast
- **Root Cause & Evidence:**
  - Previous comparison notebooks evaluated outputs exclusively on full Copernicus scenes without ground truth, and relied on Tenengrad edge gradient metrics ("acuity") that were inflated by contrast stretching.
- **Modifications:**
  - Restructured `SEN2SR_Baseline_vs_Finetuned_Comparison.ipynb` and `SEN2SR_Full_Baseline_vs_Finetuned_Comparison.ipynb` into two distinct sections:
    - **Section 1: Ground-Truth Benchmark on Held-Out Test Scenes:**
      - Evaluates $N=6$ held-out test patches from `test_scenes.json` in a 4-way visual layout: `[10m Bicubic 4×] | [Pretrained Baseline] | [Fine-Tuned (Ours)] | [Real NAIP Ground Truth]`.
      - All 4 panels share the exact same percentile stretch (2nd to 98th percentile) computed strictly from the real NAIP ground truth RGB.
      - 3× zoom-in panels inspecting fine features (roads, field edges, built-up clusters).
      - Absolute error difference maps (`|Fine-Tuned - NAIP|` vs `|Baseline - NAIP|`).
      - Ground-truth sanity table computing PSNR, SSIM, and NDVI-MAE.
    - **Section 2: Real-World Copernicus Scene Inference:**
      - Retains 2D Hann window blending inference across user-supplied Copernicus scenes.
      - Displays prominent caveat: "Visual inspection only — No aerial ground truth exists for this scene."
      - Relabeled all edge gradient metrics to "Relative Sharpness (Not Accuracy)".

---

### Task T5: Deprecation of Sub-2m Cascaded Refinement (Folders 05 & 07)
- **Root Cause & Evidence:**
  - In Folders 05 and 07, the "1.25m / 1.5m ground truth" was created by taking 2.5m NAIP aerial imagery and interpolating it 2× via bilinear or bicubic interpolation (`hr_8x = F.interpolate(..., 'bicubic')`).
  - No real sub-2.5m sensor data was present in the target; consequently, models learned smoothed or hallucinatory approximations.
  - Claims of "+87% sharpness" and "zero-brightening guarantee" were contradicted by visible contrast blowouts.
- **Modifications:**
  - Added warning deprecation banners to all notebooks in `05_ultra_highres_1m_experiments` and `07_cascaded_1.5m_vhr_experiments`.
  - Updated READMEs in 05 and 07 to document the physical resolution limits of 10m Sentinel-2 pixels and archive the experiments.

---

### Task T6: PatchGAN Adversarial Fine-Tuning (`04b_Lite_Adversarial_Finetune.ipynb`)
- **Root Cause & Evidence:**
  - Pure $L_1$ and compound regression losses suffer from regression-to-the-mean, creating slightly soft edges in ambiguous regions.
- **Modifications:**
  - Created standalone notebook `04b_Lite_Adversarial_Finetune.ipynb`.
  - Starts from best compound fine-tuned checkpoint (`sen2sr_lite_compound_best.pth`).
  - Implemented 70×70 PatchGAN Discriminator stabilized with Spectral Normalization (`spectral_norm`).
  - Added VGG-19 perceptual feature loss on RGB channels alongside shift-tolerant L1, Sobel edge, and NDVI/NDWI spectral index loss.
  - Conservative generator learning rate ($2\times 10^{-5}$) over 10 epochs.
  - Includes 5-way comparative evaluation against NAIP with explicit hallucination warning.

---

### Task T7: Documentation & Captions Overhaul
- **Modifications:**
  - Rewrote root `README.md` to define scientific scope (10m $\rightarrow$ 2.5m), list architectural fixes, provide Colab execution instructions, and document model tiers.
  - Created `results/README.md` documenting output files, benchmark metrics, and explaining legacy figure findings.
  - Created this comprehensive `CHANGELOG.md`.
