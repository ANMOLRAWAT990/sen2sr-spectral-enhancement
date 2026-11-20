# [ARCHIVED / DEPRECATED] Stage 2: Cascaded 1.25m Refiner Experiments

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
