# [ARCHIVED / DEPRECATED] Folder 07: Cascaded 1.5m VHR Refinement Experiments

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
