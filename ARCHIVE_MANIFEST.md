# Thesis Project Archive Manifest

Audit date: 2026-08-11

This file records the material retained before the large local working directory is removed. The audit compared the original project tree with the GitHub-ready repository and then added missing reproducibility-critical artifacts.

The source-code hash audit found 97 local Python/notebook files. Ninety now have exact archived copies. The remaining seven files represent five superseded historical variants: an output-heavy scratch notebook, an earlier final-analysis script/notebook pair, and older copies of `torch_models.py` and `run_pilot_seed_experiments.py`. Their maintained replacements are retained in the repository.

## Retained in GitHub

- Reusable pipeline code, command-line scripts, standalone Kaggle/Colab experiment programs, and output-free notebooks.
- Previously unarchived reporting and visualization generators under `scripts/reporting/`.
- The latest final-analysis program and its output-free notebook, including cached-robustness reuse and metadata-alignment fixes.
- Detailed CSV, JSON, Markdown, and text results for experiments 2–6 under `results/experiments/` (606 files, about 29.66 MiB).
- Final per-image, robustness, statistical, calibration, and failure-case tables under `results/final/detailed/` (121 files, about 7.06 MiB).
- The exact seed-42 full-data `train`, `val`, and `test` membership plus the full index under `results/splits/full_seed42/` (4 files, about 2.43 MiB).
- Final reports, figures, source tables, workflows, and EDA summaries already present in the repository.
- Two selected final checkpoints, their model configs, and environment records under `artifacts/checkpoints/`.

## Final checkpoint checksums

| Model | File size | SHA-256 |
|---|---:|---|
| EfficientNetB0-UNet 384 small-mask | 25,336,309 bytes | `b565f1f4a784bf38e84803b369bfd8f7205ec091ab74d64d24e0df0ec4ad319a` |
| SegFormer-B0 384 small-mask | 14,947,873 bytes | `b36befa06b9ba7d2e66ed4903b865036aa7c77c578198e07fc08958963bf41ec` |

Verify them from the repository root with PowerShell:

```powershell
Get-ChildItem artifacts/checkpoints -Recurse -Filter best_model.pth |
  Get-FileHash -Algorithm SHA256
```

## Intentionally not retained

The original working tree contained 3,623 files and approximately 29.66 GiB, excluding this repository. The following are deliberately omitted because they are reproducible, externally downloadable, duplicated, or superseded:

- Recod.ai/LUC dataset copies, TensorFlow dataset shards, and downloaded archives. The dataset is available from the Kaggle competition linked in the README.
- 72 `.npz` probability-map/prediction caches (about 11.7 GiB). Final aggregate and per-image metrics are retained, and the caches can be regenerated with the archived checkpoints.
- Superseded `last_model` files, pilot/full-study checkpoints, and duplicate checkpoint copies. Only the two models selected by the final analysis are retained.
- Repeated `experiments_full`, backup (`eski`) folders, Kaggle working-directory exports, notebook checkpoints, caches, and environment folders.
- An output-heavy early scratch notebook (`notebook9796454322.ipynb`, SHA-256 `f713054f18c8b701fb50e507ba0fc9cb77ddef77bfb53a8aad148390bbb6072c`) whose final pipeline code is superseded by the ordered notebooks and reusable package. Two output-free early pipeline notebooks are retained under `notebooks/legacy/`.
- Local thesis-office files and downloaded literature PDFs. The canonical technical report is retained as `docs/final_experimental_report.md`; office/PDF source collections are outside the code-and-results archive scope.

## Deletion criterion

Do not delete the original working tree until the archive branch is merged into GitHub `main` and a fresh clone has passed the checksum, notebook, and result-manifest checks described in the pull request.
