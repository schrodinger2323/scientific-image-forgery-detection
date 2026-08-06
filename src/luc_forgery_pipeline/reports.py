from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import ExperimentConfig


def write_model_report(
    model_dir: Path,
    model_name: str,
    config: ExperimentConfig,
    dataset_summary: pd.DataFrame,
    threshold_df: pd.DataFrame,
    test_summary: pd.DataFrame,
    image_metrics: Dict[str, float],
    framework: str = "TensorFlow/Keras",
) -> None:
    selected = threshold_df[threshold_df["selected"]].iloc[0].to_dict()
    figures = [
        "threshold_analysis.png",
        "loss_curve.png",
        "dice_curve.png",
        "iou_curve.png",
        "precision_recall_curve.png",
        "confusion_matrix.png",
        "roc_auc_curve.png",
        "prediction_examples/",
    ]
    report = f"""# {model_name} Report

## Environment info

See `environment_info.json` for exact package versions and GPU visibility.

## Dataset summary

The experiment uses a deterministic stratified subset with up to {config.samples_per_class} samples per class. The shared split is 60/20/20 train/val-tune/internal-test and is grouped by `image_id` to prevent leakage across splits.

{dataset_summary.to_markdown(index=False)}

## Preprocessing

- Images are resized to {config.img_size}x{config.img_size}.
- Authentic masks are generated as all-zero binary masks.
- Forged `.npy` masks are converted to binary; multi-channel masks are reduced with `np.any(mask > 0, axis=0)`.
- Augmentation is applied only to the training split: horizontal flip, vertical flip, 90-degree rotation, brightness, and contrast jitter.
- Model-specific image preprocessing is selected automatically for `{model_name}`.

## Model architecture

The architecture is implemented for the `{framework}` training path and saved as `model_summary.txt`. This run uses `{model_name}` with a one-channel segmentation head.

## Training setup

- Seed: {config.seed}
- Batch size: {config.batch_size}
- Max epochs: {config.epochs}
- Learning rate: {config.learning_rate}
- Optimizer: {config.optimizer}
- Loss: `0.5 * BCE + 0.5 * Dice Loss`
- Early stopping patience: {config.early_stopping_patience}
- ReduceLROnPlateau: patience {config.reduce_lr_patience}, factor {config.reduce_lr_factor}
- Checkpoint monitor: validation Dice
- Resume support: `last_model` checkpoint artifacts and `training_log.csv`

## Threshold selection

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **{selected["threshold"]:.2f}**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: {selected["pixel_f1_mean"]:.4f}
- Val image F1: {selected["image_f1"]:.4f}

## Internal-test results

{test_summary.to_markdown(index=False)}

Image-level metrics at the selected threshold:

{pd.DataFrame([image_metrics]).to_markdown(index=False)}

## Figures list

{chr(10).join(f"- `{figure}`" for figure in figures)}

## Limitations

- The first run intentionally uses a small balanced subset; final claims should be repeated on the full dataset after sanity checks.
- The threshold is tuned on val-tune only and may be sensitive to class balance.
- Encoder backbones use ImageNet pretraining, which is not domain-specific for scientific forgery artifacts.
- Kaggle `test_images` have no ground truth and are used only for `submission.csv`.
"""
    (model_dir / "report.md").write_text(report, encoding="utf-8")
