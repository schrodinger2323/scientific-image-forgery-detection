# segformer_b0 Report

## Environment info

See `environment_info.json` for exact package versions and GPU visibility.

## Dataset summary

The experiment uses a deterministic stratified subset with up to 300 samples per class. The shared split is 60/20/20 train/val-tune/internal-test and is grouped by `image_id` to prevent leakage across splits.

| split         | class_name   |   count |
|:--------------|:-------------|--------:|
| internal_test | authentic    |      60 |
| internal_test | forged       |      60 |
| train         | authentic    |     178 |
| train         | forged       |     179 |
| val_tune      | authentic    |      62 |
| val_tune      | forged       |      61 |

## Preprocessing

- Images are resized to 256x256.
- Authentic masks are generated as all-zero binary masks.
- Forged `.npy` masks are converted to binary; multi-channel masks are reduced with `np.any(mask > 0, axis=0)`.
- Augmentation is applied only to the training split: horizontal flip, vertical flip, 90-degree rotation, brightness, and contrast jitter.
- Model-specific image preprocessing is selected automatically for `segformer_b0`.

## Model architecture

The architecture is implemented for the `PyTorch` training path and saved as `model_summary.txt`. This run uses `segformer_b0` with a one-channel segmentation head.

## Training setup

- Seed: 42
- Batch size: 8
- Max epochs: 40
- Learning rate: 0.0001
- Optimizer: adamw
- Loss: `0.5 * BCE + 0.5 * Dice Loss`
- Early stopping patience: 8
- ReduceLROnPlateau: patience 4, factor 0.3
- Checkpoint monitor: validation Dice
- Resume support: `last_model` checkpoint artifacts and `training_log.csv`

## Threshold selection

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **0.85**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: 0.4550
- Val image F1: 0.6325

## Internal-test results

| metric                   |     mean |         std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|------------:|-----------:|------------:|----:|
| pixel_precision          | 0.577302 |   0.472695  |   0.492726 |    0.661877 | 120 |
| pixel_recall             | 0.601303 |   0.430548  |   0.524268 |    0.678337 | 120 |
| pixel_f1                 | 0.448889 |   0.432192  |   0.37156  |    0.526218 | 120 |
| pixel_dice               | 0.448889 |   0.432192  |   0.37156  |    0.526218 | 120 |
| pixel_iou                | 0.407373 |   0.4324    |   0.330007 |    0.484739 | 120 |
| pixel_specificity        | 0.991698 |   0.0181315 |   0.988454 |    0.994942 | 120 |
| image_accuracy           | 0.65     |   0.0431711 |   0.566667 |    0.733333 | 120 |
| image_precision          | 0.645161 |   0.0608631 |   0.522963 |    0.758685 | 120 |
| image_recall_sensitivity | 0.666667 |   0.0599736 |   0.545415 |    0.777819 | 120 |
| image_specificity        | 0.633333 |   0.0628732 |   0.516105 |    0.745775 | 120 |
| image_f1                 | 0.655738 |   0.0495072 |   0.542029 |    0.746269 | 120 |
| image_roc_auc            | 0.713889 |   0.0470554 |   0.614844 |    0.804197 | 120 |
| threshold                | 0.85     | nan         | nan        |  nan        | 120 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|             0.65 |          0.645161 |                   0.666667 |            0.633333 |   0.655738 |        0.713889 |        0.85 |

## Figures list

- `threshold_analysis.png`
- `loss_curve.png`
- `dice_curve.png`
- `iou_curve.png`
- `precision_recall_curve.png`
- `confusion_matrix.png`
- `roc_auc_curve.png`
- `prediction_examples/`

## Limitations

- The first run intentionally uses a small balanced subset; final claims should be repeated on the full dataset after sanity checks.
- The threshold is tuned on val-tune only and may be sensitive to class balance.
- Encoder backbones use ImageNet pretraining, which is not domain-specific for scientific forgery artifacts.
- Kaggle `test_images` have no ground truth and are used only for `submission.csv`.
