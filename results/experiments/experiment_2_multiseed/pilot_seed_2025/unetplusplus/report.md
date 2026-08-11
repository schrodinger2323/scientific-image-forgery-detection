# unetplusplus Report

## Environment info

See `environment_info.json` for exact package versions and GPU visibility.

## Dataset summary

The experiment uses a deterministic stratified subset with up to 300 samples per class. The shared split is 60/20/20 train/val-tune/internal-test and is grouped by `image_id` to prevent leakage across splits.

| split         | class_name   |   count |
|:--------------|:-------------|--------:|
| internal_test | authentic    |      66 |
| internal_test | forged       |      53 |
| train         | authentic    |     171 |
| train         | forged       |     182 |
| val_tune      | authentic    |      63 |
| val_tune      | forged       |      65 |

## Preprocessing

- Images are resized to 256x256.
- Authentic masks are generated as all-zero binary masks.
- Forged `.npy` masks are converted to binary; multi-channel masks are reduced with `np.any(mask > 0, axis=0)`.
- Augmentation is applied only to the training split: horizontal flip, vertical flip, 90-degree rotation, brightness, and contrast jitter.
- Model-specific image preprocessing is selected automatically for `unetplusplus`.

## Model architecture

The architecture is implemented for the `TensorFlow/Keras` training path and saved as `model_summary.txt`. This run uses `unetplusplus` with a one-channel segmentation head.

## Training setup

- Seed: 2025
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

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **0.80**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: 0.4060
- Val image F1: 0.1882

## Internal-test results

| metric                   |     mean |         std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|------------:|-----------:|------------:|----:|
| pixel_precision          | 0.509415 |   0.488655  |   0.421617 |    0.597213 | 119 |
| pixel_recall             | 0.569754 |   0.486617  |   0.482322 |    0.657186 | 119 |
| pixel_f1                 | 0.503282 |   0.490764  |   0.415105 |    0.591459 | 119 |
| pixel_dice               | 0.503282 |   0.490764  |   0.415105 |    0.591459 | 119 |
| pixel_iou                | 0.496766 |   0.494372  |   0.407941 |    0.585592 | 119 |
| pixel_specificity        | 0.994354 |   0.0175704 |   0.991197 |    0.997511 | 119 |
| image_accuracy           | 0.579832 |   0.0455228 |   0.495588 |    0.655672 | 119 |
| image_precision          | 0.578947 |   0.118799  |   0.349946 |    0.818316 | 119 |
| image_recall_sensitivity | 0.207547 |   0.0538624 |   0.106383 |    0.313799 | 119 |
| image_specificity        | 0.878788 |   0.0421445 |   0.793103 |    0.953846 | 119 |
| image_f1                 | 0.305556 |   0.0683328 |   0.169006 |    0.430811 | 119 |
| image_roc_auc            | 0.588622 |   0.055229  |   0.480466 |    0.688882 | 119 |
| threshold                | 0.8      | nan         | nan        |  nan        | 119 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.579832 |          0.578947 |                   0.207547 |            0.878788 |   0.305556 |        0.588622 |         0.8 |

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
