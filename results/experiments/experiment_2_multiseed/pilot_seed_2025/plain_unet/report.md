# plain_unet Report

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
- Model-specific image preprocessing is selected automatically for `plain_unet`.

## Model architecture

The architecture is implemented for the `TensorFlow/Keras` training path and saved as `model_summary.txt`. This run uses `plain_unet` with a one-channel segmentation head.

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

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **0.90**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: 0.2255
- Val image F1: 0.3840

## Internal-test results

| metric                   |     mean |         std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|------------:|-----------:|------------:|----:|
| pixel_precision          | 0.398734 |   0.471957  |   0.313936 |    0.483532 | 119 |
| pixel_recall             | 0.622858 |   0.45487   |   0.54113  |    0.704586 | 119 |
| pixel_f1                 | 0.398471 |   0.469571  |   0.314102 |    0.48284  | 119 |
| pixel_dice               | 0.398471 |   0.469571  |   0.314102 |    0.48284  | 119 |
| pixel_iou                | 0.386371 |   0.47437   |   0.30114  |    0.471603 | 119 |
| pixel_specificity        | 0.948692 |   0.111174  |   0.928717 |    0.968667 | 119 |
| image_accuracy           | 0.563025 |   0.0449183 |   0.470588 |    0.655462 | 119 |
| image_precision          | 0.511111 |   0.0753716 |   0.355521 |    0.659574 | 119 |
| image_recall_sensitivity | 0.433962 |   0.0693815 |   0.304328 |    0.569027 | 119 |
| image_specificity        | 0.666667 |   0.0568207 |   0.552239 |    0.769318 | 119 |
| image_f1                 | 0.469388 |   0.0633912 |   0.337079 |    0.584092 | 119 |
| image_roc_auc            | 0.570183 |   0.0539169 |   0.464644 |    0.669324 | 119 |
| threshold                | 0.9      | nan         | nan        |  nan        | 119 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.563025 |          0.511111 |                   0.433962 |            0.666667 |   0.469388 |        0.570183 |         0.9 |

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
