# efficientnetb0_unet Report

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
- Model-specific image preprocessing is selected automatically for `efficientnetb0_unet`.

## Model architecture

The architecture is implemented for the `TensorFlow/Keras` training path and saved as `model_summary.txt`. This run uses `efficientnetb0_unet` with a one-channel segmentation head.

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

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **0.90**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: 0.4465
- Val image F1: 0.5794

## Internal-test results

| metric                   |     mean |         std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|------------:|-----------:|------------:|----:|
| pixel_precision          | 0.601833 |   0.4684    |   0.518025 |    0.68564  | 120 |
| pixel_recall             | 0.565239 |   0.455351  |   0.483766 |    0.646711 | 120 |
| pixel_f1                 | 0.464144 |   0.452032  |   0.383266 |    0.545023 | 120 |
| pixel_dice               | 0.464144 |   0.452032  |   0.383266 |    0.545023 | 120 |
| pixel_iou                | 0.434417 |   0.456745  |   0.352695 |    0.51614  | 120 |
| pixel_specificity        | 0.995266 |   0.0116223 |   0.993186 |    0.997345 | 120 |
| image_accuracy           | 0.666667 |   0.0425225 |   0.583333 |    0.75     | 120 |
| image_precision          | 0.7      |   0.0647381 |   0.576907 |    0.82937  | 120 |
| image_recall_sensitivity | 0.583333 |   0.0637815 |   0.460285 |    0.700037 | 120 |
| image_specificity        | 0.75     |   0.0570664 |   0.636364 |    0.85971  | 120 |
| image_f1                 | 0.636364 |   0.0532455 |   0.529384 |    0.736842 | 120 |
| image_roc_auc            | 0.68     |   0.0489181 |   0.583704 |    0.772769 | 120 |
| threshold                | 0.9      | nan         | nan        |  nan        | 120 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.666667 |               0.7 |                   0.583333 |                0.75 |   0.636364 |            0.68 |         0.9 |

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
