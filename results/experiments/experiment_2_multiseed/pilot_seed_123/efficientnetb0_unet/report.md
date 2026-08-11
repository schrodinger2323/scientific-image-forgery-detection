# efficientnetb0_unet Report

## Environment info

See `environment_info.json` for exact package versions and GPU visibility.

## Dataset summary

The experiment uses a deterministic stratified subset with up to 300 samples per class. The shared split is 60/20/20 train/val-tune/internal-test and is grouped by `image_id` to prevent leakage across splits.

| split         | class_name   |   count |
|:--------------|:-------------|--------:|
| internal_test | authentic    |      65 |
| internal_test | forged       |      57 |
| train         | authentic    |     178 |
| train         | forged       |     183 |
| val_tune      | authentic    |      57 |
| val_tune      | forged       |      60 |

## Preprocessing

- Images are resized to 256x256.
- Authentic masks are generated as all-zero binary masks.
- Forged `.npy` masks are converted to binary; multi-channel masks are reduced with `np.any(mask > 0, axis=0)`.
- Augmentation is applied only to the training split: horizontal flip, vertical flip, 90-degree rotation, brightness, and contrast jitter.
- Model-specific image preprocessing is selected automatically for `efficientnetb0_unet`.

## Model architecture

The architecture is implemented for the `TensorFlow/Keras` training path and saved as `model_summary.txt`. This run uses `efficientnetb0_unet` with a one-channel segmentation head.

## Training setup

- Seed: 123
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

- Val pixel F1: 0.4387
- Val image F1: 0.5347

## Internal-test results

| metric                   |     mean |         std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|------------:|-----------:|------------:|----:|
| pixel_precision          | 0.360462 |   0.440959  |   0.282214 |    0.43871  | 122 |
| pixel_recall             | 0.642325 |   0.444929  |   0.563373 |    0.721278 | 122 |
| pixel_f1                 | 0.352713 |   0.437446  |   0.275088 |    0.430337 | 122 |
| pixel_dice               | 0.352713 |   0.437446  |   0.275088 |    0.430337 | 122 |
| pixel_iou                | 0.32671  |   0.4312    |   0.250193 |    0.403226 | 122 |
| pixel_specificity        | 0.962264 |   0.0543771 |   0.952615 |    0.971913 | 122 |
| image_accuracy           | 0.5      |   0.0461778 |   0.409836 |    0.590164 | 122 |
| image_precision          | 0.467742 |   0.0647678 |   0.343738 |    0.6      | 122 |
| image_recall_sensitivity | 0.508772 |   0.068673  |   0.374947 |    0.644068 | 122 |
| image_specificity        | 0.492308 |   0.0610594 |   0.375    |    0.621647 | 122 |
| image_f1                 | 0.487395 |   0.0574606 |   0.368421 |    0.596818 | 122 |
| image_roc_auc            | 0.497571 |   0.0510082 |   0.397359 |    0.595004 | 122 |
| threshold                | 0.8      | nan         | nan        |  nan        | 122 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|              0.5 |          0.467742 |                   0.508772 |            0.492308 |   0.487395 |        0.497571 |         0.8 |

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
