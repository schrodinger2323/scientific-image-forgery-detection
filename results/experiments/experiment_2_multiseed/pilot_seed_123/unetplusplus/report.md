# unetplusplus Report

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
- Model-specific image preprocessing is selected automatically for `unetplusplus`.

## Model architecture

The architecture is implemented for the `TensorFlow/Keras` training path and saved as `model_summary.txt`. This run uses `unetplusplus` with a one-channel segmentation head.

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

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **0.85**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: 0.4393
- Val image F1: 0.4000

## Internal-test results

| metric                   |     mean |          std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|-------------:|-----------:|------------:|----:|
| pixel_precision          | 0.432339 |   0.476787   |   0.347733 |    0.516944 | 122 |
| pixel_recall             | 0.563729 |   0.478158   |   0.47888  |    0.648578 | 122 |
| pixel_f1                 | 0.385693 |   0.463666   |   0.303415 |    0.467971 | 122 |
| pixel_dice               | 0.385693 |   0.463666   |   0.303415 |    0.467971 | 122 |
| pixel_iou                | 0.370941 |   0.465358   |   0.288363 |    0.453519 | 122 |
| pixel_specificity        | 0.995449 |   0.00986221 |   0.993699 |    0.997199 | 122 |
| image_accuracy           | 0.491803 |   0.0457456  |   0.409631 |    0.581967 | 122 |
| image_precision          | 0.439024 |   0.0769369  |   0.285714 |    0.591885 | 122 |
| image_recall_sensitivity | 0.315789 |   0.063411   |   0.189631 |    0.440719 | 122 |
| image_specificity        | 0.646154 |   0.0590095  |   0.522981 |    0.762811 | 122 |
| image_f1                 | 0.367347 |   0.0627156  |   0.235256 |    0.480804 | 122 |
| image_roc_auc            | 0.484345 |   0.0525574  |   0.379464 |    0.586251 | 122 |
| threshold                | 0.85     | nan          | nan        |  nan        | 122 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.491803 |          0.439024 |                   0.315789 |            0.646154 |   0.367347 |        0.484345 |        0.85 |

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
