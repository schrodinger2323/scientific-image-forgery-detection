# plain_unet Report

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
- Model-specific image preprocessing is selected automatically for `plain_unet`.

## Model architecture

The architecture is implemented for the `TensorFlow/Keras` training path and saved as `model_summary.txt`. This run uses `plain_unet` with a one-channel segmentation head.

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

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **0.75**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: 0.4944
- Val image F1: 0.0909

## Internal-test results

| metric                   |     mean |         std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|------------:|-----------:|------------:|----:|
| pixel_precision          | 0.496219 |   0.49877   |   0.406978 |    0.585461 | 120 |
| pixel_recall             | 0.50919  |   0.497887  |   0.420107 |    0.598273 | 120 |
| pixel_f1                 | 0.497694 |   0.498221  |   0.408551 |    0.586837 | 120 |
| pixel_dice               | 0.497694 |   0.498221  |   0.408551 |    0.586837 | 120 |
| pixel_iou                | 0.495349 |   0.49919   |   0.406033 |    0.584666 | 120 |
| pixel_specificity        | 0.998088 |   0.0119772 |   0.995945 |    1.00023  | 120 |
| image_accuracy           | 0.516667 |   0.045605  |   0.425    |    0.6      | 120 |
| image_precision          | 0.75     |   0.262009  |   0        |    1        | 120 |
| image_recall_sensitivity | 0.05     |   0.0270723 |   0        |    0.109418 | 120 |
| image_specificity        | 0.983333 |   0.0173567 |   0.942301 |    1        | 120 |
| image_f1                 | 0.09375  |   0.0478876 |   0        |    0.194444 | 120 |
| image_roc_auc            | 0.4475   |   0.0535103 |   0.341836 |    0.55498  | 120 |
| threshold                | 0.75     | nan         | nan        |  nan        | 120 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.516667 |              0.75 |                       0.05 |            0.983333 |    0.09375 |          0.4475 |        0.75 |

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
