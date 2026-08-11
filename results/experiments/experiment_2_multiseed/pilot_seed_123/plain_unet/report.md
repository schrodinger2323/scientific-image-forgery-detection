# plain_unet Report

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
- Model-specific image preprocessing is selected automatically for `plain_unet`.

## Model architecture

The architecture is implemented for the `TensorFlow/Keras` training path and saved as `model_summary.txt`. This run uses `plain_unet` with a one-channel segmentation head.

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

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **0.70**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: 0.4872
- Val image F1: 0.0000

## Internal-test results

| metric                   |      mean |           std |   ci95_low |   ci95_high |   n |
|:-------------------------|----------:|--------------:|-----------:|------------:|----:|
| pixel_precision          | 0.508197  |   0.501994    |   0.419118 |    0.597276 | 122 |
| pixel_recall             | 0.532787  |   0.500981    |   0.443888 |    0.621686 | 122 |
| pixel_f1                 | 0.508197  |   0.501994    |   0.419118 |    0.597276 | 122 |
| pixel_dice               | 0.508197  |   0.501994    |   0.419118 |    0.597276 | 122 |
| pixel_iou                | 0.508197  |   0.501994    |   0.419118 |    0.597276 | 122 |
| pixel_specificity        | 0.999998  |   1.08482e-05 |   0.999996 |    1        | 122 |
| image_accuracy           | 0.532787  |   0.0459604   |   0.442623 |    0.622951 | 122 |
| image_precision          | 0.5       |   0.228428    |   0        |    1        | 122 |
| image_recall_sensitivity | 0.0526316 |   0.0291886   |   0        |    0.117672 | 122 |
| image_specificity        | 0.953846  |   0.0261241   |   0.894717 |    1        | 122 |
| image_f1                 | 0.0952381 |   0.0497254   |   0        |    0.200072 | 122 |
| image_roc_auc            | 0.536977  |   0.054528    |   0.432926 |    0.63726  | 122 |
| threshold                | 0.7       | nan           | nan        |  nan        | 122 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.532787 |               0.5 |                  0.0526316 |            0.953846 |  0.0952381 |        0.536977 |         0.7 |

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
