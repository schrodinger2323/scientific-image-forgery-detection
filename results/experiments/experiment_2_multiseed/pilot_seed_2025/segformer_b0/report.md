# segformer_b0 Report

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
- Model-specific image preprocessing is selected automatically for `segformer_b0`.

## Model architecture

The architecture is implemented for the `PyTorch` training path and saved as `model_summary.txt`. This run uses `segformer_b0` with a one-channel segmentation head.

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

- Val pixel F1: 0.5194
- Val image F1: 0.7009

## Internal-test results

| metric                   |     mean |         std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|------------:|-----------:|------------:|----:|
| pixel_precision          | 0.694578 |   0.440476  |   0.615436 |    0.773719 | 119 |
| pixel_recall             | 0.621407 |   0.442258  |   0.541945 |    0.700868 | 119 |
| pixel_f1                 | 0.555338 |   0.446918  |   0.475039 |    0.635637 | 119 |
| pixel_dice               | 0.555338 |   0.446918  |   0.475039 |    0.635637 | 119 |
| pixel_iou                | 0.524038 |   0.458943  |   0.441579 |    0.606498 | 119 |
| pixel_specificity        | 0.997852 |   0.0063801 |   0.996706 |    0.998998 | 119 |
| image_accuracy           | 0.764706 |   0.040546  |   0.680672 |    0.840336 | 119 |
| image_precision          | 0.765957 |   0.0630584 |   0.630435 |    0.884659 | 119 |
| image_recall_sensitivity | 0.679245 |   0.0648705 |   0.553155 |    0.803658 | 119 |
| image_specificity        | 0.833333 |   0.0456452 |   0.736091 |    0.916667 | 119 |
| image_f1                 | 0.72     |   0.0531361 |   0.608668 |    0.813559 | 119 |
| image_roc_auc            | 0.77673  |   0.0480771 |   0.677959 |    0.862792 | 119 |
| threshold                | 0.9      | nan         | nan        |  nan        | 119 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.764706 |          0.765957 |                   0.679245 |            0.833333 |       0.72 |         0.77673 |         0.9 |

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
