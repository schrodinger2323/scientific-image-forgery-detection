# segformer_b0 Report

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
- Model-specific image preprocessing is selected automatically for `segformer_b0`.

## Model architecture

The architecture is implemented for the `PyTorch` training path and saved as `model_summary.txt`. This run uses `segformer_b0` with a one-channel segmentation head.

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

Threshold sweep was performed only on the val-tune split from 0.10 to 0.90 with step 0.05. The selected threshold is **0.90**, chosen by best val pixel-level F1 with image-level F1 as tie-break.

- Val pixel F1: 0.4091
- Val image F1: 0.7059

## Internal-test results

| metric                   |     mean |         std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|------------:|-----------:|------------:|----:|
| pixel_precision          | 0.483966 |   0.464874  |   0.401474 |    0.566458 | 122 |
| pixel_recall             | 0.651694 |   0.403451  |   0.580102 |    0.723287 | 122 |
| pixel_f1                 | 0.36549  |   0.401339  |   0.294273 |    0.436708 | 122 |
| pixel_dice               | 0.36549  |   0.401339  |   0.294273 |    0.436708 | 122 |
| pixel_iou                | 0.317836 |   0.393097  |   0.248081 |    0.387591 | 122 |
| pixel_specificity        | 0.988539 |   0.0234097 |   0.984385 |    0.992693 | 122 |
| image_accuracy           | 0.598361 |   0.0436004 |   0.516393 |    0.688525 | 122 |
| image_precision          | 0.546512 |   0.0542477 |   0.442917 |    0.651176 | 122 |
| image_recall_sensitivity | 0.824561 |   0.0509091 |   0.714286 |    0.915254 | 122 |
| image_specificity        | 0.4      |   0.0620908 |   0.279412 |    0.516129 | 122 |
| image_f1                 | 0.657343 |   0.0468092 |   0.556347 |    0.743252 | 122 |
| image_roc_auc            | 0.721997 |   0.0463793 |   0.628555 |    0.808227 | 122 |
| threshold                | 0.9      | nan         | nan        |  nan        | 122 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.598361 |          0.546512 |                   0.824561 |                 0.4 |   0.657343 |        0.721997 |         0.9 |

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
