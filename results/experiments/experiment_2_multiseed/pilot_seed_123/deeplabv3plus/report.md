# deeplabv3plus Report

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
- Model-specific image preprocessing is selected automatically for `deeplabv3plus`.

## Model architecture

The architecture is implemented for the `PyTorch` training path and saved as `model_summary.txt`. This run uses `deeplabv3plus` with a one-channel segmentation head.

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

- Val pixel F1: 0.4286
- Val image F1: 0.2105

## Internal-test results

| metric                   |     mean |          std |   ci95_low |   ci95_high |   n |
|:-------------------------|---------:|-------------:|-----------:|------------:|----:|
| pixel_precision          | 0.509175 |   0.497183   |   0.42095  |    0.5974   | 122 |
| pixel_recall             | 0.543356 |   0.491535   |   0.456133 |    0.630579 | 122 |
| pixel_f1                 | 0.443724 |   0.486234   |   0.357442 |    0.530006 | 122 |
| pixel_dice               | 0.443724 |   0.486234   |   0.357442 |    0.530006 | 122 |
| pixel_iou                | 0.436578 |   0.489492   |   0.349718 |    0.523438 | 122 |
| pixel_specificity        | 0.998957 |   0.00379701 |   0.998283 |    0.999631 | 122 |
| image_accuracy           | 0.565574 |   0.0451717  |   0.47541  |    0.655738 | 122 |
| image_precision          | 0.566667 |   0.0901802  |   0.384559 |    0.740741 | 122 |
| image_recall_sensitivity | 0.298246 |   0.0606828  |   0.177396 |    0.42     | 122 |
| image_specificity        | 0.8      |   0.0495373  |   0.694915 |    0.894756 | 122 |
| image_f1                 | 0.390805 |   0.0659225  |   0.256395 |    0.50604  | 122 |
| image_roc_auc            | 0.540351 |   0.0530332  |   0.426592 |    0.639284 | 122 |
| threshold                | 0.9      | nan          | nan        |  nan        | 122 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.565574 |          0.566667 |                   0.298246 |                 0.8 |   0.390805 |        0.540351 |         0.9 |

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
