# deeplabv3plus Report

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
- Model-specific image preprocessing is selected automatically for `deeplabv3plus`.

## Model architecture

The architecture is implemented for the `PyTorch` training path and saved as `model_summary.txt`. This run uses `deeplabv3plus` with a one-channel segmentation head.

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

- Val pixel F1: 0.4697
- Val image F1: 0.1111

## Internal-test results

| metric                   |     mean |          std |    ci95_low |   ci95_high |   n |
|:-------------------------|---------:|-------------:|------------:|------------:|----:|
| pixel_precision          | 0.551593 |   0.496517   |   0.462383  |    0.640804 | 119 |
| pixel_recall             | 0.558664 |   0.495119   |   0.469704  |    0.647623 | 119 |
| pixel_f1                 | 0.519501 |   0.496329   |   0.430324  |    0.608678 | 119 |
| pixel_dice               | 0.519501 |   0.496329   |   0.430324  |    0.608678 | 119 |
| pixel_iou                | 0.516502 |   0.498438   |   0.426947  |    0.606058 | 119 |
| pixel_specificity        | 0.999636 |   0.00220587 |   0.999239  |    1.00003  | 119 |
| image_accuracy           | 0.571429 |   0.0463491  |   0.478992  |    0.663866 | 119 |
| image_precision          | 0.583333 |   0.145594   |   0.285714  |    0.866667 | 119 |
| image_recall_sensitivity | 0.132075 |   0.0450049  |   0.0545211 |    0.233351 | 119 |
| image_specificity        | 0.924242 |   0.0332959  |   0.851351  |    0.98334  | 119 |
| image_f1                 | 0.215385 |   0.0652754  |   0.09375   |    0.352941 | 119 |
| image_roc_auc            | 0.588336 |   0.0528702  |   0.48283   |    0.695691 | 119 |
| threshold                | 0.9      | nan          | nan         |  nan        | 119 |

Image-level metrics at the selected threshold:

|   image_accuracy |   image_precision |   image_recall_sensitivity |   image_specificity |   image_f1 |   image_roc_auc |   threshold |
|-----------------:|------------------:|---------------------------:|--------------------:|-----------:|----------------:|------------:|
|         0.571429 |          0.583333 |                   0.132075 |            0.924242 |   0.215385 |        0.588336 |         0.9 |

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
