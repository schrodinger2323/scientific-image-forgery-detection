# Pilot Seed Experiment Design

## Purpose

Small-subset results are repeated over multiple seeds to test whether the model ranking is stable.

## Dataset

- Dataset root: `\content\drive\MyDrive\bitirmeProjesi\dataset`
- Samples per class: `300`
- Models: `plain_unet, unetplusplus, efficientnetb0_unet, deeplabv3plus, segformer_b0`
- Seeds: `42, 123, 2025`

## Split Policy

- Split ratio: train / val-tune / internal-test = 60 / 20 / 20
- Split type: stratified and group-aware
- Group key: `image_id` / `group_id`
- Leakage rule: the same image group cannot appear in more than one split

## Subset Policy

fixed_subset_index=\content\drive\MyDrive\bitirmeProjesi\_shared_splits_seed42_subset\full_dataset_index.csv; the exact rows in this CSV are reused for all split seeds

For the fairest comparison with the first pilot experiment, use `--fixed-subset-index` with the first experiment's `_shared_splits_seed42_subset/full_dataset_index.csv`. In that mode, the 300 authentic and 300 forged images are identical to the first pilot; only the seed-dependent split/training randomness changes.

## Training Configuration

- Image size: `256`
- Batch size: `8`
- Max epochs: `40`
- Learning rate: `0.0001`
- Optimizer: `adamw`
- Early stopping patience: `8`
- ReduceLROnPlateau patience/factor: `4` / `0.3`
- Checkpoint monitor: validation Dice
- Keras loss: `0.5 * BCE + 0.5 * Dice Loss`
- PyTorch primary loss: `BCEWithLogits + Dice Loss`
- PyTorch forensic auxiliary losses: edge/aux/image losses are used only when the model exposes those heads

## Preprocessing And Augmentation

- Images are resized to `256x256`.
- Authentic masks are generated as all-zero binary masks.
- Forged `.npy` masks are read as binary masks.
- Multi-channel masks are collapsed with `np.any(mask > 0, axis=0)`.
- Augmentation is train-only: horizontal flip, vertical flip, 90-degree rotation, brightness/contrast jitter.
- Validation and internal-test data are not augmented.

## Metric Policy

Thresholds are swept from `0.10` to `0.90` with step `0.05` only on the val-tune split. The selected threshold is then applied once to the internal-test split.

The main segmentation stability metrics are forged-only pixel F1 and forged-only pixel IoU. All-image pixel means are still saved, but they can be optimistic because authentic images have all-zero ground-truth masks.

## Output Artifacts

- Per seed/model: `training_log.csv`, `threshold_analysis.csv`, `test_per_image_metrics.csv`, `test_metrics_summary.csv`, `run_summary.json`, model checkpoints, plots, and prediction examples.
- Across seeds: `pilot_seed_per_run_results.csv`, `pilot_seed_comparison.csv`, `pilot_seed_comparison_table.md`.
- After all requested runs complete: `pilot_experiment_interpretation.md`.
