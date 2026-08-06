from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luc_forgery_pipeline.config import ExperimentConfig, KERAS_MODEL_NAMES, PYTORCH_MODEL_NAMES


PILOT_MODELS = ("plain_unet", "unetplusplus", "efficientnetb0_unet", "deeplabv3plus", "segformer_b0")
PILOT_SEEDS = (42, 123, 2025)
REQUIRED_ARTIFACTS = (
    "run_summary.json",
    "test_metrics_summary.csv",
    "test_per_image_metrics.csv",
    "threshold_analysis.csv",
    "training_log.csv",
)
AGGREGATED_METRICS = (
    "forged_pixel_f1",
    "forged_pixel_iou",
    "forged_pixel_precision",
    "forged_pixel_recall",
    "image_f1",
    "image_roc_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat pilot subset experiments over multiple seeds.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/drive/MyDrive/bitirmeProjesi/dataset"))
    parser.add_argument("--output-base", type=Path, default=Path("/content/drive/MyDrive/bitirmeProjesi/deney_2"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PILOT_SEEDS))
    parser.add_argument("--models", nargs="+", default=list(PILOT_MODELS), choices=list(PILOT_MODELS))
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument(
        "--fixed-subset-seed",
        type=int,
        default=None,
        help="Use the same sampled 300+300 subset for all split seeds. Example: 42.",
    )
    parser.add_argument(
        "--fixed-subset-index",
        type=Path,
        default=None,
        help="CSV containing the exact subset to reuse before creating new seed-specific splits.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--aggregate-only", action="store_true", help="Skip training and only rebuild comparison files.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Move existing selected seed/model folders to a backup directory and train them from scratch.",
    )
    parser.add_argument(
        "--interpretation",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Write pilot_experiment_interpretation.md. "
            "'auto' writes it only after every requested seed/model result is complete."
        ),
    )
    return parser.parse_args()


def train_model_for_seed(model_name: str, config: ExperimentConfig) -> None:
    print(f"Dispatching training function for {model_name}...", flush=True)
    if model_name in KERAS_MODEL_NAMES:
        from luc_forgery_pipeline.train import train_one_model

        train_one_model(model_name, config)
    elif model_name in PYTORCH_MODEL_NAMES:
        from luc_forgery_pipeline.torch_train import train_one_torch_model

        train_one_torch_model(model_name, config)
    else:
        raise ValueError(f"Pilot model is not registered: {model_name}")
    print(f"Returned from training function for {model_name}.", flush=True)


def model_run_is_complete(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in REQUIRED_ARTIFACTS)


def model_missing_artifacts(model_dir: Path) -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS if not (model_dir / name).exists()]


def describe_model_state(model_dir: Path) -> str:
    missing = model_missing_artifacts(model_dir)
    if not model_dir.exists():
        return "not_started"
    if not missing:
        return "complete"
    checkpoints = []
    for name in ["last_model.pt", "best_model.pt", "last_model.keras", "best_model.keras"]:
        if (model_dir / name).exists():
            checkpoints.append(name)
    checkpoint_text = ",".join(checkpoints) if checkpoints else "no_checkpoint"
    return f"incomplete; missing={missing}; checkpoints={checkpoint_text}"


def collect_seed_model_result(seed_root: Path, seed: int, model_name: str) -> dict[str, object] | None:
    model_dir = seed_root / model_name
    per_path = model_dir / "test_per_image_metrics.csv"
    summary_path = model_dir / "test_metrics_summary.csv"
    log_path = model_dir / "training_log.csv"
    threshold_path = model_dir / "threshold_analysis.csv"
    if not per_path.exists() or not summary_path.exists():
        return None

    per = pd.read_csv(per_path)
    forged = per[per["label"] == 1]
    summary = pd.read_csv(summary_path).set_index("metric")["mean"].to_dict()
    row: dict[str, object] = {"seed": seed, "model_name": model_name}
    for metric in ["pixel_f1", "pixel_iou", "pixel_precision", "pixel_recall"]:
        row[f"forged_{metric}"] = float(forged[metric].mean())
    row["authentic_pred_positive_rate"] = float((per[per["label"] == 0]["pred_label"] == 1).mean())
    row["forged_pred_positive_rate"] = float((forged["pred_label"] == 1).mean())
    row["image_f1"] = float(summary.get("image_f1", np.nan))
    row["image_roc_auc"] = float(summary.get("image_roc_auc", np.nan))
    if log_path.exists():
        row["epochs_ran"] = int(len(pd.read_csv(log_path)))
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)
        selected = thresholds[thresholds["selected"].astype(str).str.lower().isin(["true", "1"])]
        if len(selected):
            row["best_threshold"] = float(selected.iloc[0]["threshold"])
    return row


def format_mean_std(mean_value: float, std_value: float) -> str:
    if np.isnan(mean_value):
        return ""
    if np.isnan(std_value):
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def write_markdown_table(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        "model_name",
        "forged_pixel_f1",
        "forged_pixel_iou",
        "forged_pixel_precision",
        "forged_pixel_recall",
        "image_f1",
        "image_roc_auc",
        "average_epochs_ran",
    ]
    lines = ["# Pilot Seed Comparison", "", "Values are mean +/- std over completed seeds.", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in summary.iterrows():
        values = [str(row["model_name"])]
        for metric in columns[1:-1]:
            values.append(format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"]))
        values.append(f"{row['average_epochs_ran']:.2f}")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_results(output_base: Path, seeds: list[int], models: list[str]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        seed_root = output_base / f"pilot_seed_{seed}"
        for model_name in models:
            row = collect_seed_model_result(seed_root, seed, model_name)
            if row is not None:
                rows.append(row)
    per_seed = pd.DataFrame(rows)
    output_base.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_base / "pilot_seed_per_run_results.csv", index=False)
    if per_seed.empty:
        return per_seed

    parts = []
    for model_name, group in per_seed.groupby("model_name"):
        row = {"model_name": model_name, "n_seeds_completed": int(group["seed"].nunique())}
        for metric in AGGREGATED_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        row["average_epochs_ran"] = float(group["epochs_ran"].mean()) if "epochs_ran" in group else np.nan
        parts.append(row)
    summary = pd.DataFrame(parts).sort_values("forged_pixel_f1_mean", ascending=False)
    summary.to_csv(output_base / "pilot_seed_comparison.csv", index=False)
    write_markdown_table(summary, output_base / "pilot_seed_comparison_table.md")
    return summary


def _metric_text(row: pd.Series, metric: str) -> str:
    return format_mean_std(float(row[f"{metric}_mean"]), float(row[f"{metric}_std"]))


def _coefficient_of_variation(row: pd.Series, metric: str) -> float:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    if not np.isfinite(mean) or abs(mean) < 1e-12 or not np.isfinite(std):
        return np.inf
    return abs(std / mean)


def _top_two_preserved(per_seed: pd.DataFrame, candidate_names: set[str]) -> bool:
    if per_seed.empty:
        return False
    for _, group in per_seed.groupby("seed"):
        ranked = group.sort_values("forged_pixel_f1", ascending=False)["model_name"].head(2)
        if set(ranked) != candidate_names:
            return False
    return True


def should_write_interpretation(mode: str, summary: pd.DataFrame, seeds: list[int], models: list[str]) -> bool:
    if mode == "never" or summary.empty:
        return False
    if mode == "always":
        return True
    expected = len(seeds)
    completed = summary.set_index("model_name")["n_seeds_completed"].to_dict()
    return all(int(completed.get(model_name, 0)) == expected for model_name in models)


def write_interpretation(output_base: Path, seeds: list[int], models: list[str]) -> Path | None:
    per_seed_path = output_base / "pilot_seed_per_run_results.csv"
    summary_path = output_base / "pilot_seed_comparison.csv"
    if not per_seed_path.exists() or not summary_path.exists():
        return None

    per_seed = pd.read_csv(per_seed_path)
    summary = pd.read_csv(summary_path)
    if per_seed.empty or summary.empty:
        return None

    ranked = summary.sort_values("forged_pixel_f1_mean", ascending=False).reset_index(drop=True)
    top = ranked.iloc[0]
    stable = ranked[
        ranked.apply(lambda row: _coefficient_of_variation(row, "forged_pixel_f1") <= 0.25, axis=1)
    ]["model_name"].tolist()
    stable_text = ", ".join(stable) if stable else "no model met the <=25% forged-F1 CV stability rule"

    segformer_deeplab = {"segformer_b0", "deeplabv3plus"}
    top_two_preserved = _top_two_preserved(per_seed, segformer_deeplab)
    available_top_two = set(ranked.head(2)["model_name"])
    top_two_text = (
        "preserved across every completed seed"
        if top_two_preserved
        else f"not perfectly preserved; mean-ranking top two are {', '.join(ranked.head(2)['model_name'])}"
    )

    eff_rows = per_seed[per_seed["model_name"] == "efficientnetb0_unet"]
    if eff_rows.empty:
        eff_text = "EfficientNetB0 U-Net was not available in the completed runs."
    else:
        precision = float(eff_rows["forged_pixel_precision"].mean())
        recall = float(eff_rows["forged_pixel_recall"].mean())
        authentic_fp = float(eff_rows["authentic_pred_positive_rate"].mean())
        eff_text = (
            "EfficientNetB0 U-Net behaves conservatively when its forged precision is higher than "
            f"its forged recall. In these completed runs, forged precision is {precision:.4f}, "
            f"forged recall is {recall:.4f}, and the authentic positive rate is {authentic_fp:.4f}."
        )

    resnet_text = (
        "ResNet50 U-Net is intentionally not part of this seed-stability shortlist. In the previous "
        "pilot run it produced weak forged-only localization and relied on a very low selected "
        "threshold, which made it less attractive as a main candidate than SegFormer-B0, DeepLabV3+, "
        "and EfficientNetB0 U-Net."
    )

    lines = [
        "# Pilot Experiment Interpretation",
        "",
        "This file is generated from completed seed-stability outputs, not from a single seed.",
        "",
        "## Setup",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Models: {', '.join(models)}",
        "- Split: 60/20/20 stratified group-aware split with `image_id` / `group_id` as the group key",
        "- Main localization metric: forged-only pixel F1",
        "",
        "## Stability",
        "",
        f"The strongest mean forged-only pixel F1 belongs to `{top['model_name']}` "
        f"({_metric_text(top, 'forged_pixel_f1')}).",
        f"Models treated as stable by the <=25% coefficient-of-variation rule: {stable_text}.",
        "",
        "## SegFormer-B0 And DeepLabV3+",
        "",
        f"SegFormer-B0 and DeepLabV3+ superiority is {top_two_text}. "
        f"The mean-ranking top two set is {', '.join(available_top_two)}.",
        "",
        "## EfficientNetB0 U-Net",
        "",
        eff_text,
        "",
        "## ResNet50 U-Net",
        "",
        resnet_text,
        "",
        "## Why Forged-Only Pixel F1 Is Primary",
        "",
        "All-image pixel metrics can be inflated by authentic samples because their masks are all-zero. "
        "A model can look good on all-image pixel averages by predicting little or no manipulated area, "
        "while still failing to localize forged regions. Forged-only pixel F1 directly evaluates the "
        "segmentation quality on images where manipulated pixels exist, so it is the primary localization "
        "metric for this pilot stage.",
        "",
        "## Summary Table",
        "",
        "| model_name | forged_pixel_f1 | forged_pixel_iou | image_f1 | image_roc_auc | average_epochs_ran |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    _metric_text(row, "forged_pixel_f1"),
                    _metric_text(row, "forged_pixel_iou"),
                    _metric_text(row, "image_f1"),
                    _metric_text(row, "image_roc_auc"),
                    f"{float(row['average_epochs_ran']):.2f}",
                ]
            )
            + " |"
        )
    path = output_base / "pilot_experiment_interpretation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_design_record(args: argparse.Namespace) -> None:
    args.output_base.mkdir(parents=True, exist_ok=True)
    subset_policy = (
        f"fixed_subset_index={args.fixed_subset_index}; the exact rows in this CSV are reused for all split seeds"
        if args.fixed_subset_index is not None
        else (
        f"fixed_subset_seed={args.fixed_subset_seed}; the same sampled 300+300 image subset is reused for all split seeds"
        if args.fixed_subset_seed is not None
        else "subset_seed follows each seed; each seed may sample a different 300+300 image subset"
        )
    )
    record = {
        "experiment_name": "deney_2_pilot_seed_stability",
        "dataset_root": str(args.dataset_root),
        "output_base": str(args.output_base),
        "seeds": args.seeds,
        "models": args.models,
        "samples_per_class": args.samples_per_class,
        "image_size": args.img_size,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "early_stopping_patience": ExperimentConfig().early_stopping_patience,
        "reduce_lr_patience": ExperimentConfig().reduce_lr_patience,
        "reduce_lr_factor": ExperimentConfig().reduce_lr_factor,
        "threshold_values": list(ExperimentConfig().threshold_values),
        "image_decision_mode": ExperimentConfig().image_decision_mode,
        "split": "60/20/20 stratified group-aware split",
        "group_column": "image_id/group_id",
        "subset_policy": subset_policy,
        "fixed_subset_seed": args.fixed_subset_seed,
        "fixed_subset_index": str(args.fixed_subset_index) if args.fixed_subset_index is not None else None,
        "preprocessing": [
            "Images are resized to img_size x img_size.",
            "Authentic images receive all-zero masks.",
            "Forged .npy masks are converted to binary masks.",
            "Multi-channel forged masks are reduced with np.any(mask > 0, axis=0).",
            "Keras models use model-specific preprocessing; PyTorch models use ImageNet mean/std normalization.",
        ],
        "augmentation": [
            "Applied only to train split.",
            "Horizontal flip with p=0.5.",
            "Vertical flip with p=0.5.",
            "Random 90-degree rotation.",
            "Brightness/contrast jitter with alpha in [0.85, 1.15] and beta in [-18, 18].",
        ],
        "losses_and_selection": [
            "Keras segmentation loss: 0.5 * BCE + 0.5 * Dice loss.",
            "PyTorch primary segmentation loss: BCEWithLogits + Dice loss.",
            "PyTorch forensic heads may add edge/aux/image losses when the model exposes them.",
            "Best checkpoint is selected by validation Dice.",
            "Threshold is tuned only on val-tune; internal-test is never used for threshold selection.",
        ],
        "fairness_controls": [
            "Use --fixed-subset-index to reuse the exact first-experiment 300+300 subset.",
            "Within each seed folder, all models reuse the same generated split CSV files.",
            "All hyperparameters are kept at the first-pilot defaults unless explicitly overridden in the command.",
            "The only intended varying factor across seed folders is the seed-driven split/training randomness.",
        ],
        "notes": [
            "All models within the same pilot_seed_<seed> folder reuse the same split CSV files.",
            "Kaggle test_images are only used for submission generation, not for threshold tuning or internal testing.",
            "Primary localization stability metric is forged-only pixel F1/IoU, because all-image pixel means can be inflated by authentic all-zero masks.",
            "For strict comparison against the previous single-seed pilot, compare the new seed=42 rerun produced by this script under the same code revision.",
        ],
    }
    (args.output_base / "pilot_experiment_design.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = f"""# Pilot Seed Experiment Design

## Purpose

Small-subset results are repeated over multiple seeds to test whether the model ranking is stable.

## Dataset

- Dataset root: `{args.dataset_root}`
- Samples per class: `{args.samples_per_class}`
- Models: `{', '.join(args.models)}`
- Seeds: `{', '.join(str(seed) for seed in args.seeds)}`

## Split Policy

- Split ratio: train / val-tune / internal-test = 60 / 20 / 20
- Split type: stratified and group-aware
- Group key: `image_id` / `group_id`
- Leakage rule: the same image group cannot appear in more than one split

## Subset Policy

{subset_policy}

For the fairest comparison with the first pilot experiment, use `--fixed-subset-index` with the first experiment's `_shared_splits_seed42_subset/full_dataset_index.csv`. In that mode, the 300 authentic and 300 forged images are identical to the first pilot; only the seed-dependent split/training randomness changes.

## Training Configuration

- Image size: `{args.img_size}`
- Batch size: `{args.batch_size}`
- Max epochs: `{args.epochs}`
- Learning rate: `{args.lr}`
- Optimizer: `{args.optimizer}`
- Early stopping patience: `{ExperimentConfig().early_stopping_patience}`
- ReduceLROnPlateau patience/factor: `{ExperimentConfig().reduce_lr_patience}` / `{ExperimentConfig().reduce_lr_factor}`
- Checkpoint monitor: validation Dice
- Keras loss: `0.5 * BCE + 0.5 * Dice Loss`
- PyTorch primary loss: `BCEWithLogits + Dice Loss`
- PyTorch forensic auxiliary losses: edge/aux/image losses are used only when the model exposes those heads

## Preprocessing And Augmentation

- Images are resized to `{args.img_size}x{args.img_size}`.
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
"""
    (args.output_base / "pilot_experiment_design.md").write_text(md, encoding="utf-8")


def backup_existing_model_dir(model_dir: Path, backup_root: Path) -> None:
    if not model_dir.exists():
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / model_dir.parent.name / model_dir.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(model_dir), str(destination))
    print(f"Moved existing run to backup: {destination}", flush=True)


def _csv_sample_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path)["sample_id"].astype(str))


def backup_existing_split_root(split_root: Path, backup_root: Path) -> None:
    if not split_root.exists():
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / split_root.parent.name / split_root.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(split_root), str(destination))
    print(f"Moved existing split cache to backup: {destination}", flush=True)


def materialize_fixed_subset_index(
    config: ExperimentConfig,
    fixed_subset_index: Path | None,
    fresh: bool,
    backup_root: Path,
) -> None:
    if fixed_subset_index is None:
        return
    if not fixed_subset_index.exists():
        raise FileNotFoundError(f"Fixed subset index not found: {fixed_subset_index}")
    split_root = config.resolved_split_root()
    split_root.mkdir(parents=True, exist_ok=True)
    target = split_root / "full_dataset_index.csv"
    if target.exists() and _csv_sample_ids(target) != _csv_sample_ids(fixed_subset_index):
        if not fresh:
            raise RuntimeError(
                f"Existing split cache at {split_root} was created from a different subset. "
                "For a fair fixed-subset run, restart this seed with --fresh so the old split "
                "cache and model folders are backed up before new split files are generated."
            )
        backup_existing_split_root(split_root, backup_root)
        split_root.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(fixed_subset_index, target)
        print(f"Copied fixed subset index to {target}", flush=True)


def validate_split_file_paths(config: ExperimentConfig) -> None:
    from luc_forgery_pipeline.data import make_or_load_splits

    _, train_df, val_df, test_df = make_or_load_splits(config)
    missing: list[str] = []
    for split_name, split_df in [("train", train_df), ("val_tune", val_df), ("internal_test", test_df)]:
        for row in split_df.itertuples(index=False):
            if not Path(row.image_path).exists():
                missing.append(f"{split_name}: image not found: {row.image_path}")
            if int(row.label) == 1 and not Path(row.mask_path).exists():
                missing.append(f"{split_name}: mask not found: {row.mask_path}")
            if len(missing) >= 20:
                break
        if len(missing) >= 20:
            break
    if missing:
        preview = "\n".join(missing)
        raise FileNotFoundError(
            "One or more split rows point to files that cannot be read. "
            "Check the dataset mount/path before training.\n"
            f"{preview}"
        )


def main() -> None:
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)
    write_design_record(args)

    if not args.aggregate_only:
        backup_root = args.output_base / f"_fresh_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for seed in args.seeds:
            seed_root = args.output_base / f"pilot_seed_{seed}"
            config = ExperimentConfig(
                dataset_root=args.dataset_root,
                output_root=seed_root,
                seed=seed,
                samples_per_class=args.samples_per_class,
                subset_seed=args.fixed_subset_seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                learning_rate=args.lr,
                optimizer=args.optimizer,
                models=tuple(args.models),
                num_workers=args.num_workers,
            )
            materialize_fixed_subset_index(config, args.fixed_subset_index, args.fresh, backup_root)
            validate_split_file_paths(config)
            for model_name in args.models:
                model_dir = seed_root / model_name
                if args.fresh:
                    backup_existing_model_dir(model_dir, backup_root)
                if model_run_is_complete(model_dir):
                    print(f"\n===== Seed {seed} | {model_name} already complete; skipping =====")
                    continue
                print(f"\n===== Seed {seed} | Training {model_name} =====")
                print(f"State before run: {describe_model_state(model_dir)}")
                train_model_for_seed(model_name, config)
                missing_after = model_missing_artifacts(model_dir)
                if missing_after:
                    raise RuntimeError(
                        f"Seed {seed} | {model_name} finished without required artifacts: {missing_after}. "
                        f"Check the model folder and console output for the earlier failure."
                    )

    summary = aggregate_results(args.output_base, args.seeds, args.models)
    if summary.empty:
        print("No completed pilot results found yet.")
    else:
        print(summary.to_string(index=False))
        print(f"\nWrote: {args.output_base / 'pilot_seed_comparison.csv'}")
        print(f"Wrote: {args.output_base / 'pilot_seed_comparison_table.md'}")
        if should_write_interpretation(args.interpretation, summary, args.seeds, args.models):
            interpretation_path = write_interpretation(args.output_base, args.seeds, args.models)
            if interpretation_path is not None:
                print(f"Wrote: {interpretation_path}")
        elif args.interpretation == "auto":
            print("Interpretation not written yet because not every requested seed/model result is complete.")


if __name__ == "__main__":
    main()
