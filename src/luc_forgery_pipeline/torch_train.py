from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import ExperimentConfig, FORENSIC_MODEL_NAMES, PYTORCH_MODEL_NAMES
from .data import (
    copy_shared_splits_to_model_dir,
    list_kaggle_test_images,
    load_binary_mask,
    load_rgb_image,
    make_or_load_splits,
)
from .evaluation import (
    build_per_image_metrics,
    evaluate_thresholds,
    metric_summary_with_ci,
    rle_encode,
    save_final_figures,
    save_prediction_examples,
    save_threshold_plot,
    save_training_curves,
)
from .reports import write_model_report
from .torch_data import IMAGENET_MEAN, IMAGENET_STD, make_torch_loader
from .torch_losses import dice_from_logits, forensic_loss, iou_from_logits, primary_logits
from .torch_models import build_torch_model
from .utils import ensure_dir, environment_info, save_json, set_global_determinism


def torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_paths(model_dir: Path) -> Dict[str, Path]:
    return {
        "best": model_dir / "best_model.pt",
        "last": model_dir / "last_model.pt",
        "best_dir": model_dir / "best_model",
        "last_dir": model_dir / "last_model",
        "training_log": model_dir / "training_log.csv",
    }


def make_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_dice: float,
    patience_counter: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_dice": best_val_dice,
            "patience_counter": patience_counter,
        },
        path,
    )


def load_checkpoint_if_available(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float, int]:
    if not path.exists():
        return 0, -np.inf, 0
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        float(checkpoint.get("best_val_dice", -np.inf)),
        int(checkpoint.get("patience_counter", 0)),
    )


def run_epoch(model, loader, device, optimizer=None) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice_coefficient": 0.0, "iou_coefficient": 0.0}
    n_batches = 0
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="Train" if training else "Val", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            logits = primary_logits(outputs)
            loss = forensic_loss(outputs, masks, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["dice_coefficient"] += float(dice_from_logits(logits, masks).detach().cpu())
            totals["iou_coefficient"] += float(iou_from_logits(logits, masks).detach().cpu())
            n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def append_training_log(log_path: Path, row: Dict[str, float]) -> None:
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def export_torch_model_files(model_dir: Path, paths: Dict[str, Path]) -> None:
    for source, target_dir in [(paths["best"], paths["best_dir"]), (paths["last"], paths["last_dir"])]:
        ensure_dir(target_dir)
        if source.exists():
            shutil.copy2(source, target_dir / source.name)


def predict_torch_dataframe(model, df: pd.DataFrame, config: ExperimentConfig, device: torch.device) -> List[Dict[str, object]]:
    model.eval()
    predictions: List[Dict[str, object]] = []
    with torch.no_grad():
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Predict"):
            image = load_rgb_image(row.image_path, config.img_size)
            mask = load_binary_mask(row.mask_path, row.label, config.img_size)
            x = image / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
            tensor = torch.from_numpy(x).to(device)
            outputs = model(tensor)
            prob = torch.sigmoid(primary_logits(outputs))[0, 0].detach().cpu().numpy()[..., None].astype(np.float32)
            predictions.append(
                {
                    "sample_id": row.sample_id,
                    "image_id": row.image_id,
                    "image_path": row.image_path,
                    "mask_path": row.mask_path,
                    "label": int(row.label),
                    "y_true": mask,
                    "y_prob": prob,
                    "pred_max_probability": float(np.max(prob)),
                }
            )
    return predictions


def create_torch_submission(model, config: ExperimentConfig, threshold: float, output_path: Path, device: torch.device) -> None:
    sample_path = config.dataset_root / "sample_submission.csv"
    rows = []
    model.eval()
    with torch.no_grad():
        for image_path in tqdm(list_kaggle_test_images(config.dataset_root), desc="Submission"):
            image = load_rgb_image(image_path, config.img_size)
            x = image / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
            outputs = model(torch.from_numpy(x).to(device))
            prob = torch.sigmoid(primary_logits(outputs))[0, 0].detach().cpu().numpy()
            score = float(np.max(prob))
            rows.append(
                {
                    "id": image_path.stem,
                    "file": image_path.name,
                    "label": int(score >= threshold),
                    "score": score,
                    "rle": rle_encode((prob >= threshold).astype(np.uint8)),
                }
            )

    sub = pd.DataFrame(rows)
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        id_col = sample.columns[0]
        target_cols = [c for c in sample.columns if c != id_col]
        merge_key = "file" if sample[id_col].astype(str).isin(sub["file"].astype(str)).any() else "id"
        sub_for_merge = sub.rename(columns={merge_key: id_col})
        sample_keys = sample[[id_col]].copy()
        sample_keys[id_col] = sample_keys[id_col].astype(str)
        sub_for_merge[id_col] = sub_for_merge[id_col].astype(str)
        merged = sample_keys.merge(sub_for_merge, on=id_col, how="left")
        out = sample.copy()
        for target_col in target_cols:
            lower = target_col.lower()
            if any(token in lower for token in ["rle", "encoded", "mask", "pixel"]):
                out[target_col] = merged["rle"].fillna("")
            elif any(token in lower for token in ["score", "prob"]):
                out[target_col] = merged["score"].fillna(0.0)
            else:
                out[target_col] = merged["label"].fillna(0).astype(int)
        if not target_cols:
            out["label"] = merged["label"].fillna(0).astype(int)
        out.to_csv(output_path, index=False)
    else:
        sub[["id", "label"]].to_csv(output_path, index=False)


def train_one_torch_model(model_name: str, config: ExperimentConfig) -> Dict[str, float]:
    if model_name not in PYTORCH_MODEL_NAMES + FORENSIC_MODEL_NAMES:
        raise ValueError(f"Unknown PyTorch model: {model_name}")

    set_global_determinism(config.seed)
    device = torch_device()
    model_dir = ensure_dir(config.model_dir(model_name))
    print(f"[{model_name}] device={device} model_dir={model_dir}", flush=True)
    save_json(config.to_json_dict(), model_dir / "experiment_config.json")
    env = environment_info()
    env["torch"] = torch.__version__
    env["torch_device"] = str(device)
    save_json(env, model_dir / "environment_info.json")

    full_df, train_df, val_df, test_df = make_or_load_splits(config)
    print(
        f"[{model_name}] splits loaded: train={len(train_df)} val={len(val_df)} test={len(test_df)}",
        flush=True,
    )
    copy_shared_splits_to_model_dir(full_df, train_df, val_df, test_df, model_dir)
    dataset_summary = (
        pd.concat(
            [
                train_df.assign(split="train"),
                val_df.assign(split="val_tune"),
                test_df.assign(split="internal_test"),
            ],
            ignore_index=True,
        )
        .groupby(["split", "class_name"])
        .size()
        .rename("count")
        .reset_index()
    )

    print(f"[{model_name}] building model...", flush=True)
    model = build_torch_model(model_name).to(device)
    print(f"[{model_name}] model built.", flush=True)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config.reduce_lr_patience,
        factor=config.reduce_lr_factor,
        min_lr=1e-7,
    )
    paths = checkpoint_paths(model_dir)
    start_epoch, best_val_dice, patience_counter = load_checkpoint_if_available(
        paths["last"], model, optimizer, scheduler, device
    )
    (model_dir / "model_summary.txt").write_text(str(model), encoding="utf-8")

    train_loader = make_torch_loader(train_df, config, augment=True, shuffle=True, seed_offset=100)
    val_loader = make_torch_loader(val_df, config, augment=False, shuffle=False, seed_offset=200)
    print(f"[{model_name}] loaders ready. start_epoch={start_epoch} max_epochs={config.epochs}", flush=True)

    for epoch in range(start_epoch, config.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_metrics["dice_coefficient"])
        row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "dice_coefficient": train_metrics["dice_coefficient"],
            "iou_coefficient": train_metrics["iou_coefficient"],
            "val_loss": val_metrics["loss"],
            "val_dice_coefficient": val_metrics["dice_coefficient"],
            "val_iou_coefficient": val_metrics["iou_coefficient"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_training_log(paths["training_log"], row)
        improved = val_metrics["dice_coefficient"] > best_val_dice
        if improved:
            best_val_dice = val_metrics["dice_coefficient"]
            patience_counter = 0
            save_checkpoint(paths["best"], model, optimizer, scheduler, epoch, best_val_dice, patience_counter)
        else:
            patience_counter += 1
        save_checkpoint(paths["last"], model, optimizer, scheduler, epoch, best_val_dice, patience_counter)
        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"loss={row['loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_dice={row['val_dice_coefficient']:.4f}"
        )
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if paths["training_log"].exists():
        log = pd.read_csv(paths["training_log"])
        save_json(log.to_dict(orient="list"), model_dir / "history.json")
    save_training_curves(paths["training_log"], model_dir)

    if paths["best"].exists():
        checkpoint = torch.load(paths["best"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    export_torch_model_files(model_dir, paths)

    val_predictions = predict_torch_dataframe(model, val_df, config, device)
    threshold_df = evaluate_thresholds(val_predictions, config.threshold_values, config.image_decision_mode)
    threshold_df.to_csv(model_dir / "threshold_analysis.csv", index=False)
    save_threshold_plot(threshold_df, model_dir / "threshold_analysis.png")
    best_threshold = float(threshold_df[threshold_df["selected"]].iloc[0]["threshold"])

    test_predictions = predict_torch_dataframe(model, test_df, config, device)
    per_image, image_metrics, cm = build_per_image_metrics(test_predictions, best_threshold, config.image_decision_mode)
    per_image.to_csv(model_dir / "test_per_image_metrics.csv", index=False)
    test_summary = metric_summary_with_ci(per_image, image_metrics)
    test_summary.to_csv(model_dir / "test_metrics_summary.csv", index=False)
    pd.DataFrame(cm, index=["true_authentic", "true_forged"], columns=["pred_authentic", "pred_forged"]).to_csv(
        model_dir / "test_confusion_matrix.csv"
    )
    save_final_figures(per_image, cm, model_dir)
    save_prediction_examples(
        test_predictions,
        best_threshold,
        model_dir / "prediction_examples",
        max_examples=config.prediction_examples,
    )
    create_torch_submission(model, config, best_threshold, model_dir / "submission.csv", device)
    write_model_report(
        model_dir,
        model_name,
        config,
        dataset_summary,
        threshold_df,
        test_summary,
        image_metrics,
        framework="PyTorch",
    )
    summary = {
        "model_name": model_name,
        "best_threshold": best_threshold,
        **{row["metric"]: row["mean"] for row in test_summary.to_dict(orient="records")},
    }
    save_json({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summary.items()}, model_dir / "run_summary.json")
    return summary
