from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import PREPROCESSORS, list_kaggle_test_images, load_binary_mask, load_rgb_image
from .utils import ensure_dir


EPS = 1e-7


def _safe_div(num: float, den: float, zero_value: float = 0.0) -> float:
    if den == 0:
        return zero_value
    return float(num / den)


def per_image_pixel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    true = (y_true.reshape(-1) > 0.5).astype(np.uint8)
    pred = (y_prob.reshape(-1) >= threshold).astype(np.uint8)

    tp = float(np.sum((true == 1) & (pred == 1)))
    fp = float(np.sum((true == 0) & (pred == 1)))
    fn = float(np.sum((true == 1) & (pred == 0)))
    tn = float(np.sum((true == 0) & (pred == 0)))

    precision = _safe_div(tp, tp + fp, zero_value=1.0 if true.sum() == 0 and pred.sum() == 0 else 0.0)
    recall = _safe_div(tp, tp + fn, zero_value=1.0 if true.sum() == 0 else 0.0)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn, zero_value=1.0 if true.sum() == 0 and pred.sum() == 0 else 0.0)
    iou = _safe_div(tp, tp + fp + fn, zero_value=1.0 if true.sum() == 0 and pred.sum() == 0 else 0.0)
    specificity = _safe_div(tn, tn + fp, zero_value=0.0)

    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_dice": dice,
        "pixel_iou": iou,
        "pixel_specificity": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def predict_dataframe(model, df: pd.DataFrame, config: ExperimentConfig, model_name: str) -> List[Dict[str, object]]:
    preprocess = PREPROCESSORS[model_name]
    results: List[Dict[str, object]] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Predict"):
        image = load_rgb_image(row.image_path, config.img_size)
        true_mask = load_binary_mask(row.mask_path, row.label, config.img_size)
        x = preprocess(image)[None, ...].astype(np.float32)
        pred = model.predict(x, verbose=0)[0].astype(np.float32)
        results.append(
            {
                "sample_id": row.sample_id,
                "image_id": row.image_id,
                "image_path": row.image_path,
                "mask_path": row.mask_path,
                "label": int(row.label),
                "y_true": true_mask,
                "y_prob": pred,
                "pred_max_probability": float(np.max(pred)),
            }
        )
    return results


def image_score_from_prediction(y_prob: np.ndarray, threshold: float, mode: str) -> float:
    if mode == "pred_mask_ratio":
        return float(np.mean(y_prob >= threshold))
    if mode == "max_probability":
        return float(np.max(y_prob))
    raise ValueError(f"Unknown image_decision_mode: {mode}")


def summarize_image_metrics(y_true: Iterable[int], y_score: Iterable[float], threshold: float) -> Tuple[Dict[str, float], np.ndarray]:
    labels = np.asarray(list(y_true)).astype(int)
    scores = np.asarray(list(y_score)).astype(float)
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = _safe_div(float(tn), float(tn + fp))
    try:
        roc_auc = float(roc_auc_score(labels, scores))
    except ValueError:
        roc_auc = float("nan")
    return (
        {
            "image_accuracy": float(accuracy_score(labels, preds)),
            "image_precision": float(precision),
            "image_recall_sensitivity": float(recall),
            "image_specificity": float(specificity),
            "image_f1": float(f1),
            "image_roc_auc": roc_auc,
            "threshold": float(threshold),
        },
        cm,
    )


def evaluate_thresholds(
    predictions: List[Dict[str, object]],
    thresholds: Iterable[float],
    image_decision_mode: str,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        pixel_rows = [
            per_image_pixel_metrics(item["y_true"], item["y_prob"], threshold) for item in predictions
        ]
        pixel_df = pd.DataFrame(pixel_rows)
        image_scores = [
            image_score_from_prediction(item["y_prob"], threshold, image_decision_mode) for item in predictions
        ]
        image_metrics, _ = summarize_image_metrics([item["label"] for item in predictions], image_scores, threshold)
        row = {"threshold": float(threshold)}
        for col in [
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
            "pixel_dice",
            "pixel_iou",
            "pixel_specificity",
        ]:
            row[f"{col}_mean"] = float(pixel_df[col].mean())
            row[f"{col}_std"] = float(pixel_df[col].std(ddof=1))
        row.update(image_metrics)
        rows.append(row)
    analysis = pd.DataFrame(rows)
    best_idx = analysis.sort_values(["pixel_f1_mean", "image_f1"], ascending=False).index[0]
    analysis["selected"] = False
    analysis.loc[best_idx, "selected"] = True
    return analysis


def build_per_image_metrics(
    predictions: List[Dict[str, object]],
    threshold: float,
    image_decision_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    rows = []
    for item in predictions:
        metrics = per_image_pixel_metrics(item["y_true"], item["y_prob"], threshold)
        score = image_score_from_prediction(item["y_prob"], threshold, image_decision_mode)
        pred_label = int(score >= threshold)
        rows.append(
            {
                "sample_id": item["sample_id"],
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "label": item["label"],
                "image_score": score,
                "pred_label": pred_label,
                **metrics,
            }
        )
    per_image = pd.DataFrame(rows)
    image_metrics, cm = summarize_image_metrics(per_image["label"], per_image["image_score"], threshold)
    return per_image, image_metrics, cm


def metric_summary_with_ci(per_image: pd.DataFrame, image_metrics: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for col in [
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pixel_dice",
        "pixel_iou",
        "pixel_specificity",
    ]:
        values = per_image[col].astype(float).to_numpy()
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci95 = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(
            {
                "metric": col,
                "mean": mean,
                "std": std,
                "ci95_low": mean - ci95,
                "ci95_high": mean + ci95,
                "n": int(len(values)),
            }
        )
    labels = per_image["label"].astype(int).to_numpy()
    scores = per_image["image_score"].astype(float).to_numpy()
    threshold = float(image_metrics["threshold"])
    image_ci = bootstrap_image_metric_ci(labels, scores, threshold)
    for key, value in image_metrics.items():
        ci = image_ci.get(key, {})
        rows.append(
            {
                "metric": key,
                "mean": value,
                "std": ci.get("std", np.nan),
                "ci95_low": ci.get("ci95_low", np.nan),
                "ci95_high": ci.get("ci95_high", np.nan),
                "n": len(per_image),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_image_metric_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    metric_values: Dict[str, List[float]] = {}
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        metrics, _ = summarize_image_metrics(labels[idx], scores[idx], threshold)
        for key, value in metrics.items():
            if key == "threshold" or not np.isfinite(value):
                continue
            metric_values.setdefault(key, []).append(float(value))
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in metric_values.items():
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            continue
        summary[key] = {
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return summary


def save_threshold_plot(threshold_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["pixel_f1_mean"], marker="o", label="Pixel F1")
    plt.plot(threshold_df["threshold"], threshold_df["image_f1"], marker="s", label="Image F1")
    selected = threshold_df[threshold_df["selected"]]
    if not selected.empty:
        plt.axvline(float(selected.iloc[0]["threshold"]), color="black", linestyle="--", linewidth=1)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_training_curves(log_path: Path, model_dir: Path) -> None:
    if not log_path.exists():
        return
    log = pd.read_csv(log_path)
    curve_specs = {
        "loss_curve.png": ("loss", "val_loss", "Loss"),
        "dice_curve.png": ("dice_coefficient", "val_dice_coefficient", "Dice"),
        "iou_curve.png": ("iou_coefficient", "val_iou_coefficient", "IoU"),
    }
    for filename, (train_col, val_col, title) in curve_specs.items():
        if train_col not in log.columns:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(log["epoch"], log[train_col], label=f"Train {title}")
        if val_col in log.columns:
            plt.plot(log["epoch"], log[val_col], label=f"Val {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_dir / filename, dpi=220)
        plt.close()


def save_final_figures(per_image: pd.DataFrame, cm: np.ndarray, model_dir: Path) -> None:
    labels = per_image["label"].astype(int).to_numpy()
    scores = per_image["image_score"].astype(float).to_numpy()

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Authentic", "Forged"])
    plt.yticks([0, 1], ["Authentic", "Forged"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png", dpi=220)
    plt.close()

    try:
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
            model_dir / "test_roc_auc_data.csv", index=False
        )
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_dir / "roc_auc_curve.png", dpi=220)
        plt.close()
    except ValueError:
        pd.DataFrame(columns=["fpr", "tpr", "threshold"]).to_csv(model_dir / "test_roc_auc_data.csv", index=False)

    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.r_[pr_thresholds, np.nan],
        }
    )
    pr_df.to_csv(model_dir / "test_precision_recall_data.csv", index=False)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(model_dir / "precision_recall_curve.png", dpi=220)
    plt.close()


def save_prediction_examples(
    predictions: List[Dict[str, object]],
    threshold: float,
    output_dir: Path,
    max_examples: int,
) -> None:
    ensure_dir(output_dir)
    for idx, item in enumerate(predictions[:max_examples]):
        image = load_rgb_image(item["image_path"], item["y_true"].shape[0]).astype(np.uint8)
        true_mask = item["y_true"].squeeze()
        pred_prob = item["y_prob"].squeeze()
        pred_mask = (pred_prob >= threshold).astype(np.float32)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(true_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("GT")
        axes[2].imshow(pred_prob, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Prob")
        axes[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Pred")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{idx:03d}_{item['sample_id'].replace('/', '_')}.png", dpi=180)
        plt.close(fig)


def create_submission(model, config: ExperimentConfig, model_name: str, threshold: float, output_path: Path) -> None:
    sample_path = config.dataset_root / "sample_submission.csv"
    test_images = list_kaggle_test_images(config.dataset_root)
    preprocess = PREPROCESSORS[model_name]
    rows = []
    for image_path in tqdm(test_images, desc="Submission"):
        image = load_rgb_image(image_path, config.img_size)
        pred = model.predict(preprocess(image)[None, ...].astype(np.float32), verbose=0)[0]
        score = float(np.max(pred))
        rle = rle_encode((pred.squeeze() >= threshold).astype(np.uint8))
        rows.append(
            {
                "id": image_path.stem,
                "file": image_path.name,
                "label": int(score >= threshold),
                "score": score,
                "rle": rle,
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


def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
