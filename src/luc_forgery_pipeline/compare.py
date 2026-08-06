from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from .config import ExperimentConfig
from .utils import ensure_dir


KEY_METRICS = [
    "pixel_f1",
    "pixel_dice",
    "pixel_iou",
    "pixel_precision",
    "pixel_recall",
    "pixel_specificity",
]


def _load_model_summary(model_dir: Path, model_name: str) -> pd.DataFrame | None:
    path = model_dir / "test_metrics_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = {"model_name": model_name}
    for record in df.to_dict(orient="records"):
        row[record["metric"]] = record["mean"]
        if record["metric"].startswith("pixel_"):
            row[f"{record['metric']}_std"] = record["std"]
            row[f"{record['metric']}_ci95_low"] = record["ci95_low"]
            row[f"{record['metric']}_ci95_high"] = record["ci95_high"]
    threshold_path = model_dir / "threshold_analysis.csv"
    if threshold_path.exists():
        th = pd.read_csv(threshold_path)
        selected = th[th["selected"]].iloc[0]
        row["best_threshold"] = selected["threshold"]
        row["val_pixel_f1"] = selected["pixel_f1_mean"]
        row["val_image_f1"] = selected["image_f1"]
    return pd.DataFrame([row])


def build_all_models_summary(config: ExperimentConfig, model_names: Iterable[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for model_name in model_names:
        summary = _load_model_summary(config.model_dir(model_name), model_name)
        if summary is not None:
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_statistical_tests(config: ExperimentConfig, model_names: Iterable[str]) -> pd.DataFrame:
    per_image = {}
    for model_name in model_names:
        path = config.model_dir(model_name) / "test_per_image_metrics.csv"
        if path.exists():
            per_image[model_name] = pd.read_csv(path)

    rows = []
    for left, right in combinations(per_image.keys(), 2):
        merged = per_image[left].merge(
            per_image[right],
            on="sample_id",
            suffixes=(f"_{left}", f"_{right}"),
        )
        for metric in KEY_METRICS:
            x = merged[f"{metric}_{left}"].astype(float).to_numpy()
            y = merged[f"{metric}_{right}"].astype(float).to_numpy()
            diff = x - y
            try:
                t_stat, t_p = ttest_rel(x, y, nan_policy="omit")
            except Exception:
                t_stat, t_p = np.nan, np.nan
            try:
                w_stat, w_p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, w_p = np.nan, np.nan
            rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "metric": metric,
                    "n_pairs": len(merged),
                    "mean_a": float(np.nanmean(x)),
                    "mean_b": float(np.nanmean(y)),
                    "mean_difference_a_minus_b": float(np.nanmean(diff)),
                    "paired_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                    "paired_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
                    "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
                    "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_comparison_barplots(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = ["pixel_f1", "pixel_dice", "pixel_iou", "image_f1", "image_roc_auc"]
    available = [m for m in metrics if m in summary.columns]
    if summary.empty or not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4), squeeze=False)
    for ax, metric in zip(axes[0], available):
        ax.bar(summary["model_name"], summary[metric].astype(float))
        ax.set_title(metric)
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_markdown_table(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        output_path.write_text("# Model Comparison\n\nNo completed model summaries found.\n", encoding="utf-8")
        return
    cols = [
        "model_name",
        "best_threshold",
        "pixel_f1",
        "pixel_dice",
        "pixel_iou",
        "image_accuracy",
        "image_f1",
        "image_roc_auc",
    ]
    cols = [c for c in cols if c in summary.columns]
    text = "# Model Comparison\n\n" + summary[cols].to_markdown(index=False) + "\n"
    output_path.write_text(text, encoding="utf-8")


def create_model_comparison(config: ExperimentConfig, model_names: Iterable[str]) -> Path:
    out_dir = ensure_dir(config.output_root / "model_comparison")
    summary = build_all_models_summary(config, model_names)
    summary.to_csv(out_dir / "all_models_summary.csv", index=False)
    tests = paired_statistical_tests(config, model_names)
    tests.to_csv(out_dir / "statistical_tests.csv", index=False)
    write_markdown_table(summary, out_dir / "model_comparison_table.md")
    save_comparison_barplots(summary, out_dir / "comparison_barplots.png")
    return out_dir
