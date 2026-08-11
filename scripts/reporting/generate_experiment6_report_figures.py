"""
Deney 6 raporu icin eksik Sekil 4 ve Sekil 5 grafiklerini uretir.

Calistirma:
    python analysis_review/generate_experiment6_report_figures.py

Uretilen dosyalar:
    analysis_review/figure4_experiment5_vs_experiment6_balanced.png
    analysis_review/figure5_component_f1_by_iou_balanced.png
    analysis_review/figure5_component_metrics_balanced.csv

Not:
    Sekil 5 icin dogru kaynak test_component_details_*.csv degildir.
    O dosyalar per-image component eslesme detaylarini tutar.
    IoU 0.10/0.25/0.50 precision/recall/F1 ozetleri test_results_by_strategy.csv
    dosyasindadir.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])


ensure_package("matplotlib")
import matplotlib.pyplot as plt


ROOT = Path("deney_6/experiments_full/experiment6_smallmask_384")
ANALYSIS = Path("analysis_review")
ANALYSIS.mkdir(exist_ok=True)

MODEL_ORDER = ["EfficientNetB0-UNet 384", "SegFormer-B0 384"]
MODEL_DIR = {
    "EfficientNetB0-UNet 384": "efficientnetb0_unet_rgb_384_smallmask",
    "SegFormer-B0 384": "segformer_b0_rgb_384_smallmask",
}


def prettify_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def make_figure4():
    """Deney 5 vs Deney 6 balanced-vs-balanced ana metrik karsilastirmasi."""
    path = Path(r"c:/Users/asus/Documents/bitirmeProjesi/bitirmeProjesi/analysis_review/experiment_6_vs_experiment_5_reconstructed_comparison.csv")
    df = pd.read_csv(path)
    df = df[df["exp6_strategy"].eq("balanced_final_score")].copy()
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values("model")

    metrics = [
        ("forged_dice", "Forged Dice", "higher"),
        ("q1_dice", "Q1 Dice", "higher"),
        ("q2_dice", "Q2 Dice", "higher"),
        ("component_f1_iou010", "Component F1 @0.10", "higher"),
        ("authentic_fp_rate", "Authentic FP Rate", "lower"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4.2), sharey=False)
    x = np.arange(len(df))
    width = 0.36
    colors = {"Deney 5": "#8a8f98", "Deney 6": "#2f6f9f"}

    for ax, (suffix, title, direction) in zip(axes, metrics):
        y5 = df[f"exp5_{suffix}"].astype(float).to_numpy()
        y6 = df[f"exp6_{suffix}"].astype(float).to_numpy()
        ax.bar(x - width / 2, y5, width, label="Deney 5", color=colors["Deney 5"])
        ax.bar(x + width / 2, y6, width, label="Deney 6", color=colors["Deney 6"])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"].astype(str), rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, max(np.nanmax(y5), np.nanmax(y6)) * 1.25 + 1e-6)
        prettify_axis(ax)
        for i, (a, b) in enumerate(zip(y5, y6)):
            delta = b - a
            sign = "+" if delta >= 0 else ""
            if direction == "lower":
                marker = "↓" if delta < 0 else "↑"
            else:
                marker = "↑" if delta > 0 else "↓"
            ax.text(i, max(a, b) * 1.04 + 0.005, f"{marker}{sign}{delta:.3f}", ha="center", fontsize=8)

    axes[0].set_ylabel("Skor / oran")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Deney 5 ve Deney 6 balanced strateji karşılaştırması", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = ANALYSIS / "figure4_experiment5_vs_experiment6_balanced.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_figure5():
    """Balanced strateji icin IoU esigine gore component F1 karsilastirmasi."""
    rows = []
    for model, model_dir in MODEL_DIR.items():
        path = Path(r"c:/Users/asus/Documents/bitirmeProjesi/bitirmeProjesi/deney_6/experiments_full/experiment6_smallmask_384") / model_dir / "test_results_by_strategy.csv"
        df = pd.read_csv(path)
        row = df[df["strategy"].eq("balanced_final_score")].iloc[0]
        for suffix, label in [("010", "0.10"), ("025", "0.25"), ("050", "0.50")]:
            rows.append(
                {
                    "model": model,
                    "iou_threshold": label,
                    "component_precision": row[f"component_precision_iou{suffix}"],
                    "component_recall": row[f"component_recall_iou{suffix}"],
                    "component_f1": row[f"component_f1_iou{suffix}"],
                    "component_tp": row[f"component_tp_iou{suffix}"],
                    "component_fp": row[f"component_fp_iou{suffix}"],
                    "component_fn": row[f"component_fn_iou{suffix}"],
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(ANALYSIS / "figure5_component_metrics_balanced.csv", index=False, encoding="utf-8-sig")

    pivot = metrics.pivot(index="iou_threshold", columns="model", values="component_f1").loc[["0.10", "0.25", "0.50"]]
    x = np.arange(len(pivot.index))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, pivot["EfficientNetB0-UNet 384"], width, label="EfficientNetB0-UNet 384", color="#5b8c5a")
    ax.bar(x + width / 2, pivot["SegFormer-B0 384"], width, label="SegFormer-B0 384", color="#2f6f9f")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("Component match IoU eşiği")
    ax.set_ylabel("Component F1")
    ax.set_ylim(0, max(pivot.max()) * 1.22)
    ax.set_title("Balanced strateji için bileşen F1 karşılaştırması")
    prettify_axis(ax)
    ax.legend(frameon=False)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=3)

    fig.tight_layout()
    out = ANALYSIS / "figure5_component_f1_by_iou_balanced.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


if __name__ == "__main__":
    fig4 = make_figure4()
    fig5 = make_figure5()
    print(f"Saved: {fig4}")
    print(f"Saved: {fig5}")
