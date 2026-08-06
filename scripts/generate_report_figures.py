"""Generate the report figures that are derived from archived CSV results.

Run from the repository root:

    python scripts/generate_report_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results" / "report_sources"
OUTPUT_DIR = ROOT / "docs" / "figures"

MODEL_LABELS = {
    "segformer_b0": "SegFormer-B0",
    "segformer_b0_rgb_full": "SegFormer-B0",
    "deeplabv3plus": "DeepLabv3+",
    "efficientnetb0_unet": "EfficientNetB0-UNet",
    "efficientnetb0_unet_rgb_full": "EfficientNetB0-UNet",
    "resnet50_unet": "ResNet50-UNet",
    "doagan": "DOA-GAN",
    "siamese_cmfd": "Siamese-CMFD",
    "busternet": "BusterNet",
    "qdl_cmfd": "QDL-CMFD",
    "cmfdformer": "CMFDFormer",
    "selfcorr_cmfd": "SelfCorr-CMFD",
    "plain_unet": "Plain U-Net",
    "unetplusplus": "U-Net++ ResNet34",
    "unetpp_resnet34_rgb_full": "U-Net++ ResNet34",
    "mvssnet": "MVSS-Net",
    "mvssnetpp": "MVSS-Net++",
    "dinov2_seg": "DINOv2-lite",
    "dinov2_lite_decoder_rgb_full": "DINOv2-lite",
    "mantranet": "ManTraNet",
}


def label_model(value: str) -> str:
    return MODEL_LABELS.get(value, value.replace("_", " ").title())


def save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def experiment1_model_screening() -> None:
    df = pd.read_csv(SOURCE_DIR / "experiment1_model_comparison.csv")
    df = df[["model_name", "forged_pixel_f1"]].dropna().sort_values("forged_pixel_f1")
    labels = [label_model(value) for value in df["model_name"]]
    values = df["forged_pixel_f1"].to_numpy()
    top_three = set(df.nlargest(3, "forged_pixel_f1")["model_name"])
    colors = ["#dd8452" if name in top_three else "#4c72b0" for name in df["model_name"]]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.set_title("Deney 1 — Forged Pixel F1 ile Model Taraması", weight="bold")
    ax.set_xlabel("Forged Pixel F1")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "experiment1_forged_pixel_f1.png")


def experiment2_seed_stability() -> None:
    df = pd.read_csv(SOURCE_DIR / "experiment2_per_seed.csv")
    order = (
        df.groupby("model_name")["forged_pixel_f1"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    x = np.arange(len(order))
    means = df.groupby("model_name")["forged_pixel_f1"].mean().reindex(order)
    stds = df.groupby("model_name")["forged_pixel_f1"].std().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        markersize=8,
        capsize=5,
        color="#2f5f8f",
        ecolor="#6f7f8f",
        linewidth=2,
        label="Ortalama ± standart sapma",
        zorder=3,
    )
    offsets = {seed: offset for seed, offset in zip(sorted(df["seed"].unique()), [-0.12, 0, 0.12])}
    seed_colors = {42: "#55a868", 123: "#c44e52", 2025: "#8172b2"}
    for seed, group in df.groupby("seed"):
        values = group.set_index("model_name")["forged_pixel_f1"].reindex(order)
        ax.scatter(
            x + offsets.get(seed, 0),
            values,
            s=42,
            color=seed_colors.get(seed, "#777777"),
            label=f"Seed {seed}",
            zorder=4,
        )
    ax.set_title("Deney 2 — Üç Tohumda Forged Pixel F1 Kararlılığı", weight="bold")
    ax.set_ylabel("Forged Pixel F1")
    ax.set_xticks(x, [label_model(value) for value in order], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False)
    save(fig, "experiment2_seed_stability.png")


def experiment4_confusion_matrices() -> None:
    df = pd.read_csv(SOURCE_DIR / "experiment4_confusion.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, row in zip(axes.flat, df.itertuples(index=False)):
        matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() * 0.55 else "#1f2933",
                    fontsize=13,
                    weight="bold",
                )
        ax.set_title(label_model(row.experiment_name), weight="bold")
        ax.set_xticks([0, 1], ["Gerçek", "Sahte"])
        ax.set_yticks([0, 1], ["Gerçek", "Sahte"])
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek sınıf")
    fig.suptitle("Deney 4 — Görüntü Düzeyi Hata Matrisleri", weight="bold", fontsize=15)
    fig.subplots_adjust(top=0.91)
    save(fig, "experiment4_confusion_matrices.png")


def _bin_lower_bound(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else float("inf")


def experiment4_mask_quartiles() -> None:
    df = pd.read_csv(SOURCE_DIR / "experiment4_mask_quartiles.csv")
    bins = sorted(df["gt_area_bin"].unique(), key=_bin_lower_bound)
    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    models = [
        "efficientnetb0_unet_rgb_full",
        "unetpp_resnet34_rgb_full",
        "segformer_b0_rgb_full",
        "dinov2_lite_decoder_rgb_full",
    ]
    x = np.arange(len(bins))
    width = 0.19
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b2"]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for index, (model, color) in enumerate(zip(models, colors)):
        values = (
            df[df["experiment_name"] == model]
            .set_index("gt_area_bin")["mean_dice"]
            .reindex(bins)
            .to_numpy()
        )
        ax.bar(x + (index - 1.5) * width, values, width, label=label_model(model), color=color)
    ax.set_title("Deney 4 — Maske Büyüklüğüne Göre Ortalama Dice", weight="bold")
    ax.set_ylabel("Ortalama Dice")
    ax.set_xlabel("Ground-truth maske büyüklüğü")
    ax.set_xticks(x, quartile_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    save(fig, "experiment4_mask_quartiles.png")


def experiment6_resolution_comparison() -> None:
    df = pd.read_csv(SOURCE_DIR / "experiment6_vs_experiment5.csv")
    df = df[
        (df["exp6_strategy"] == "balanced_final_score")
        & (df["exp5_strategy"] == "balanced_final_score")
    ].drop_duplicates("model")
    metrics = ["q1_dice", "q2_dice", "q3_dice", "q4_dice"]
    labels = ["Q1", "Q2", "Q3", "Q4"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, row in zip(axes, df.itertuples(index=False)):
        exp5 = [getattr(row, f"exp5_{metric}") for metric in metrics]
        exp6 = [getattr(row, f"exp6_{metric}") for metric in metrics]
        x = np.arange(4)
        width = 0.36
        ax.bar(x - width / 2, exp5, width, label="Deney 5 · 256×256", color="#9aa8b5")
        ax.bar(x + width / 2, exp6, width, label="Deney 6 · 384×384", color="#4c72b0")
        ax.set_title(row.model, weight="bold")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Maske büyüklüğü çeyreği")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Ortalama Dice")
    axes[1].legend(frameon=False)
    fig.suptitle("Deney 5–6 — Çözünürlük Artışının Q1–Q4 Performansına Etkisi", weight="bold")
    fig.subplots_adjust(top=0.84)
    save(fig, "experiment6_resolution_comparison.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    experiment1_model_screening()
    experiment2_seed_stability()
    experiment4_confusion_matrices()
    experiment4_mask_quartiles()
    experiment6_resolution_comparison()


if __name__ == "__main__":
    main()
