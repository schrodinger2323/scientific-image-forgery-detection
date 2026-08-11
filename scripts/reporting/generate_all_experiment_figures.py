from __future__ import annotations

import math
import shutil
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_review" / "generated_figures"
FORMATS = ("png", "pdf", "svg")
DPI = 350

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


MODEL_SHORT = {
    "segformer_b0": "SegFormer-B0",
    "segformer-b0": "SegFormer-B0",
    "efficientnetb0_unet": "EffNetB0-UNet",
    "efficientnetb0-unet": "EffNetB0-UNet",
    "effnetb0": "EffNetB0-UNet",
    "unetpp": "UNet++",
    "unetplusplus": "UNet++",
    "unet++": "UNet++",
    "dinov2": "DINOv2-lite",
    "deeplabv3plus": "DeepLabV3+",
    "plain_unet": "Plain UNet",
    "resnet50_unet": "ResNet50-UNet",
    "mantranet": "ManTraNet",
    "mvssnetpp": "MVSSNet++",
    "mvssnet": "MVSSNet",
    "selfcorr_cmfd": "SelfCorr-CMFD",
    "siamese_cmfd": "Siamese-CMFD",
    "cmfdformer": "CMFDFormer",
}


def info(message: str) -> None:
    print(f"[figures] {message}")


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        info(f"skip missing: {path.relative_to(ROOT)}")
        return None
    return pd.read_csv(path)


def short_model(name: object) -> str:
    text = str(name)
    low = text.lower()
    for key, value in MODEL_SHORT.items():
        if key in low:
            return value
    return text.replace("_rgb_full", "").replace("_", " ")


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, stem: str, formats: tuple[str, ...] = FORMATS) -> None:
    ensure_out()
    for ext in formats:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def metric_from_summary(path: Path, metric_names: list[str]) -> float | None:
    df = read_csv(path)
    if df is None or "metric" not in df.columns or "mean" not in df.columns:
        return None
    metric_lookup = {str(v).lower(): i for i, v in enumerate(df["metric"])}
    for metric in metric_names:
        idx = metric_lookup.get(metric.lower())
        if idx is not None:
            return float(df.iloc[idx]["mean"])
    return None


def summary_row(path: Path, metrics: dict[str, list[str]]) -> dict[str, float] | None:
    df = read_csv(path)
    if df is None or "metric" not in df.columns or "mean" not in df.columns:
        return None
    metric_lookup = {str(v).lower(): i for i, v in enumerate(df["metric"])}
    out: dict[str, float] = {}
    for out_name, candidates in metrics.items():
        for metric in candidates:
            idx = metric_lookup.get(metric.lower())
            if idx is not None:
                out[out_name] = float(df.iloc[idx]["mean"])
                break
    return out if out else None


def grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    metrics: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    stem: str,
    width: float = 0.78,
) -> None:
    plot_df = df.copy()
    x = np.arange(len(plot_df))
    n = len(metrics)
    bar_w = width / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(7, len(plot_df) * 1.25), 4.8))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x - width / 2 + bar_w / 2 + i * bar_w, plot_df[metric], width=bar_w, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[x_col], rotation=20, ha="right")
    ax.set_ylim(bottom=0)
    ax.legend(ncols=min(n, 3))
    save_fig(fig, stem)


def experiment1_figures() -> None:
    rows = []
    excluded = {"deney_2", "deney_4", "deney_5", "deney_6", "experiments", "final_analysis", "analysis_review"}
    for model_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in excluded):
        path = model_dir / "test_metrics_summary.csv"
        if not path.exists():
            continue
        metrics = summary_row(
            path,
            {
                "forged_f1": ["forged_f1", "pixel_f1", "pixel_dice"],
                "precision": ["forged_precision", "pixel_precision"],
                "recall": ["forged_recall", "pixel_recall"],
            },
        )
        if metrics and {"forged_f1", "precision", "recall"} <= set(metrics):
            rows.append({"model": short_model(model_dir.name), **metrics})
    df = pd.DataFrame(rows)
    if df.empty:
        info("Deney 1 skipped: no root-level test_metrics_summary.csv files found")
        return
    df = df.sort_values("forged_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(df))))
    ax.barh(df["model"], df["forged_f1"], color="#4C78A8")
    ax.set_xlabel("Forged F1")
    ax.set_title("Deney 1 - Forged F1 siralamasi")
    ax.set_xlim(0, max(1.0, df["forged_f1"].max() * 1.08))
    save_fig(fig, "deney1_forged_f1_horizontal_bar")

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 80 + 260 * df["forged_f1"].clip(0, 1)
    ax.scatter(df["precision"], df["recall"], s=sizes, alpha=0.78, color="#F58518", edgecolor="white", linewidth=0.8)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["precision"], row["recall"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Deney 1 - Precision/Recall scatter")
    ax.set_xlim(0, min(1.02, max(1.0, df["precision"].max() * 1.1)))
    ax.set_ylim(0, min(1.02, max(1.0, df["recall"].max() * 1.1)))
    save_fig(fig, "deney1_precision_recall_scatter")


def experiment2_figures() -> None:
    rows = []
    for seed_dir in sorted(p for p in (ROOT / "deney_2").glob("pilot_seed_*") if p.is_dir()):
        seed = seed_dir.name.replace("pilot_seed_", "")
        for model_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            if model_dir.name.startswith("_shared"):
                continue
            path = model_dir / "test_metrics_summary.csv"
            f1 = metric_from_summary(path, ["forged_f1", "pixel_f1", "pixel_dice"])
            if f1 is not None:
                rows.append({"seed": seed, "model": short_model(model_dir.name), "forged_f1": f1})
    df = pd.DataFrame(rows)
    if df.empty:
        info("Deney 2 skipped: no seed-level summaries found")
        return

    agg = (
        df.groupby("model", as_index=False)
        .agg(mean_forged_f1=("forged_f1", "mean"), std=("forged_f1", "std"), n=("forged_f1", "count"))
        .sort_values("mean_forged_f1", ascending=False)
    )
    agg["se"] = agg["std"].fillna(0) / np.sqrt(agg["n"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(agg["model"], agg["mean_forged_f1"], yerr=agg["se"], fmt="o", capsize=5, color="#4C78A8")
    ax.set_ylabel("Mean Forged F1 +/- SE")
    ax.set_title("Deney 2 - Seedler arasi model ortalamasi")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "deney2_mean_forged_f1_errorbar")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model, part in df.sort_values("seed").groupby("model"):
        ax.plot(part["seed"], part["forged_f1"], marker="o", linewidth=1.8, label=model)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Forged F1")
    ax.set_title("Deney 2 - Seed bazli line/dot plot")
    ax.set_ylim(bottom=0)
    ax.legend(ncols=2)
    save_fig(fig, "deney2_seed_line_dot")


def experiment3_figures() -> None:
    path = ROOT / "experiments" / "experiment_comparison.csv"
    df = read_csv(path)
    if df is None or df.empty:
        return
    df["model"] = df["experiment_name"].map(
        {
            "unetpp_resnet34_rgb_baseline": "UNet++ RGB",
            "unetpp_resnet34_rgb_srm_edge_multitask": "UNet++ R34 SRM+Edge",
            "unetpp_resnet50_rgb_srm_edge_multitask": "UNet++ R50 SRM+Edge",
        }
    ).fillna(df["experiment_name"].map(short_model))
    grouped_bar(
        df,
        "model",
        ["test_dice", "test_iou", "test_auprc", "image_level_f1", "boundary_f1"],
        ["Dice", "IoU", "AUPRC", "Image F1", "Boundary F1"],
        "Deney 3 - U-Net++ konfigrasyon karsilastirmasi",
        "Score",
        "deney3_unetpp_grouped_metrics",
    )

    rows = []
    for model_dir in sorted((ROOT / "experiments").glob("unetpp_*multitask")):
        tm = read_csv(model_dir / "test_metrics.csv")
        if tm is None or tm.empty or "inference_mode" not in tm.columns:
            continue
        for _, row in tm.iterrows():
            if row["inference_mode"] in {"surface", "edge_enhanced"}:
                rows.append(
                    {
                        "model": "R34" if "resnet34" in model_dir.name else "R50",
                        "mode": row["inference_mode"],
                        "Dice": row["dice"],
                        "IoU": row["iou"],
                        "Boundary F1": row["boundary_f1"],
                    }
                )
    slope = pd.DataFrame(rows)
    if slope.empty:
        info("Deney 3 slope skipped: no surface/edge_enhanced test_metrics rows found")
        return
    metrics = ["Dice", "IoU", "Boundary F1"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, metric in zip(axes, metrics):
        for model, part in slope.groupby("model"):
            part = part.set_index("mode").loc[["surface", "edge_enhanced"]].reset_index()
            ax.plot(part["mode"], part[metric], marker="o", linewidth=2, label=model)
        ax.set_title(metric)
        ax.set_ylabel("Score")
        ax.set_ylim(bottom=0)
    axes[0].legend(title="Backbone")
    fig.suptitle("Deney 3 - Surface vs edge-enhanced")
    save_fig(fig, "deney3_surface_vs_edge_slope")


def experiment4_figures() -> None:
    df = read_csv(ROOT / "analysis_review" / "experiment_4_model_comparison_key_metrics.csv")
    if df is None:
        df = read_csv(ROOT / "deney_4" / "experiments_full" / "model_comparison_full.csv")
    if df is None or df.empty:
        return
    df["model"] = df["experiment_name"].map(short_model)
    grouped_bar(
        df,
        "model",
        ["test_dice_forged_only", "test_iou_forged_only", "component_f1", "image_f1"],
        ["Forged Dice", "IoU", "Component F1", "Image F1"],
        "Deney 4 - Ana metrikler",
        "Score",
        "deney4_grouped_main_metrics",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = df.sort_values("authentic_fp_rate", ascending=True)
    ax.bar(order["model"], order["authentic_fp_rate"], color="#E45756")
    ax.set_ylabel("Authentic FP Rate")
    ax.set_title("Deney 4 - Authentic FP Rate (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "deney4_authentic_fp_rate_lower_better")

    bins = read_csv(ROOT / "analysis_review" / "experiment_4_forged_area_bin_analysis.csv")
    if bins is None or bins.empty:
        return
    bins = bins.copy()
    bins["model"] = bins["experiment_name"].map(short_model)
    bin_order = {b: i + 1 for i, b in enumerate(sorted(bins["gt_area_bin"].unique(), key=str))}
    bins["quartile"] = bins["gt_area_bin"].map(bin_order).map(lambda i: f"Q{i}")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model, part in bins.sort_values("quartile").groupby("model"):
        ax.plot(part["quartile"], part["mean_dice"], marker="o", linewidth=2, label=model)
    ax.set_xlabel("Forged mask area quartile")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Deney 4 - Q1-Q4 Dice")
    ax.set_ylim(bottom=0)
    ax.legend(ncols=2)
    save_fig(fig, "deney4_q1_q4_dice_line")


def experiment5_figures() -> None:
    df = read_csv(ROOT / "deney_5" / "experiments_4_full" / "experiment5_calibration_postprocessing" / "test_results_all_strategies.csv")
    if df is None or df.empty:
        return
    df = df.copy()
    df["model"] = df["model_name"].map(short_model)

    before_after = df[df["strategy"].isin(["raw_reference", "balanced_final_score"])].copy()
    if not before_after.empty:
        before_after["stage"] = before_after["strategy"].map({"raw_reference": "Before raw", "balanced_final_score": "After PP"})
        for metric, label, stem in [
            ("dice_forged_only", "Forged Dice", "deney5_before_after_forged_dice"),
            ("authentic_fp_rate", "Authentic FP Rate", "deney5_before_after_auth_fp_lower_better"),
            ("component_f1_iou010", "Component F1@0.10", "deney5_before_after_component_f1"),
        ]:
            pivot = before_after.pivot_table(index="model", columns="stage", values=metric, aggfunc="first").reset_index()
            cols = [c for c in ["Before raw", "After PP"] if c in pivot.columns]
            grouped_bar(pivot, "model", cols, cols, f"Deney 5 - Before/after {label}", label, stem)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, y, title in [
        (axes[0], "dice_forged_only", "Auth FP vs Forged Dice"),
        (axes[1], "final_score", "Auth FP vs Final Score"),
    ]:
        if y not in df.columns:
            ax.axis("off")
            continue
        for model, part in df.groupby("model"):
            ax.scatter(part["authentic_fp_rate"], part[y], s=65, alpha=0.8, label=model)
        ax.set_xlabel("Authentic FP Rate (lower is better)")
        ax.set_ylabel(y)
        ax.set_title(title)
    axes[0].legend(ncols=2)
    fig.suptitle("Deney 5 - Strateji trade-off scatter")
    save_fig(fig, "deney5_strategy_tradeoff_scatter")


def experiment6_figures() -> None:
    path = ROOT / "analysis_review" / "experiment_6_vs_experiment_5_reconstructed_comparison.csv"
    df = read_csv(path)
    if df is None:
        df = read_csv(ROOT / "deney_6" / "experiments_full_eski_2" / "experiment6_smallmask_384" / "experiment6_vs_experiment5_comparison.csv")
    if df is None or df.empty:
        return
    df = df.copy()
    model_col = "model" if "model" in df.columns else "experiment6_model"
    df["model_short"] = df[model_col].map(short_model)
    if "exp6_strategy" in df.columns:
        plot_df = df[df["exp6_strategy"].eq("balanced_final_score")].copy()
        if plot_df.empty:
            plot_df = df.copy()
    else:
        plot_df = df.copy()

    delta_candidates = [
        ("delta_forged_dice", "Forged Dice"),
        ("delta_component_f1_iou010", "Component F1"),
        ("delta_authentic_fp_rate", "Auth FP"),
        ("delta_image_f1", "Image F1"),
        ("delta_dice_lt_005_count", "Dice<0.05 count"),
        ("delta_q1_dice", "Q1 Dice"),
        ("delta_q2_dice", "Q2 Dice"),
    ]
    metrics = [c for c, _ in delta_candidates if c in plot_df.columns]
    labels = [lab for c, lab in delta_candidates if c in plot_df.columns]
    if metrics:
        grouped_bar(
            plot_df,
            "model_short",
            metrics,
            labels,
            "Deney 6 - Deney 5'e gore delta",
            "Delta",
            "deney6_delta_grouped_bar",
        )

    panels = [
        ("delta_q1_dice", "Q1 Dice"),
        ("delta_q2_dice", "Q2 Dice"),
        ("delta_component_f1_iou010", "Component F1"),
        ("delta_authentic_fp_rate", "Auth FP"),
        ("delta_dice_lt_005_count", "Dice<0.05 count"),
    ]
    available = [(c, t) for c, t in panels if c in plot_df.columns]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4.4), sharex=False)
    if len(available) == 1:
        axes = [axes]
    colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#E45756", "#E45756"]
    for ax, (col, title), color in zip(axes, available, colors):
        ax.axhline(0, color="black", linewidth=0.8)
        ax.bar(plot_df["model_short"], plot_df[col], color=color)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Delta")
    fig.suptitle("Deney 6 - Ayrik delta panelleri")
    save_fig(fig, "deney6_delta_panels")


def final_analysis_figures() -> None:
    final = read_csv(ROOT / "final_analysis" / "final_analysis" / "tables" / "final_model_comparison.csv")
    if final is None or final.empty:
        return
    final = final[final["role"].isin(["best_localization_model", "low_false_alarm_model"])].copy()
    if final.empty:
        final = final.head(2).copy()
    final["model"] = final["model_name"].map(short_model)
    grouped_bar(
        final,
        "model",
        ["forged_dice", "forged_iou", "q1_dice", "q2_dice", "component_f1_iou010", "image_f1"],
        ["Forged Dice", "IoU", "Q1 Dice", "Q2 Dice", "Component F1", "Image F1"],
        "Final analiz - Scorecard",
        "Score",
        "final_scorecard_grouped_bar",
    )
    grouped_bar(
        final,
        "model",
        ["authentic_fp_rate", "brier_score", "ece_10_bins"],
        ["Auth FP", "Brier", "ECE"],
        "Final analiz - Lower-is-better metrikler",
        "Value",
        "final_scorecard_lower_better",
    )

    rob = read_csv(ROOT / "final_analysis" / "final_analysis" / "robustness_metrics_all.csv")
    if rob is not None and not rob.empty:
        rob = rob.copy()
        rob["model"] = rob["model_name"].map(short_model)
        order = ["clean_png", "jpeg_q90", "jpeg_q70", "jpeg_q50", "gaussian_noise_light", "gaussian_noise_medium", "gaussian_blur_light", "gaussian_blur_medium", "combined_jpeg70_blur_light"]
        rob["degradation"] = pd.Categorical(rob["degradation"], categories=[x for x in order if x in set(rob["degradation"])], ordered=True)
        rob = rob.sort_values(["model", "degradation"])
        for metrics, labels, stem, title in [
            (
                ["forged_dice", "q1_dice", "component_f1_iou010", "image_f1"],
                ["Forged Dice", "Q1 Dice", "Component F1", "Image F1"],
                "final_robustness_good_metrics_line",
                "Final analiz - Robustness metrikleri",
            ),
            (
                ["authentic_fp_rate", "dice_lt_005_count"],
                ["Auth FP", "Dice<0.05 count"],
                "final_robustness_lower_better_line",
                "Final analiz - Robustness lower-is-better",
            ),
        ]:
            fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.3), sharex=True)
            if len(metrics) == 1:
                axes = [axes]
            for ax, metric, label in zip(axes, metrics, labels):
                for model, part in rob.groupby("model"):
                    ax.plot(part["degradation"].astype(str), part[metric], marker="o", linewidth=1.8, label=model)
                ax.set_title(label)
                ax.tick_params(axis="x", rotation=45)
            axes[0].legend()
            fig.suptitle(title)
            save_fig(fig, stem)

    make_failure_case_contact_sheet()


def make_failure_case_contact_sheet() -> None:
    pngs: list[Path] = []
    for model_dir in [
        ROOT / "final_analysis" / "final_analysis" / "segformer_b0_rgb_384_smallmask" / "failure_cases",
        ROOT / "final_analysis" / "final_analysis" / "efficientnetb0_unet_rgb_384_smallmask" / "failure_cases",
    ]:
        pngs.extend(sorted(model_dir.glob("*.png")))
    if not pngs:
        info("Final failure grid skipped: no existing failure-case PNG grids found")
        return

    selected = [p for p in pngs if p.stem in {"worst_forged_cases", "small_mask_failures", "false_positive_authentic", "false_negative_forged"}]
    if not selected:
        selected = pngs[:8]
    n = len(selected)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.8 * rows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes_arr, selected):
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        parent = short_model(path.parents[1].name)
        ax.set_title(f"{parent}\n{path.stem}", fontsize=9)
        ax.axis("off")
    for ax in axes_arr[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Final analiz - Failure case grid")
    save_fig(fig, "final_failure_case_grid")

    # Also keep a direct copy of the existing detailed grids.
    copy_dir = OUT / "failure_case_source_grids"
    copy_dir.mkdir(parents=True, exist_ok=True)
    for path in selected:
        shutil.copy2(path, copy_dir / f"{short_model(path.parents[1].name)}_{path.name}")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_out()
    experiment1_figures()
    experiment2_figures()
    experiment3_figures()
    experiment4_figures()
    experiment5_figures()
    experiment6_figures()
    final_analysis_figures()
    info(f"done: {OUT}")


if __name__ == "__main__":
    main()
