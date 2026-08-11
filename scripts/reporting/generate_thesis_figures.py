from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_review" / "thesis_figures_rebuilt"
OUT.mkdir(parents=True, exist_ok=True)

PNG_DPI = 320
PALETTE = [
    "#2F5D8C",
    "#C75050",
    "#4A8A62",
    "#D59B2D",
    "#6F5FA8",
    "#5B8791",
    "#A85F2F",
    "#7A7A7A",
]

MODEL_LABELS = {
    "segformer_b0": "SegFormer-B0",
    "segformer_b0_rgb_full": "SegFormer-B0",
    "segformer_b0_rgb_384_smallmask": "SegFormer-B0 384",
    "deeplabv3plus": "DeepLabV3+",
    "efficientnetb0_unet": "EfficientNetB0-UNet",
    "efficientnetb0_unet_rgb_full": "EfficientNetB0-UNet",
    "efficientnetb0_unet_rgb_384_smallmask": "EfficientNetB0-UNet 384",
    "unetplusplus": "U-Net++ ResNet34",
    "unetpp_resnet34_rgb_full": "U-Net++ ResNet34",
    "unetpp_resnet34_rgb_baseline": "U-Net++ ResNet34 RGB",
    "unetpp_resnet34_rgb_srm_edge_multitask": "U-Net++ ResNet34 RGB+SRM",
    "unetpp_resnet50_rgb_srm_edge_multitask": "U-Net++ ResNet50 RGB+SRM",
    "dinov2_lite_decoder_rgb_full": "DINOv2-lite",
    "plain_unet": "Plain U-Net",
    "cmfdformer": "CMFDFormer",
    "selfcorr_cmfd": "SelfCorr-CMFD",
    "qdl_cmfd": "QDL-CMFD",
    "mvssnet": "MVSS-Net",
    "mvssnetpp": "MVSS-Net++",
    "mantranet": "ManTraNet",
    "busternet": "BusterNet",
    "doagan": "DOA-GAN",
    "dinov2_seg": "DINOv2-Seg",
    "resnet50_unet": "ResNet50-UNet",
    "siamese_cmfd": "Siamese-CMFD",
}

STRATEGY_LABELS = {
    "raw_reference": "Ham çıktı",
    "best_forged_dice": "En iyi forged Dice",
    "best_component_f1": "En iyi Component F1",
    "balanced_final_score": "Dengeli final skor",
    "low_false_alarm": "Düşük yanlış alarm",
    "small_object_focused": "Küçük maske odaklı",
    "best_small_mask_q1_dice": "En iyi Q1 Dice",
    "small_object_practical": "Pratik küçük maske",
}


manifest: list[dict[str, str]] = []


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: str | Path) -> pd.DataFrame:
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not path.exists():
        raise FileNotFoundError(rel(path))
    return pd.read_csv(path)


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def label_model(name: str) -> str:
    if pd.isna(name):
        return ""
    key = str(name)
    return MODEL_LABELS.get(key, key.replace("_", " ").title())


def label_strategy(name: str) -> str:
    if pd.isna(name):
        return ""
    key = str(name)
    return STRATEGY_LABELS.get(key, key.replace("_", " ").title())


def setup_ax(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.grid(axis="y", color="#D8DCE2", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAB0B8")
    ax.spines["bottom"].set_color("#AAB0B8")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.tick_params(labelsize=9)


def add_title(fig: plt.Figure, title: str) -> None:
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}", horizontal: bool = False) -> None:
    for bar in bars:
        val = bar.get_width() if horizontal else bar.get_height()
        if not np.isfinite(val):
            continue
        if horizontal:
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, fmt.format(val),
                    va="center", ha="left", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, fmt.format(val),
                    va="bottom", ha="center", fontsize=8, rotation=0)


def save_record(
    fig_id: str,
    fig: plt.Figure,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    caption: str,
    comment: str,
    source: str,
    code_ref: str,
) -> None:
    png = OUT / f"{filename}.png"
    svg = OUT / f"{filename}.svg"
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append(
        {
            "id": fig_id,
            "status": "created",
            "png": rel(png),
            "svg": rel(svg),
            "title": title,
            "x_axis": xlabel,
            "y_axis": ylabel,
            "caption": caption,
            "comment": comment,
            "source": source,
            "code": code_ref,
        }
    )


def skip_record(fig_id: str, title: str, reason: str, source: str, code_ref: str) -> None:
    manifest.append(
        {
            "id": fig_id,
            "status": "skipped",
            "png": "",
            "svg": "",
            "title": title,
            "x_axis": "",
            "y_axis": "",
            "caption": reason,
            "comment": "",
            "source": source,
            "code": code_ref,
        }
    )


def safe_plot(func):
    def wrapper():
        try:
            func()
        except Exception as exc:
            skip_record(func.__name__, func.__doc__ or func.__name__, f"Üretilemedi: {exc}", "", func.__name__)

    return wrapper


def grouped_bar(
    df: pd.DataFrame,
    category: str,
    metrics: list[tuple[str, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    fig_id: str,
    filename: str,
    caption: str,
    comment: str,
    source: str,
    code_ref: str,
    figsize: tuple[float, float] = (11, 6),
    rotate: int = 25,
) -> None:
    data = numeric(df, [m[0] for m in metrics]).dropna(subset=[category])
    x = np.arange(len(data))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=figsize)
    for i, (col, lab) in enumerate(metrics):
        bars = ax.bar(x + (i - (len(metrics) - 1) / 2) * width, data[col], width,
                      label=lab, color=PALETTE[i % len(PALETTE)])
        if len(data) <= 6:
            add_bar_labels(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(data[category], rotation=rotate, ha="right")
    setup_ax(ax, xlabel, ylabel)
    ax.legend(frameon=False, fontsize=9, ncol=min(3, len(metrics)))
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record(fig_id, fig, filename, title, xlabel, ylabel, caption, comment, source, code_ref)


@safe_plot
def plot_01_method_flow():
    """Deneysel Sürecin Genel Akışı: Geniş Model Taramasından Final Model Analizine"""
    title = "Deneysel Sürecin Genel Akışı: Geniş Model Taramasından Final Model Analizine"
    steps = [
        ("Deney 1", "15 model ile\npilot tarama"),
        ("Deney 2", "Seed kararlılık\nanalizi"),
        ("Deney 3", "U-Net++ varyant\nkarşılaştırması"),
        ("Deney 4", "4 mimari ailesi\nkarşılaştırması"),
        ("Deney 5", "Post-processing ve\nkalibrasyon"),
        ("Deney 6", "384×384 küçük\nmaske iyileştirmesi"),
        ("Final Analiz", "İki final modelin\nkapsamlı testi"),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, ((head, body), x) in enumerate(zip(steps, xs)):
        width = 0.115 if i < 6 else 0.14
        rect = plt.Rectangle((x - width / 2, 0.43), width, 0.27, facecolor=PALETTE[i % len(PALETTE)],
                             edgecolor="none", alpha=0.95)
        ax.add_patch(rect)
        ax.text(x, 0.61, head, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        ax.text(x, 0.49, body, ha="center", va="center", fontsize=8.7, color="white")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.065, 0.565), xytext=(x + 0.065, 0.565),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="#444A54"))
    add_title(fig, title)
    save_record(
        "01",
        fig,
        "01_deneysel_surec_genel_akis",
        title,
        "Deney aşamaları",
        "Araştırma kapsamı",
        "Bu akış diyagramı, geniş model taramasından final model değerlendirmesine kadar izlenen deneysel hattı özetler.",
        "Deneyler birbirini eleme, kararlılık, mimari karşılaştırma ve karar-katmanı optimizasyonu mantığıyla takip ediyor.",
        "Kullanıcı tanımlı deney listesi ve repo rapor yapısı",
        "generate_thesis_figures.py::plot_01_method_flow",
    )


@safe_plot
def plot_02_split_distribution():
    """Veri Seti Bölmeleri ve Sınıf Dağılımı"""
    title = "Veri Seti Bölmeleri ve Sınıf Dağılımı"
    candidates = [
        "final_analysis/final_analysis/split_summary.csv",
        "deney_6/experiments_full/experiment6_smallmask_384/split_summary.csv",
        "deney_4/_shared_splits_seed42/split_summary.csv",
    ]
    path = next((p for p in candidates if (ROOT / p).exists()), None)
    if path is None:
        raise FileNotFoundError("split_summary.csv bulunamadı")
    df = numeric(read_csv(path), ["authentic", "forged"])
    df = df[df["split"].isin(["train", "val", "test"])].copy()
    df["split_label"] = df["split"].map({"train": "Eğitim", "val": "Doğrulama", "test": "Test"}).fillna(df["split"])
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(df["split_label"], df["authentic"], label="Authentic", color=PALETTE[0])
    bars2 = ax.bar(df["split_label"], df["forged"], bottom=df["authentic"], label="Forged", color=PALETTE[1])
    for a, f, x in zip(df["authentic"], df["forged"], df["split_label"]):
        ax.text(x, a / 2, f"{int(a)}", ha="center", va="center", color="white", fontsize=9)
        ax.text(x, a + f / 2, f"{int(f)}", ha="center", va="center", color="white", fontsize=9)
    setup_ax(ax, "Veri bölmesi", "Görüntü sayısı")
    ax.legend(frameon=False)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("02", fig, "02_veri_seti_bolmeleri_sinif_dagilimi", title, "Veri bölmesi", "Görüntü sayısı",
                "Bu stacked bar grafik eğitim, doğrulama ve test bölmelerindeki authentic/forged dağılımını gösterir.",
                "Sınıf dengesinin bölmeler arasında nasıl korunduğu hızlıca görülebilir.", path,
                "generate_thesis_figures.py::plot_02_split_distribution")


@safe_plot
def plot_03_mask_quartiles():
    """Test Setinde Sahtecilik Bölgesi Boyutlarının Q1-Q4 Gruplarına Göre Dağılımı"""
    title = "Test Setinde Sahtecilik Bölgesi Boyutlarının Q1-Q4 Gruplarına Göre Dağılımı"
    paths = [
        "deney_6/experiments_full/experiment6_smallmask_384/segformer_b0_rgb_384_smallmask/small_mask_bin_metrics_balanced_final_score.csv",
        "final_analysis/final_analysis/segformer_b0_rgb_384_smallmask/small_mask_bin_metrics_balanced_final_score.csv",
    ]
    path = next((p for p in paths if (ROOT / p).exists() and "mean_gt_area_ratio" in pd.read_csv(ROOT / p, nrows=1).columns), None)
    if path is None:
        raise FileNotFoundError("small_mask_bin_metrics CSV bulunamadı")
    df = numeric(read_csv(path), ["mean_gt_area_ratio"])
    qcol = "mask_quartile" if "mask_quartile" in df.columns else "area_bin"
    df["area_pct"] = df["mean_gt_area_ratio"] * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df[qcol], df["area_pct"], color=PALETTE[2])
    add_bar_labels(ax, bars, "{:.2f}%")
    setup_ax(ax, "Q1-Q4 maske boyutu grupları", "Ortalama GT alan oranı (%)")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("03", fig, "03_q1_q4_maske_boyutu_dagilimi", title,
                "Q1-Q4 maske boyutu grupları", "Ortalama GT alan oranı (%)",
                "Bu grafik forged test örneklerini ground-truth sahtecilik alanı büyüklüğüne göre dört grupta özetler.",
                "Q1 en küçük maske grubunu temsil ettiği için küçük nesne başarımı yorumlarında ayrı izlenmelidir.",
                path, "generate_thesis_figures.py::plot_03_mask_quartiles")


@safe_plot
def plot_04_exp1_forged_f1():
    """Deney 1 – Modellerin Forged Pixel F1 Performans Karşılaştırması"""
    title = "Deney 1 – Modellerin Forged Pixel F1 Performans Karşılaştırması"
    path = "analysis_review/all_small_subset_results_review.csv"
    df = numeric(read_csv(path), ["pixel_f1"]).sort_values("pixel_f1", ascending=True)
    df["model"] = df["model"].map(label_model)
    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(df["model"], df["pixel_f1"], color=PALETTE[0])
    add_bar_labels(ax, bars, "{:.3f}", horizontal=True)
    setup_ax(ax, "Forged Pixel F1", "Model")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_record("04", fig, "04_deney1_forged_pixel_f1_bar", title, "Forged Pixel F1", "Model",
                "Bu yatay bar grafik Deney 1'deki 15 modelin forged piksel F1 performansını azalan başarı düzeninde karşılaştırır.",
                "En üstteki modeller daha yüksek forged piksel örtüşümü üretmiştir.", path,
                "generate_thesis_figures.py::plot_04_exp1_forged_f1")


@safe_plot
def plot_05_exp1_iou():
    """Deney 1 – Modellerin Forged IoU Değerlerinin Karşılaştırılması"""
    title = "Deney 1 – Modellerin Forged IoU Değerlerinin Karşılaştırılması"
    path = "analysis_review/all_small_subset_results_review.csv"
    df = numeric(read_csv(path), ["pixel_iou"]).sort_values("pixel_iou", ascending=True)
    df["model"] = df["model"].map(label_model)
    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(df["model"], df["pixel_iou"], color=PALETTE[1])
    add_bar_labels(ax, bars, "{:.3f}", horizontal=True)
    setup_ax(ax, "Forged IoU", "Model")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_record("05", fig, "05_deney1_forged_iou_bar", title, "Forged IoU", "Model",
                "Bu grafik Deney 1 modellerinin forged sınıfı için Intersection over Union değerlerini gösterir.",
                "IoU, tahmin ve gerçek sahtecilik maskesi kesişiminin birleşime oranıdır.", path,
                "generate_thesis_figures.py::plot_05_exp1_iou")


@safe_plot
def plot_06_exp1_precision_recall():
    """Deney 1 – Modellerin Precision-Recall Davranışı"""
    title = "Deney 1 – Modellerin Precision-Recall Davranışı"
    path = "analysis_review/all_small_subset_results_review.csv"
    df = numeric(read_csv(path), ["pixel_precision", "pixel_recall", "pixel_f1"])
    df["model"] = df["model"].map(label_model)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df["pixel_recall"], df["pixel_precision"], s=80 + 450 * df["pixel_f1"].fillna(0),
                    c=np.arange(len(df)), cmap="tab20", edgecolor="#333333", linewidth=0.6)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["pixel_recall"], row["pixel_precision"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    setup_ax(ax, "Forged Recall", "Forged Precision")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("06", fig, "06_deney1_precision_recall_scatter", title, "Forged Recall", "Forged Precision",
                "Her nokta bir Deney 1 modelidir; x ekseni forged recall, y ekseni forged precision değerini verir.",
                "Grafik yüksek recall üretirken precision kaybeden modelleri görünür kılar.", path,
                "generate_thesis_figures.py::plot_06_exp1_precision_recall")


@safe_plot
def plot_07_exp1_image_level():
    """Deney 1 – Görüntü Düzeyinde Sahte/Gerçek Ayrım Başarısı"""
    path = "analysis_review/all_small_subset_results_review.csv"
    df = read_csv(path)
    df["model"] = df["model"].map(label_model)
    df = df.sort_values("image_f1", ascending=False)
    grouped_bar(
        df,
        "model",
        [("image_f1", "Image F1"), ("image_roc_auc", "Image ROC-AUC")],
        "Deney 1 – Görüntü Düzeyinde Sahte/Gerçek Ayrım Başarısı",
        "Model",
        "Skor",
        "07",
        "07_deney1_image_f1_rocauc_grouped",
        "Bu grouped bar grafik Deney 1 modellerinin görüntü düzeyinde sahte/gerçek ayrım gücünü gösterir.",
        "Image ROC-AUC eşik bağımsız ayrım gücünü, Image F1 seçilmiş eşikteki sınıflandırma başarısını özetler.",
        path,
        "generate_thesis_figures.py::plot_07_exp1_image_level",
        figsize=(11, 6.5),
        rotate=35,
    )


def plot_training_curve(fig_id: str, folder: str, title: str, filename: str) -> None:
    path = Path(folder) / "training_log.csv"
    df = read_csv(path)
    df = numeric(df, list(df.columns))
    epoch = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epoch, df["loss"], label="Train loss", color=PALETTE[0], lw=2)
    if "val_loss" in df.columns:
        axes[0].plot(epoch, df["val_loss"], label="Val loss", color=PALETTE[1], lw=2)
    setup_ax(axes[0], "Epoch", "Loss")
    axes[0].legend(frameon=False)
    dice_col = "dice_coefficient" if "dice_coefficient" in df.columns else None
    val_dice_col = "val_dice_coefficient" if "val_dice_coefficient" in df.columns else None
    iou_col = "iou_coefficient" if "iou_coefficient" in df.columns else None
    val_iou_col = "val_iou_coefficient" if "val_iou_coefficient" in df.columns else None
    if dice_col:
        axes[1].plot(epoch, df[dice_col], label="Train Dice", color=PALETTE[2], lw=2)
    if val_dice_col:
        axes[1].plot(epoch, df[val_dice_col], label="Val Dice", color=PALETTE[3], lw=2)
    if iou_col:
        axes[1].plot(epoch, df[iou_col], label="Train IoU", color=PALETTE[4], lw=1.5, ls="--")
    if val_iou_col:
        axes[1].plot(epoch, df[val_iou_col], label="Val IoU", color=PALETTE[5], lw=1.5, ls="--")
    setup_ax(axes[1], "Epoch", "Dice / IoU")
    axes[1].legend(frameon=False)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save_record(fig_id, fig, filename, title, "Epoch", "Loss ve Dice/IoU",
                f"Bu eğitim eğrisi {title.split('–')[-1].strip()} için train/validation loss ve varsa Dice-IoU gelişimini gösterir.",
                "Eğitim ve doğrulama çizgileri arasındaki ayrışma olası overfitting veya optimizasyon davranışını okumayı sağlar.",
                rel(ROOT / path), "generate_thesis_figures.py::plot_training_curve")


@safe_plot
def plot_08a_segformer_training():
    """Deney 1 – SegFormer-B0 Eğitim ve Doğrulama Eğrileri"""
    plot_training_curve("08a", "segformer_b0", "Deney 1 – SegFormer-B0 Eğitim ve Doğrulama Eğrileri",
                        "08a_deney1_segformer_b0_egitim_egrileri")


@safe_plot
def plot_08b_deeplab_training():
    """Deney 1 – DeepLabV3+ Eğitim Süreci"""
    plot_training_curve("08b", "deeplabv3plus", "Deney 1 – DeepLabV3+ Eğitim Süreci",
                        "08b_deney1_deeplabv3plus_egitim_egrileri")


@safe_plot
def plot_08c_effnet_training():
    """Deney 1 – EfficientNetB0-UNet Eğitim Eğrileri"""
    plot_training_curve("08c", "efficientnetb0_unet", "Deney 1 – EfficientNetB0-UNet Eğitim Eğrileri",
                        "08c_deney1_efficientnetb0_unet_egitim_egrileri")


def wrap_existing_image(fig_id: str, source_path: Path, title: str, filename: str, caption: str, comment: str) -> None:
    if not source_path.exists():
        raise FileNotFoundError(rel(source_path))
    img = Image.open(source_path).convert("RGB")
    fig_w = 12
    fig_h = max(5, fig_w * img.height / img.width + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img)
    ax.axis("off")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_record(fig_id, fig, filename, title, "Panel sütunları", "Görsel çıktı",
                caption, comment, rel(source_path), f"generate_thesis_figures.py::wrap_existing_image({rel(source_path)})")


@safe_plot
def plot_09_prediction_panel():
    """Deney 1 – Örnek Sahtecilik Lokalizasyon Çıktısı"""
    source = next((ROOT / "segformer_b0" / "prediction_examples").glob("*forged*.png"), None)
    if source is None:
        raise FileNotFoundError("segformer_b0/prediction_examples içinde forged panel bulunamadı")
    wrap_existing_image(
        "09",
        source,
        "Deney 1 – Örnek Sahtecilik Lokalizasyon Çıktısı",
        "09_deney1_ornek_lokalizasyon_paneli",
        "Bu panel giriş görüntüsü, ground truth, olasılık haritası ve ikili tahmin çıktısını birlikte gösterir.",
        "Model çıktısının yalnız skor değil, lokalizasyon kalitesi açısından da incelenmesini sağlar.",
    )


@safe_plot
def plot_10_seed_mean_std():
    """Deney 2 – Üç Farklı Tohum Altında Model Kararlılığı (Mean ± Std)"""
    title = "Deney 2 – Üç Farklı Tohum Altında Model Kararlılığı (Mean ± Std)"
    path = "deney_2/pilot_seed_comparison.csv"
    df = numeric(read_csv(path), ["forged_pixel_f1_mean", "forged_pixel_f1_std"]).sort_values("forged_pixel_f1_mean", ascending=False)
    df["model"] = df["model_name"].map(label_model)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["forged_pixel_f1_mean"], yerr=df["forged_pixel_f1_std"], capsize=5, color=PALETTE[0])
    add_bar_labels(ax, bars, "{:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right")
    setup_ax(ax, "Model", "Ortalama Forged Pixel F1")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("10", fig, "10_deney2_seed_stability_errorbar", title, "Model", "Ortalama Forged Pixel F1",
                "Bu grafik üç seed altında ölçülen forged pixel F1 ortalamasını ve standart sapmasını gösterir.",
                "Hata çubukları model performansının rastgele tohuma duyarlılığını özetler.", path,
                "generate_thesis_figures.py::plot_10_seed_mean_std")


@safe_plot
def plot_11_seed_lines():
    """Deney 2 – Rastgele Tohuma Göre Model Performans Değişimi"""
    title = "Deney 2 – Rastgele Tohuma Göre Model Performans Değişimi"
    rows = []
    for seed in [42, 123, 2025]:
        for path in (ROOT / "deney_2" / f"pilot_seed_{seed}").glob("*/test_metrics_summary.csv"):
            model = path.parent.name
            df = pd.read_csv(path)
            val = pd.to_numeric(df.loc[df["metric"].eq("pixel_f1"), "mean"], errors="coerce")
            if not val.empty:
                rows.append({"seed": seed, "model": label_model(model), "Forged Pixel F1": float(val.iloc[0])})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("seed bazlı test_metrics_summary dosyaları okunamadı")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (model, grp) in enumerate(df.groupby("model")):
        grp = grp.sort_values("seed")
        ax.plot(grp["seed"], grp["Forged Pixel F1"], marker="o", lw=2, label=model, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks([42, 123, 2025])
    setup_ax(ax, "Seed", "Forged Pixel F1")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("11", fig, "11_deney2_seed_bazli_degisım_line", title, "Seed", "Forged Pixel F1",
                "Bu çizgi grafik her modelin 42, 123 ve 2025 seed'leri altındaki forged pixel F1 değişimini gösterir.",
                "Çizginin dikliği modelin seed değişimine ne kadar duyarlı olduğunu gösterir.", "deney_2/pilot_seed_*/test_metrics_summary.csv",
                "generate_thesis_figures.py::plot_11_seed_lines")


@safe_plot
def plot_12_exp3_config_comparison():
    """Deney 3 – U-Net++ Konfigürasyonlarının Karşılaştırmalı Test Performansı"""
    path = "experiments/experiment_comparison.csv"
    df = read_csv(path)
    df["model"] = df["experiment_name"].map(label_model)
    grouped_bar(
        df,
        "model",
        [("test_dice", "Dice"), ("test_iou", "IoU"), ("test_auprc", "AUPRC"),
         ("image_level_f1", "Image F1"), ("boundary_f1", "Boundary-F1")],
        "Deney 3 – U-Net++ Konfigürasyonlarının Karşılaştırmalı Test Performansı",
        "U-Net++ konfigürasyonu",
        "Skor",
        "12",
        "12_deney3_unetpp_konfigurasyon_karsilastirma",
        "Bu grafik üç U-Net++ konfigürasyonunu Dice, IoU, AUPRC, Image F1 ve Boundary-F1 metrikleriyle karşılaştırır.",
        "RGB baseline örtüşüm metriklerinde, edge-aware varyantlar ise sınır duyarlılığı açısından yorumlanır.",
        path,
        "generate_thesis_figures.py::plot_12_exp3_config_comparison",
        figsize=(11, 6),
        rotate=20,
    )


@safe_plot
def plot_13_exp3_surface_edge():
    """Deney 3 – Surface ve Edge-Enhanced Çıkarım Modlarının Karşılaştırılması"""
    title = "Deney 3 – Surface ve Edge-Enhanced Çıkarım Modlarının Karşılaştırılması"
    rows = []
    for folder in ["unetpp_resnet34_rgb_srm_edge_multitask", "unetpp_resnet50_rgb_srm_edge_multitask"]:
        df = read_csv(Path("experiments") / folder / "test_metrics.csv")
        df["model"] = label_model(folder)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    metrics = [("dice", "Dice"), ("iou", "IoU"), ("boundary_f1", "Boundary-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=False)
    for ax, (col, lab) in zip(axes, metrics):
        for i, (model, grp) in enumerate(df.groupby("model")):
            vals = grp.set_index("inference_mode")[col]
            if {"surface", "edge_enhanced"}.issubset(vals.index):
                ax.plot(["Surface", "Edge-enhanced"], [vals["surface"], vals["edge_enhanced"]],
                        marker="o", lw=2, label=model, color=PALETTE[i])
        setup_ax(ax, "Çıkarım modu", lab)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save_record("13", fig, "13_deney3_surface_edge_slope", title, "Çıkarım modu", "Skor",
                "Bu slope plot, çok görevli U-Net++ modellerinde surface ve edge-enhanced çıkarım modlarını karşılaştırır.",
                "Edge-enhanced mod sınır metriğini artırırken Dice/IoU tarafında ödünleşim yaratabilir.", "experiments/*/test_metrics.csv",
                "generate_thesis_figures.py::plot_13_exp3_surface_edge")


@safe_plot
def plot_14_exp3_loss_curves():
    """Deney 3 – U-Net++ Modellerinin Eğitim/Doğrulama Kayıp Eğrileri"""
    title = "Deney 3 – U-Net++ Modellerinin Eğitim/Doğrulama Kayıp Eğrileri"
    folders = [
        "unetpp_resnet34_rgb_baseline",
        "unetpp_resnet34_rgb_srm_edge_multitask",
        "unetpp_resnet50_rgb_srm_edge_multitask",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, folder in zip(axes, folders):
        df = numeric(read_csv(Path("experiments") / folder / "metrics.csv"), ["epoch", "train_loss", "val_loss"])
        ax.plot(df["epoch"], df["train_loss"], label="Train loss", color=PALETTE[0], lw=2)
        ax.plot(df["epoch"], df["val_loss"], label="Val loss", color=PALETTE[1], lw=2)
        setup_ax(ax, "Epoch", "Loss")
        ax.set_title(label_model(folder), fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save_record("14", fig, "14_deney3_unetpp_loss_curves", title, "Epoch", "Loss",
                "Bu grafik Deney 3 U-Net++ modellerinin eğitim ve doğrulama kayıp eğrilerini yan yana verir.",
                "Kayıp eğrileri model yakınsaması ve train/validation ayrışmasını takip etmek için kullanılır.",
                "experiments/*/metrics.csv", "generate_thesis_figures.py::plot_14_exp3_loss_curves")


@safe_plot
def plot_15_exp4_bubble():
    """Deney 4 – Performans, Çıkarım Süresi ve Model Büyüklüğü İlişkisi"""
    title = "Deney 4 – Performans, Çıkarım Süresi ve Model Büyüklüğü İlişkisi"
    path = "deney_4/experiments_full/model_comparison_full.csv"
    df = numeric(read_csv(path), ["inference_time_per_image_ms", "test_dice_forged_only", "trainable_params"])
    df["model"] = df["experiment_name"].map(label_model)
    sizes = 180 + 900 * (df["trainable_params"] / df["trainable_params"].max())
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["inference_time_per_image_ms"], df["test_dice_forged_only"], s=sizes,
               c=PALETTE[: len(df)], alpha=0.8, edgecolor="#333333")
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["inference_time_per_image_ms"], row["test_dice_forged_only"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    setup_ax(ax, "Çıkarım süresi (ms/görüntü)", "Forged Dice")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("15", fig, "15_deney4_performans_sure_parametre_bubble", title,
                "Çıkarım süresi (ms/görüntü)", "Forged Dice",
                "Bu bubble scatter grafikte x ekseni çıkarım süresini, y ekseni forged Dice'ı, balon boyutu eğitilebilir parametre sayısını gösterir.",
                "Grafik doğruluk-hız-model büyüklüğü ödünleşimini aynı anda okumayı sağlar.", path,
                "generate_thesis_figures.py::plot_15_exp4_bubble")


@safe_plot
def plot_16_exp4_main_metrics():
    """Deney 4 – Dört Mimari Ailesinin Test Seti Üzerindeki Ana Performansları"""
    path = "deney_4/experiments_full/model_comparison_full.csv"
    df = read_csv(path)
    df["model"] = df["experiment_name"].map(label_model)
    grouped_bar(
        df,
        "model",
        [("test_dice_forged_only", "Forged Dice"), ("test_iou_forged_only", "Forged IoU"),
         ("component_f1", "Component F1@0.10"), ("image_f1", "Image F1")],
        "Deney 4 – Dört Mimari Ailesinin Test Seti Üzerindeki Ana Performansları",
        "Model",
        "Skor",
        "16",
        "16_deney4_ana_test_metrikleri_grouped",
        "Bu grafik dört mimari ailesini forged Dice, forged IoU, Component F1@0.10 ve Image F1 açısından karşılaştırır.",
        "Piksel, bileşen ve görüntü düzeyi metriklerin aynı anda verilmesi model davranışını daha bütünlüklü gösterir.",
        path,
        "generate_thesis_figures.py::plot_16_exp4_main_metrics",
        figsize=(10.5, 6),
        rotate=18,
    )


@safe_plot
def plot_17_exp4_auth_fp():
    """Deney 4 – Gerçek Görüntülerde Yanlış Pozitif Bileşen Oranı"""
    title = "Deney 4 – Gerçek Görüntülerde Yanlış Pozitif Bileşen Oranı"
    path = "deney_4/experiments_full/model_comparison_full.csv"
    df = numeric(read_csv(path), ["authentic_fp_rate"]).sort_values("authentic_fp_rate", ascending=False)
    df["model"] = df["experiment_name"].map(label_model)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(df["model"], df["authentic_fp_rate"], color=PALETTE[1])
    add_bar_labels(ax, bars)
    setup_ax(ax, "Model", "Authentic False Positive Rate")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["model"], rotation=18, ha="right")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("17", fig, "17_deney4_authentic_false_positive_rate", title, "Model", "Authentic False Positive Rate",
                "Bu grafik gerçek görüntülerde en az bir yanlış pozitif bileşen üretilme oranını gösterir.",
                "Deney 5'in post-processing motivasyonu bu yanlış alarm davranışını azaltma ihtiyacından doğar.", path,
                "generate_thesis_figures.py::plot_17_exp4_auth_fp")


@safe_plot
def plot_18_exp4_avg_components():
    """Deney 4 – Model Başına Ortalama Tahmin Bileşeni Sayısı"""
    title = "Deney 4 – Model Başına Ortalama Tahmin Bileşeni Sayısı"
    path = "deney_4/experiments_full/model_comparison_full.csv"
    df = numeric(read_csv(path), ["avg_pred_component_count"]).sort_values("avg_pred_component_count", ascending=False)
    df["model"] = df["experiment_name"].map(label_model)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(df["model"], df["avg_pred_component_count"], color=PALETTE[5])
    add_bar_labels(ax, bars, "{:.2f}")
    setup_ax(ax, "Model", "Ortalama tahmin bileşeni sayısı")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["model"], rotation=18, ha="right")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("18", fig, "18_deney4_ortalama_tahmin_bileseni", title, "Model", "Ortalama tahmin bileşeni sayısı",
                "Bu grafik her modelin görüntü başına ortalama kaç tahmin bileşeni ürettiğini gösterir.",
                "Yüksek bileşen sayısı parçalı tahminler ve yanlış alarm riski açısından önemlidir.", path,
                "generate_thesis_figures.py::plot_18_exp4_avg_components")


@safe_plot
def plot_19_exp4_q1_q4():
    """Deney 4 – Maske Boyutu Gruplarına Göre Model Performansı"""
    title = "Deney 4 – Maske Boyutu Gruplarına Göre Model Performansı"
    rows = []
    for path in (ROOT / "deney_4" / "experiments_4_full").glob("*/test_per_image_metrics.csv"):
        model = label_model(path.parent.name)
        df = numeric(pd.read_csv(path), ["gt_area", "dice"])
        forged = df[df.get("class_name", "").eq("forged") if "class_name" in df.columns else df["gt_area"].gt(0)].copy()
        if forged.empty:
            continue
        forged["quartile"] = pd.qcut(forged["gt_area"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        grp = forged.groupby("quartile", observed=True)["dice"].mean().reset_index()
        grp["model"] = model
        rows.append(grp)
    data = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (model, grp) in enumerate(data.groupby("model")):
        ax.plot(grp["quartile"], grp["dice"], marker="o", lw=2, label=model, color=PALETTE[i % len(PALETTE)])
    setup_ax(ax, "Q1, Q2, Q3, Q4 maske boyutu grupları", "Per-image Ortalama Dice")
    ax.legend(frameon=False, fontsize=8)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("19", fig, "19_deney4_q1_q4_lineplot", title,
                "Q1, Q2, Q3, Q4 maske boyutu grupları", "Per-image Ortalama Dice",
                "Bu çizgi grafik Deney 4 modellerinin forged örneklerde maske boyutu quartile'larına göre ortalama Dice değerini gösterir.",
                "Q1-Q2 çizgileri küçük sahtecilik bölgelerinde model hassasiyetini okumayı sağlar.",
                "deney_4/experiments_4_full/*/test_per_image_metrics.csv",
                "generate_thesis_figures.py::plot_19_exp4_q1_q4")


def confusion_from_per_image(path: Path) -> tuple[int, int, int, int]:
    df = pd.read_csv(path)
    y = pd.to_numeric(df["image_label"], errors="coerce").fillna(0).astype(int)
    pred = pd.to_numeric(df["image_pred"], errors="coerce").fillna(0).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return tp, fn, tn, fp


@safe_plot
def plot_20_exp4_combined_confusion_matrices():
    """Deney 4 - Dort modelin goruntu duzeyi karmasiklik matrisleri"""
    title = "Deney 4 - D\u00f6rt Modelin G\u00f6r\u00fcnt\u00fc D\u00fczeyi Karma\u015f\u0131kl\u0131k Matrisleri"
    paths = sorted((ROOT / "deney_4" / "experiments_4_full").glob("*/test_per_image_metrics.csv"))
    if not paths:
        raise FileNotFoundError("deney_4/experiments_4_full/*/test_per_image_metrics.csv bulunamadi")
    rows = []
    for path in paths:
        tp, fn, tn, fp = confusion_from_per_image(path)
        rows.append({"path": path, "model": label_model(path.parent.name), "matrix": np.array([[tp, fn], [fp, tn]])})
    vmax = max(int(item["matrix"].max()) for item in rows)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.8))
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.86, hspace=0.38, wspace=0.28)
    for ax, item in zip(axes.ravel(), rows):
        mat = item["matrix"]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(item["model"], fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Tahmin\nForged", "Tahmin\nAuthentic"], fontsize=9)
        ax.set_yticklabels(["Ger\u00e7ek\nForged", "Ger\u00e7ek\nAuthentic"], fontsize=9)
        ax.tick_params(length=0)
        for (r, c), val in np.ndenumerate(mat):
            ax.text(
                c,
                r,
                f"{int(val)}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if val > vmax * 0.52 else "#1F2933",
            )
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("G\u00f6r\u00fcnt\u00fc say\u0131s\u0131", rotation=90)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.96)
    save_record(
        "20",
        fig,
        "20_deney4_dort_model_confusion_matrices",
        title,
        "Tahmin s\u0131n\u0131f\u0131",
        "Ger\u00e7ek s\u0131n\u0131f",
        "Bu 2x2 panel Deney 4'teki d\u00f6rt modelin g\u00f6r\u00fcnt\u00fc d\u00fczeyi TP, FN, FP ve TN say\u0131lar\u0131n\u0131 tek g\u00f6rselde kar\u015f\u0131la\u015ft\u0131r\u0131r.",
        "Ortak renk \u00f6l\u00e7e\u011fi kullan\u0131ld\u0131; b\u00f6ylece modeller aras\u0131 yanl\u0131\u015f pozitif ve yanl\u0131\u015f negatif davran\u0131\u015f\u0131 do\u011frudan okunabilir.",
        "deney_4/experiments_4_full/*/test_per_image_metrics.csv",
        "generate_thesis_figures.py::plot_20_exp4_combined_confusion_matrices",
    )


@safe_plot
def plot_20_exp4_confusion_matrices():
    """Deney 4 – Görüntü Düzeyi Karışıklık Matrisleri"""
    for idx, path in enumerate((ROOT / "deney_4" / "experiments_4_full").glob("*/test_per_image_metrics.csv"), start=1):
        model = label_model(path.parent.name)
        title = f"Deney 4 – {model} Görüntü Düzeyi Karışıklık Matrisi"
        tp, fn, tn, fp = confusion_from_per_image(path)
        mat = np.array([[tp, fn], [fp, tn]])
        fig, ax = plt.subplots(figsize=(5.3, 4.8))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Tahmin: Forged", "Tahmin: Authentic"], rotation=15, ha="right")
        ax.set_yticks([0, 1], labels=["Gerçek: Forged", "Gerçek: Authentic"])
        for (r, c), val in np.ndenumerate(mat):
            ax.text(c, r, str(val), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if val > mat.max() * 0.55 else "#1F2933")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        add_title(fig, title)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        save_record(f"20{chr(96 + idx)}", fig, f"20{chr(96 + idx)}_deney4_{path.parent.name}_confusion_matrix",
                    title, "Tahmin sınıfı", "Gerçek sınıf",
                    "Bu heatmap görüntü düzeyinde TP, FN, FP ve TN sayılarını gösterir.",
                    "Yanlış pozitif ve yanlış negatif dağılımı modelin görüntü sınıflandırma davranışını açıklar.",
                    rel(path), "generate_thesis_figures.py::plot_20_exp4_confusion_matrices")


@safe_plot
def plot_21_exp5_before_after():
    """Deney 5 – Post-processing Öncesi ve Sonrası Performans Değişimi"""
    title = "Deney 5 – Post-processing Öncesi ve Sonrası Performans Değişimi"
    path = "deney_5/experiments_4_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv"
    df = read_csv(path)
    metrics = [
        ("dice_forged_only", "Forged Dice"),
        ("component_f1_iou010", "Component F1@0.10"),
        ("image_f1", "Image F1"),
        ("authentic_fp_rate", "Auth FP Rate"),
    ]
    selected = df[df["strategy"].isin(["raw_reference", "balanced_final_score"])].copy()
    selected["stage"] = selected["strategy"].map({"raw_reference": "Ham çıktı", "balanced_final_score": "Post-processing sonrası"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, lab) in zip(axes.ravel(), metrics):
        for i, (model, grp) in enumerate(selected.groupby("model_name")):
            vals = grp.set_index("stage")[col]
            if len(vals) >= 2:
                ax.plot(vals.index, vals.values, marker="o", lw=2, color=PALETTE[i % len(PALETTE)],
                        label=label_model(model) if col == metrics[0][0] else None)
        setup_ax(ax, "Aşama", lab)
    axes[0, 0].legend(frameon=False, fontsize=8)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("21", fig, "21_deney5_before_after_performance", title, "Aşama", "Metrik skoru / oran",
                "Bu dumbbell/slope yaklaşımı ham referans çıktı ile post-processing sonrası dengeli stratejiyi dört metrikte karşılaştırır.",
                "Auth FP Rate düşerken Dice, Component F1 ve Image F1 tarafındaki olası ödünleşimler aynı anda görülebilir.",
                path, "generate_thesis_figures.py::plot_21_exp5_before_after")


@safe_plot
def plot_21_exp5_before_after():
    """Deney 5 - Post-processing oncesi ve sonrasi performans degisimi"""
    title = "Deney 5 - Post-processing \u00d6ncesi ve Sonras\u0131 Performans De\u011fi\u015fimi"
    path = "deney_5/experiments_4_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv"
    df = read_csv(path)
    metrics = [
        ("dice_forged_only", "Forged Dice"),
        ("component_f1_iou010", "Component F1@0.10"),
        ("image_f1", "Image F1"),
        ("authentic_fp_rate", "Auth FP Rate"),
    ]
    selected = df[df["strategy"].isin(["raw_reference", "balanced_final_score"])].copy()
    selected["stage"] = selected["strategy"].map({"raw_reference": "Ham \u00e7\u0131kt\u0131", "balanced_final_score": "Post-processing"})
    stage_order = ["Ham \u00e7\u0131kt\u0131", "Post-processing"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, lab) in zip(axes.ravel(), metrics):
        for i, (model, grp) in enumerate(selected.groupby("model_name")):
            vals = pd.to_numeric(grp.set_index("stage").reindex(stage_order)[col], errors="coerce")
            if vals.notna().sum() >= 2:
                color = PALETTE[i % len(PALETTE)]
                ax.plot([0, 1], vals.values, marker="o", lw=2.3, color=color,
                        label=label_model(model) if col == metrics[0][0] else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(stage_order)
        ax.set_xlim(-0.18, 1.18)
        setup_ax(ax, "A\u015fama", lab)
    axes[0, 0].legend(frameon=False, fontsize=8)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("21", fig, "21_deney5_before_after_performance", title, "Ham \u00e7\u0131kt\u0131 -> Post-processing", "Metrik skoru / oran",
                "Bu slope grafik, Deney 5 ham referans ciktisi ile post-processing sonrasi dengeli stratejiyi dort metrikte karsilastirir.",
                "X ekseni once ham ciktiyi, sonra post-processing asamasini verir; Auth FP Rate icin dusus olumlu yorumlanir.",
                path, "generate_thesis_figures.py::plot_21_exp5_before_after")


@safe_plot
def plot_22_exp5_authfp_drop():
    """Deney 5 – Post-processing ile Authentic False Positive Rate Azalması"""
    title = "Deney 5 – Post-processing ile Authentic False Positive Rate Azalması"
    path = "deney_5/experiments_4_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv"
    df = read_csv(path)
    df = numeric(df[df["strategy"].isin(["raw_reference", "low_false_alarm"])].copy(), ["authentic_fp_rate"])
    df["model"] = df["model_name"].map(label_model)
    df["strategy_label"] = df["strategy"].map(label_strategy)
    pivot = df.pivot_table(index="model", columns="strategy_label", values="authentic_fp_rate", aggfunc="mean").reset_index()
    cols = [(c, c) for c in ["Ham çıktı", "Düşük yanlış alarm"] if c in pivot.columns]
    grouped_bar(
        pivot,
        "model",
        cols,
        title,
        "Model",
        "Authentic False Positive Rate",
        "22",
        "22_deney5_authfp_drop",
        "Bu grafik ham referans çıktı ile Deney 5 düşük yanlış alarm stratejisinin authentic false positive rate değerlerini karşılaştırır.",
        "Yanlış alarm oranındaki düşüş Deney 5 post-processing kararlarının pratik katkısını doğrudan gösterir.",
        path,
        "generate_thesis_figures.py::plot_22_exp5_authfp_drop",
        figsize=(9, 5.6),
        rotate=18,
    )


@safe_plot
def plot_23_exp5_tradeoff():
    """Deney 5 – Kalibrasyon Stratejileri Arasındaki Performans-Yanlış Alarm Dengesi"""
    title = "Deney 5 – Kalibrasyon Stratejileri Arasındaki Performans-Yanlış Alarm Dengesi"
    path = "deney_5/experiments_4_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv"
    df = numeric(read_csv(path), ["authentic_fp_rate", "dice_forged_only", "final_score"])
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "P", "X"]
    for i, (model, grp) in enumerate(df.groupby("model_name")):
        for j, (_, row) in enumerate(grp.iterrows()):
            ax.scatter(row["authentic_fp_rate"], row["dice_forged_only"], s=70,
                       color=PALETTE[i % len(PALETTE)], marker=markers[j % len(markers)],
                       edgecolor="#333333", linewidth=0.5,
                       label=label_model(model) if j == 0 else None)
    setup_ax(ax, "Authentic False Positive Rate", "Forged Dice")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("23", fig, "23_deney5_tradeoff_scatter", title, "Authentic False Positive Rate", "Forged Dice",
                "Bu scatter plot Deney 5 stratejilerinin forged Dice ile authentic false positive rate arasındaki dengesini gösterir.",
                "Sol üst bölge yüksek lokalizasyon başarımı ve düşük yanlış alarm anlamına gelir.", path,
                "generate_thesis_figures.py::plot_23_exp5_tradeoff")


@safe_plot
def plot_24_exp5_calibration_metrics():
    """Deney 5 – Görüntü Düzeyi Kalibrasyon ve Ayrım Gücü Metrikleri"""
    title = "Deney 5 – Görüntü Düzeyi Kalibrasyon ve Ayrım Gücü Metrikleri"
    path = "deney_5/experiments_4_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv"
    df = read_csv(path)
    df = df[df["strategy"].isin(["balanced_final_score", "low_false_alarm"])].copy()
    df["model_strategy"] = df["model_name"].map(label_model) + "\n" + df["strategy"].map(label_strategy)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    metrics_left = [("image_roc_auc", "ROC-AUC"), ("image_auprc", "AUPRC")]
    metrics_right = [("image_brier", "Brier Score"), ("image_ece_10bin", "ECE")]
    for ax, metrics, ylabel in [(axes[0], metrics_left, "Yüksek daha iyi"), (axes[1], metrics_right, "Düşük daha iyi")]:
        x = np.arange(len(df))
        width = 0.35
        for i, (col, lab) in enumerate(metrics):
            ax.bar(x + (i - 0.5) * width, pd.to_numeric(df[col], errors="coerce"), width,
                   label=lab, color=PALETTE[i + (0 if ax is axes[0] else 2)])
        ax.set_xticks(x)
        ax.set_xticklabels(df["model_strategy"], rotation=28, ha="right")
        setup_ax(ax, "Model / strateji", ylabel)
        ax.legend(frameon=False, fontsize=9)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_record("24", fig, "24_deney5_calibration_metrics", title, "Model / strateji", "Kalibrasyon ve ayrım metriği",
                "Bu grafik ROC-AUC/AUPRC ile Brier Score/ECE metriklerini ayrı panellerde verir.",
                "Brier ve ECE düşük daha iyi olduğu için ayrım gücü metriklerinden ayrı okunmalıdır.", path,
                "generate_thesis_figures.py::plot_24_exp5_calibration_metrics")


@safe_plot
def plot_25_exp6_delta():
    """Deney 6 – 384×384 Çözünürlüğün Önceki Ayarlara Göre Sağladığı Değişim"""
    title = "Deney 6 – 384×384 Çözünürlüğün Önceki Ayarlara Göre Sağladığı Değişim"
    path = "final_analysis/final_analysis/final_model_comparison.csv"
    df = read_csv(path)
    pairs = [
        ("SegFormer-B0 384", "SegFormer-B0 256"),
        ("EfficientNetB0-UNet 384", "EfficientNetB0-UNet 256"),
    ]
    metrics = [
        ("forged_dice", "Δ Forged Dice"),
        ("q1_dice", "Δ Q1 Dice"),
        ("q2_dice", "Δ Q2 Dice"),
        ("component_f1_iou010", "Δ Component F1"),
        ("image_f1", "Δ Image F1"),
        ("authentic_fp_rate", "Δ Auth FP Rate"),
    ]
    rows = []
    for new, old in pairs:
        if new in set(df["model_name"]) and old in set(df["model_name"]):
            nrow = df[df["model_name"].eq(new)].iloc[0]
            orow = df[df["model_name"].eq(old)].iloc[0]
            for col, lab in metrics:
                rows.append({"model": new.replace(" 384", ""), "metric": lab, "delta": pd.to_numeric(nrow[col], errors="coerce") - pd.to_numeric(orow[col], errors="coerce")})
    data = pd.DataFrame(rows).dropna()
    fig, ax = plt.subplots(figsize=(11, 5.8))
    metric_order = [m[1] for m in metrics if m[1] in set(data["metric"])]
    x = np.arange(len(metric_order))
    width = 0.35
    for i, (model, grp) in enumerate(data.groupby("model")):
        grp = grp.set_index("metric").reindex(metric_order)
        ax.bar(x[: len(grp)] + (i - 0.5) * width, grp["delta"], width, label=model, color=PALETTE[i])
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_order, rotation=25, ha="right")
    setup_ax(ax, "Metrik", "384 - 256 değişimi")
    ax.legend(frameon=False)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_record("25", fig, "25_deney6_delta_metrics", title, "Metrik", "384 - 256 değişimi",
                "Bu delta bar grafik final analizdeki 384 çözünürlük sonuçlarını 256 referanslarıyla karşılaştırır.",
                "Pozitif değer iyileşme, negatif değer düşüş anlamına gelir; Auth FP Rate için negatif değer daha iyi yorumlanır.",
                path, "generate_thesis_figures.py::plot_25_exp6_delta")


@safe_plot
def plot_26_exp6_strategy_performance():
    """Deney 6 – Stratejilere Göre Test Performansı"""
    path = "deney_6/experiments_full/experiment6_smallmask_384/test_results_all_strategies.csv"
    df = read_csv(path)
    df["model_strategy"] = df["model_name"].map(label_model) + "\n" + df["strategy"].map(label_strategy)
    grouped_bar(
        df,
        "model_strategy",
        [("final_score", "Final Score"), ("dice_forged_only", "Forged Dice"), ("q1_dice", "Q1 Dice"),
         ("q2_dice", "Q2 Dice"), ("component_f1_iou010", "Component F1@0.10"), ("image_f1", "Image F1"),
         ("authentic_fp_rate", "Auth FP Rate")],
        "Deney 6 – Stratejilere Göre Test Performansı",
        "Model / strateji",
        "Skor / oran",
        "26",
        "26_deney6_strategy_test_performance",
        "Bu grafik Deney 6 stratejilerinin final skor, küçük maske Dice, bileşen F1, görüntü F1 ve yanlış alarm değerlerini birlikte verir.",
        "Strateji seçimi yalnız tek metriğe değil, küçük maske başarımı ve yanlış alarm dengesine göre okunmalıdır.",
        path,
        "generate_thesis_figures.py::plot_26_exp6_strategy_performance",
        figsize=(14, 7),
        rotate=35,
    )


@safe_plot
def plot_27_exp6_q1_q4():
    """Deney 6 – 384×384 Çözürlükte Q1-Q4 Maske Gruplarına Göre Performans"""
    title = "Deney 6 – 384×384 Çözünürlükte Q1-Q4 Maske Gruplarına Göre Performans"
    path = "final_analysis/final_analysis/final_model_comparison.csv"
    df = numeric(read_csv(path), ["q1_dice", "q2_dice", "q3_dice", "q4_dice"])
    df = df.dropna(subset=["q1_dice", "q2_dice", "q3_dice", "q4_dice"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot(["Q1", "Q2", "Q3", "Q4"], [row["q1_dice"], row["q2_dice"], row["q3_dice"], row["q4_dice"]],
                marker="o", lw=2, label=row["model_name"], color=PALETTE[i % len(PALETTE)])
    setup_ax(ax, "Q1-Q4 maske boyutu grupları", "Per-image Ortalama Dice")
    ax.legend(frameon=False, fontsize=8)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("27", fig, "27_deney6_q1_q4_performance", title,
                "Q1-Q4 maske boyutu grupları", "Per-image Ortalama Dice",
                "Bu grafik 384×384 final modellerinin ve varsa 256 referanslarının Q1-Q4 maske gruplarındaki Dice performansını gösterir.",
                "Q1 ve Q2 eğrileri küçük sahtecilik alanlarında çözünürlük etkisini okumak için kritiktir.",
                path, "generate_thesis_figures.py::plot_27_exp6_q1_q4")


@safe_plot
def plot_28_exp6_small_vs_fp():
    """Deney 6 – Küçük Maske Başarımı ile Yanlış Alarm Arasındaki Denge"""
    title = "Deney 6 – Küçük Maske Başarımı ile Yanlış Alarm Arasındaki Denge"
    path = "deney_6/experiments_full/experiment6_smallmask_384/test_results_all_strategies.csv"
    df = numeric(read_csv(path), ["authentic_fp_rate", "q1_dice"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (model, grp) in enumerate(df.groupby("model_name")):
        ax.scatter(grp["authentic_fp_rate"], grp["q1_dice"], s=85, color=PALETTE[i], label=label_model(model),
                   edgecolor="#333333", linewidth=0.6)
        for _, row in grp.iterrows():
            ax.annotate(label_strategy(row["strategy"]), (row["authentic_fp_rate"], row["q1_dice"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    setup_ax(ax, "Authentic False Positive Rate", "Q1 Dice")
    ax.legend(frameon=False)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("28", fig, "28_deney6_q1dice_authfp_scatter", title, "Authentic False Positive Rate", "Q1 Dice",
                "Bu scatter plot Deney 6 stratejilerinde küçük maske başarımı ile gerçek görüntülerde yanlış alarm oranını karşılaştırır.",
                "Sol üst bölge küçük maske başarımı yüksek ve yanlış alarmı düşük stratejileri gösterir.", path,
                "generate_thesis_figures.py::plot_28_exp6_small_vs_fp")


@safe_plot
def plot_29_exp6_component_iou_thresholds():
    """Deney 6 – Farklı IoU Eşiklerinde Bileşen Düzeyi Tespit Başarısı"""
    path = "deney_6/experiments_full/experiment6_smallmask_384/test_results_all_strategies.csv"
    df = read_csv(path)
    df = df[df["strategy"].isin(["balanced_final_score", "low_false_alarm", "best_small_mask_q1_dice"])].copy()
    df["model_strategy"] = df["model_name"].map(label_model) + "\n" + df["strategy"].map(label_strategy)
    grouped_bar(
        df,
        "model_strategy",
        [("component_f1_iou010", "IoU 0.10"), ("component_f1_iou025", "IoU 0.25"), ("component_f1_iou050", "IoU 0.50")],
        "Deney 6 – Farklı IoU Eşiklerinde Bileşen Düzeyi Tespit Başarısı",
        "Model / strateji",
        "Component F1",
        "29",
        "29_deney6_component_f1_iou_thresholds",
        "Bu grouped bar grafik bileşen düzeyi F1 skorunu IoU 0.10, 0.25 ve 0.50 eşiklerinde verir.",
        "Yüksek IoU eşiğine geçildikçe düşüş, tahmin bileşenlerinin konumsal örtüşüm hassasiyetini gösterir.",
        path,
        "generate_thesis_figures.py::plot_29_exp6_component_iou_thresholds",
        figsize=(12, 6),
        rotate=30,
    )


@safe_plot
def plot_30_final_scorecard():
    """Final Analiz – İki Final Modelin Karşılaştırmalı Performans Özeti"""
    title = "Final Analiz – İki Final Modelin Karşılaştırmalı Performans Özeti"
    path = "final_analysis/final_analysis/final_model_comparison.csv"
    df = read_csv(path)
    df = df[df["model_name"].isin(["SegFormer-B0 384", "EfficientNetB0-UNet 384"])].copy()
    metric_map = [
        ("forged_dice", "Forged Dice"),
        ("forged_iou", "Forged IoU"),
        ("q1_dice", "Q1 Dice"),
        ("q2_dice", "Q2 Dice"),
        ("component_f1_iou010", "Component F1"),
        ("image_f1", "Image F1"),
        ("image_roc_auc", "ROC-AUC"),
        ("image_auprc", "AUPRC"),
        ("authentic_fp_rate", "Auth FP Rate"),
    ]
    matrix = []
    for col, _ in metric_map:
        matrix.append(pd.to_numeric(df[col], errors="coerce").to_numpy())
    matrix = np.vstack(matrix)
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([name.replace(" 384", "\n384") for name in df["model_name"]], rotation=0, ha="center")
    ax.set_yticks(np.arange(len(metric_map)))
    ax.set_yticklabels([m[1] for m in metric_map])
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    color="white" if val > 0.58 else "#1F2933", fontsize=10, fontweight="bold")
    ax.set_xlabel("Final model", fontsize=11, labelpad=10)
    ax.set_ylabel("Metrik", fontsize=11, labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Skor / oran", rotation=90)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("30", fig, "30_final_model_scorecard", title, "Final model", "Metrik",
                "Bu scorecard final iki modeli piksel, küçük maske, bileşen, görüntü düzeyi ve yanlış alarm metrikleriyle birlikte karşılaştırır.",
                "SegFormer-B0 384 lokalizasyon tarafında, EfficientNetB0-UNet 384 düşük yanlış alarm rolünde birlikte okunmalıdır; Auth FP Rate için düşük değer daha iyidir.",
                path, "generate_thesis_figures.py::plot_30_final_scorecard")


@safe_plot
def plot_31_final_robustness():
    """Final Analiz – Bozulma Türlerine Karşı Model Dayanıklılığı"""
    title = "Final Analiz – Bozulma Türlerine Karşı Model Dayanıklılığı"
    path = "final_analysis/final_analysis/robustness_metrics_all.csv"
    df = read_csv(path)
    metrics = [
        ("forged_dice", "Forged Dice"),
        ("q1_dice", "Q1 Dice"),
        ("component_f1_iou010", "Component F1@0.10"),
        ("authentic_fp_rate", "Auth FP Rate"),
    ]
    order = ["clean_png", "jpeg_q90", "jpeg_q70", "jpeg_q50", "gaussian_blur_light", "gaussian_blur_medium",
             "gaussian_noise_light", "gaussian_noise_medium", "combined_jpeg70_blur_light"]
    df["degradation"] = pd.Categorical(df["degradation"], categories=order, ordered=True)
    df = df.sort_values("degradation")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (col, lab) in zip(axes.ravel(), metrics):
        for i, (model, grp) in enumerate(df.groupby("model_name", observed=True)):
            ax.plot(grp["degradation"].astype(str), pd.to_numeric(grp[col], errors="coerce"),
                    marker="o", lw=2, label=label_model(model), color=PALETTE[i])
        setup_ax(ax, "Bozulma türü", lab)
        ax.tick_params(axis="x", rotation=35)
    axes[0, 0].legend(frameon=False)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_record("31", fig, "31_final_robustness_small_multiples", title, "Bozulma türü", "Metrik değeri",
                "Bu small-multiples grafik final modellerin JPEG, blur, noise ve birleşik bozulmalara karşı metrik değişimini gösterir.",
                "Clean çizgisine göre düşüşler model dayanıklılığının hangi bozulmalarda zayıfladığını gösterir.",
                path, "generate_thesis_figures.py::plot_31_final_robustness")


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y = y_true[order]
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.r_[0, tp / positives, 1]
    fpr = np.r_[0, fp / negatives, 1]
    return fpr, tpr


def pr_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((y == 1).sum()), 1)
    return np.r_[0, recall], np.r_[1, precision]


def final_per_image_files() -> list[tuple[str, Path]]:
    return [
        ("SegFormer-B0 384", ROOT / "final_analysis/final_analysis/segformer_b0_rgb_384_smallmask/test_per_image_metrics_balanced_final_score.csv"),
        ("EfficientNetB0-UNet 384", ROOT / "final_analysis/final_analysis/efficientnetb0_unet_rgb_384_smallmask/test_per_image_metrics_low_false_alarm.csv"),
    ]


@safe_plot
def plot_32_final_roc():
    """Final Analiz – İki Final Modelin ROC Eğrileri"""
    title = "Final Analiz – İki Final Modelin ROC Eğrileri"
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for i, (label, path) in enumerate(final_per_image_files()):
        df = read_csv(path)
        y = pd.to_numeric(df["image_label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        s = pd.to_numeric(df["image_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
        fpr, tpr = roc_curve_points(y, s)
        ax.plot(fpr, tpr, lw=2.2, label=label, color=PALETTE[i])
    ax.plot([0, 1], [0, 1], color="#777777", lw=1, ls="--")
    setup_ax(ax, "False Positive Rate", "True Positive Rate")
    ax.legend(frameon=False)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_record("32", fig, "32_final_roc_curves", title, "False Positive Rate", "True Positive Rate",
                "Bu ROC eğrileri final modellerin görüntü düzeyinde sahte/gerçek ayrımını eşik bağımsız olarak gösterir.",
                "Eğrinin sol üst köşeye yakınlığı daha güçlü ayrım anlamına gelir.",
                "final_analysis/final_analysis/*/test_per_image_metrics_*.csv",
                "generate_thesis_figures.py::plot_32_final_roc")


@safe_plot
def plot_33_final_pr():
    """Final Analiz – İki Final Modelin Precision-Recall Eğrileri"""
    title = "Final Analiz – İki Final Modelin Precision-Recall Eğrileri"
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for i, (label, path) in enumerate(final_per_image_files()):
        df = read_csv(path)
        y = pd.to_numeric(df["image_label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        s = pd.to_numeric(df["image_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
        recall, precision = pr_curve_points(y, s)
        ax.plot(recall, precision, lw=2.2, label=label, color=PALETTE[i])
    setup_ax(ax, "Recall", "Precision")
    ax.legend(frameon=False)
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_record("33", fig, "33_final_precision_recall_curves", title, "Recall", "Precision",
                "Bu Precision-Recall eğrileri final modellerin forged görüntü yakalama ve yanlış alarm dengesini gösterir.",
                "Özellikle sınıf dengesizliği durumlarında PR eğrisi ROC eğrisini tamamlayıcı bilgi sağlar.",
                "final_analysis/final_analysis/*/test_per_image_metrics_*.csv",
                "generate_thesis_figures.py::plot_33_final_pr")


@safe_plot
def plot_34_failure_cases():
    """Final Analiz – Başarısızlık Örnekleri ve Tipik Hata Durumları"""
    source = ROOT / "final_analysis/final_analysis/plots/model_disagreement_examples.png"
    if not source.exists():
        raise FileNotFoundError("Hazır failure/disagreement paneli bulunamadı; ham Kaggle görüntüleri yerel değil.")
    wrap_existing_image(
        "34",
        source,
        "Final Analiz – Başarısızlık Örnekleri ve Tipik Hata Durumları",
        "34_final_failure_case_examples",
        "Bu panel final analizde modellerin ayrıştığı veya hata örüntüsü gösteren örnekleri birlikte sunar.",
        "Tipik hata örnekleri metrik özetlerinin arkasındaki görsel davranışı açıklamak için kullanılır.",
    )


def draw_flow_figure(
    fig_id: str,
    filename: str,
    title: str,
    steps: list[tuple[str, str]],
    caption: str,
    comment: str,
    source: str,
    code_ref: str,
    figsize: tuple[float, float] = (13, 4.8),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(steps))
    box_w = min(0.14, 0.82 / max(len(steps), 1))
    for i, ((head, body), x) in enumerate(zip(steps, xs)):
        color = PALETTE[i % len(PALETTE)]
        rect = plt.Rectangle((x - box_w / 2, 0.42), box_w, 0.30, facecolor=color, edgecolor="none", alpha=0.96)
        ax.add_patch(rect)
        ax.text(x, 0.62, "\n".join(textwrap.wrap(head, 18)), ha="center", va="center",
                fontsize=9.4, fontweight="bold", color="white")
        ax.text(x, 0.49, "\n".join(textwrap.wrap(body, 20)), ha="center", va="center",
                fontsize=8.0, color="white")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - box_w / 2 - 0.008, 0.57), xytext=(x + box_w / 2 + 0.008, 0.57),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="#3B414A"))
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_record(fig_id, fig, filename, title, "Pipeline asamasi", "Islem / karar",
                caption, comment, source, code_ref)


@safe_plot
def plot_35_preprocessing_training_flow():
    """3.3 On isleme ve egitim veri akisi"""
    draw_flow_figure(
        "35",
        "35_3_3_on_isleme_egitim_akisi",
        "3.3 \u00d6n \u0130\u015fleme ve E\u011fitim Veri Ak\u0131\u015f\u0131",
        [
            ("G\u00f6r\u00fcnt\u00fc / maske okuma", "RGB image ve .npy maske"),
            ("Binary union", "Tum pozitif maske kanallari birlesir"),
            ("Resize", "G\u00f6r\u00fcnt\u00fc sabit \u00e7\u00f6z\u00fcn\u00fcrl\u00fc\u011fe iner"),
            ("Nearest mask resize", "Maske sinirlari korunur"),
            ("Normalize", "ImageNet mean/std"),
            ("Augmentasyon", "Train-only geometrik ve fotometrik"),
            ("Model giri\u015fi", "Tensor image + binary target"),
        ],
        "Bu sema, ham veri okumasindan model girdisine kadar kullanilan on isleme ve egitim veri akisini ozetler.",
        "Maske icin nearest-neighbor resize ayrica gosterilmistir; cunku binary target yapisini koruyan kritik adimdir.",
        "proje_kodlari_codex/recod_luc_scientific_forgery_colab.py; recod_luc_4model_evaluate_existing_checkpoints.py",
        "generate_thesis_figures.py::plot_35_preprocessing_training_flow",
    )


@safe_plot
def plot_36_model_family_map():
    """3.4 Model aileleri ve secim gerekcesi"""
    title = "3.4 Model Aileleri ve Se\u00e7im Gerek\u00e7esi"
    families = [
        ("U-Net++", "Nested skip connection\nCNN segmentation baseline", PALETTE[0]),
        ("EfficientNetB0-UNet", "Parametre verimli\ntransfer encoder-decoder", PALETTE[2]),
        ("SegFormer-B0", "Transformer tabanli\nsemantic segmentation", PALETTE[4]),
        ("DINOv2-lite", "Foundation feature\nhafif decoder", PALETTE[5]),
        ("U-Net++ RGB+SRM edge", "Forensic residual +\nedge multitask ablation", PALETTE[1]),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.axis("off")
    ax.text(0.5, 0.83, "Model havuzu", ha="center", va="center", fontsize=13, fontweight="bold", color="#1F2933")
    ax.text(0.5, 0.75, "Ayn\u0131 binary forged-mask hedefi \u00fczerinde farkl\u0131 temsil ve mimari aileleri",
            ha="center", va="center", fontsize=10, color="#4B5563")
    xs = np.linspace(0.10, 0.90, len(families))
    for i, (head, body, color) in enumerate(families):
        rect = plt.Rectangle((xs[i] - 0.085, 0.32), 0.17, 0.27, facecolor=color, edgecolor="none", alpha=0.96)
        ax.add_patch(rect)
        ax.text(xs[i], 0.50, head, ha="center", va="center", fontsize=9.3, fontweight="bold", color="white")
        ax.text(xs[i], 0.39, body, ha="center", va="center", fontsize=8.0, color="white")
        ax.annotate("", xy=(xs[i], 0.60), xytext=(0.5, 0.72),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#606B78", alpha=0.8))
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_record(
        "36", fig, "36_3_4_model_aileleri_haritasi", title, "Model ailesi", "Secim gerekcesi",
        "Bu harita, tezde karsilastirilan ana model ailelerini ve her ailenin deneysel roldeki gerekcesini ozetler.",
        "Model aileleri ayni segmentation hedefinde, farkli temsil/encoder varsayimlarini sinamak icin kullanilmistir.",
        "analysis_review/previous_pilot_experiment_design.md; analysis_review/experiment_3_codex_full_analysis.md; deney_4/experiments_full/model_comparison_full.csv",
        "generate_thesis_figures.py::plot_36_model_family_map",
    )


@safe_plot
def plot_37_metric_framework():
    """3.5 Cok katmanli degerlendirme cercevesi"""
    title = "3.5 \u00c7ok Katmanl\u0131 De\u011ferlendirme \u00c7er\u00e7evesi"
    layers = [
        ("Piksel d\u00fczeyi", ["Dice / F1", "IoU", "Precision", "Recall", "AUPRC"]),
        ("G\u00f6r\u00fcnt\u00fc d\u00fczeyi", ["Image score", "Accuracy", "F1", "ROC-AUC", "Specificity"]),
        ("Bile\u015fen d\u00fczeyi", ["Connected components", "Hungarian matching", "Component F1", "Auth FP rate"]),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.axis("off")
    xs = [0.18, 0.50, 0.82]
    for i, ((head, items), x) in enumerate(zip(layers, xs)):
        rect = plt.Rectangle((x - 0.13, 0.24), 0.26, 0.48, facecolor="#F7F8FA", edgecolor=PALETTE[i], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.64, head, ha="center", va="center", fontsize=12, fontweight="bold", color=PALETTE[i])
        for j, item in enumerate(items):
            ax.text(x, 0.55 - j * 0.075, item, ha="center", va="center", fontsize=9.2, color="#1F2933")
    ax.text(0.5, 0.12, "Nihai yorum: lokalizasyon kalitesi + g\u00f6r\u00fcnt\u00fc karar\u0131 + pratik false-alarm davran\u0131\u015f\u0131 birlikte okunur",
            ha="center", va="center", fontsize=10, color="#374151", fontweight="bold")
    add_title(fig, title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_record(
        "37", fig, "37_3_5_degerlendirme_metrikleri_cercevesi", title,
        "Degerlendirme katmani", "Metrikler",
        "Bu sema, piksel, goruntu ve bilesen duzeyi metriklerin birbirini tamamlayan rollerini gosterir.",
        "Tek bir Dice skoru yerine cok katmanli okuma, sahte bolge lokalizasyonu ve gercek goruntu false alarm davranisini birlikte degerlendirir.",
        "recod_luc_final_analysis.py; recod_luc_experiment5_calibration_postprocessing.py",
        "generate_thesis_figures.py::plot_37_metric_framework",
    )


@safe_plot
def plot_38_threshold_postprocessing_pipeline():
    """3.6 Esikleme ve post-processing pipeline'i"""
    draw_flow_figure(
        "38",
        "38_3_6_esikleme_postprocessing_pipeline",
        "3.6 E\u015fikleme ve Post-processing Pipeline'\u0131",
        [
            ("G\u00f6r\u00fcnt\u00fc", "Test image"),
            ("Model", "E\u011fitilmi\u015f checkpoint"),
            ("Probability map", "P(sahte piksel)"),
            ("Threshold", "Validation se\u00e7imli e\u015fik"),
            ("Raw mask", "\u0130kili tahmin"),
            ("Connected components", "Bile\u015fen analizi"),
            ("Post-processing", "Alan / olas\u0131l\u0131k / morfoloji"),
            ("Final mask", "Temizlenmi\u015f tahmin"),
            ("Image score", "G\u00f6r\u00fcnt\u00fc d\u00fczeyi karar"),
        ],
        "Bu pipeline, model cikti haritasinin once ham maskeye, sonra post-processing sonrasi final maske ve image score'a donusumunu gosterir.",
        "Validation setinde secilen esik ve filtreler testte sabit uygulanir; test seti karar secimi icin kullanilmaz.",
        "deney_5/experiments_4_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv",
        "generate_thesis_figures.py::plot_38_threshold_postprocessing_pipeline",
        figsize=(14.5, 4.9),
    )


@safe_plot
def plot_39_robustness_failure_protocol():
    """3.7 Robustness ve failure case analiz protokolu"""
    draw_flow_figure(
        "39",
        "39_3_7_robustness_failure_case_analizi",
        "3.7 Robustness / Failure Case Analizi Protokol\u00fc",
        [
            ("Clean test", "Referans performans"),
            ("Bozulma testleri", "JPEG / blur / noise"),
            ("Delta analizi", "Clean'e g\u00f6re metrik fark\u0131"),
            ("Failure case se\u00e7imi", "FP / FN / low Dice / small mask"),
            ("G\u00f6rsel panel analizi", "GT, prediction ve model ayr\u0131\u015fmas\u0131"),
        ],
        "Bu sema, final analizde temiz testten bozulma testlerine ve hata ornegi panellerine uzanan analiz protokolunu ozetler.",
        "Robustness ve failure-case analizi sayisal metrikleri, modelin tipik hata davranisini gosteren gorsel kanitlarla tamamlar.",
        "final_analysis/final_analysis/robustness_metrics_all.csv; final_analysis/final_analysis/*/failure_cases/*.csv",
        "generate_thesis_figures.py::plot_39_robustness_failure_protocol",
        figsize=(12.5, 4.8),
    )


def write_catalog() -> None:
    csv_path = OUT / "figure_manifest.csv"
    fields = ["id", "status", "png", "svg", "title", "x_axis", "y_axis", "caption", "comment", "source", "code"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    md_path = OUT / "figure_catalog.md"
    lines = [
        "# Tez Figürleri Yeniden Üretim Kataloğu",
        "",
        f"Üretim scripti: `{rel(Path(__file__))}`",
        f"Çıktı klasörü: `{rel(OUT)}`",
        "",
        "Metrik notları: Dice = Dice/F1 overlap measure; IoU = Intersection over Union; AUPRC = Area Under Precision-Recall Curve; Auth FP Rate = Authentic False Positive Rate; Component F1 = connected component düzeyi F1 skoru.",
        "",
    ]
    for item in manifest:
        lines.append(f"## {item['id']} - {item['title']}")
        lines.append("")
        lines.append(f"- Durum: `{item['status']}`")
        if item["png"]:
            lines.append(f"- PNG: `{item['png']}`")
            lines.append(f"- SVG: `{item['svg']}`")
        lines.append(f"- X ekseni: {item['x_axis'] or '-'}")
        lines.append(f"- Y ekseni: {item['y_axis'] or '-'}")
        lines.append(f"- Şekil altı açıklaması: {item['caption']}")
        if item["comment"]:
            lines.append(f"- Kısa yorum: {item['comment']}")
        lines.append(f"- Veri kaynağı: `{item['source']}`")
        lines.append(f"- Grafik kodu: `{item['code']}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for func in [
        plot_01_method_flow,
        plot_02_split_distribution,
        plot_03_mask_quartiles,
        plot_04_exp1_forged_f1,
        plot_05_exp1_iou,
        plot_06_exp1_precision_recall,
        plot_07_exp1_image_level,
        plot_08a_segformer_training,
        plot_08b_deeplab_training,
        plot_08c_effnet_training,
        plot_09_prediction_panel,
        plot_10_seed_mean_std,
        plot_11_seed_lines,
        plot_12_exp3_config_comparison,
        plot_13_exp3_surface_edge,
        plot_14_exp3_loss_curves,
        plot_15_exp4_bubble,
        plot_16_exp4_main_metrics,
        plot_17_exp4_auth_fp,
        plot_18_exp4_avg_components,
        plot_19_exp4_q1_q4,
        plot_20_exp4_combined_confusion_matrices,
        plot_20_exp4_confusion_matrices,
        plot_21_exp5_before_after,
        plot_22_exp5_authfp_drop,
        plot_23_exp5_tradeoff,
        plot_24_exp5_calibration_metrics,
        plot_25_exp6_delta,
        plot_26_exp6_strategy_performance,
        plot_27_exp6_q1_q4,
        plot_28_exp6_small_vs_fp,
        plot_29_exp6_component_iou_thresholds,
        plot_30_final_scorecard,
        plot_31_final_robustness,
        plot_32_final_roc,
        plot_33_final_pr,
        plot_34_failure_cases,
        plot_35_preprocessing_training_flow,
        plot_36_model_family_map,
        plot_37_metric_framework,
        plot_38_threshold_postprocessing_pipeline,
        plot_39_robustness_failure_protocol,
    ]:
        func()
    write_catalog()
    created = sum(1 for item in manifest if item["status"] == "created")
    skipped = sum(1 for item in manifest if item["status"] == "skipped")
    print(f"Created {created} figures; skipped {skipped}. Output: {OUT}")


if __name__ == "__main__":
    main()
