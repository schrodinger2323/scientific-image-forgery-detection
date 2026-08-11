"""Generate slide-ready 256 vs 384 small-mask comparison visuals.

Run this on Kaggle when the Recod.ai/LUC dataset is mounted. The script uses
cached probability maps from Experiment 5 (256x256) and Experiment 6 (384x384),
then reads the original images and masks from the dataset for clean overlays.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # Summary-only mode does not need OpenCV.
    cv2 = None


EPS = 1e-7


@dataclass(frozen=True)
class RunSpec:
    key: str
    model_dir: str
    strategy: str
    label: str
    resolution: int


RUNS = [
    RunSpec("seg256", "segformer_b0_rgb_full", "balanced_final_score", "SegFormer 256", 256),
    RunSpec("seg384", "segformer_b0_rgb_384_smallmask", "balanced_final_score", "SegFormer 384", 384),
    RunSpec("eff256", "efficientnetb0_unet_rgb_full", "low_false_alarm", "EfficientNet 256", 256),
    RunSpec("eff384", "efficientnetb0_unet_rgb_384_smallmask", "low_false_alarm", "EfficientNet 384", 384),
]


def first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_dataset_root(user_path: Optional[str]) -> Path:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    candidates += [
        Path("/kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection"),
        Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection"),
        Path("/content/drive/MyDrive/bitirmeProjesi/dataset"),
        Path("dataset"),
    ]
    root = first_existing(candidates)
    if root is None:
        raise FileNotFoundError("Dataset root bulunamadi. --dataset-root ile verin.")
    return root


def discover_experiments_root(user_path: Optional[str]) -> Path:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    candidates += [
        Path("/kaggle/working/experiments_full"),
        Path("experiments_full"),
        Path("final_analysis/experiments_full"),
    ]
    root = first_existing(candidates)
    if root is None:
        raise FileNotFoundError("experiments_full bulunamadi. --experiments-root ile verin.")
    return root


def resolve_exp5_root(experiments_root: Path) -> Path:
    root = first_existing(
        [
            experiments_root / "experiment5_calibration_postprocessing",
            Path("deney_5/experiments_4_full/experiment5_calibration_postprocessing"),
        ]
    )
    if root is None:
        raise FileNotFoundError("Experiment 5 root bulunamadi.")
    return root


def resolve_exp6_root(experiments_root: Path) -> Path:
    root = first_existing(
        [
            experiments_root / "experiment6_smallmask_384",
            Path("deney_6/experiments_full/experiment6_smallmask_384"),
        ]
    )
    if root is None:
        raise FileNotFoundError("Experiment 6 root bulunamadi.")
    return root


def case_key(row: pd.Series) -> str:
    return f"{str(row['class_name'])}__{str(row['image_id'])}"


def repair_image_path(row: pd.Series, dataset_root: Path) -> Path:
    path = Path(str(row.get("image_path", "")))
    if path.exists():
        return path
    return dataset_root / "train_images" / str(row["class_name"]) / f"{row['image_id']}.png"


def mask_path_for_row(row: pd.Series, dataset_root: Path) -> str:
    if str(row.get("class_name", "")) != "forged":
        return ""
    for value in (row.get("mask_path", ""), row.get("mask_paths", "")):
        if not value or str(value) == "nan":
            continue
        first = str(value).split("|")[0]
        if Path(first).exists():
            return first
    candidate = dataset_root / "train_masks" / f"{row['image_id']}.npy"
    return str(candidate)


def load_rgb_image(path: Path) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV gerekli. Kaggle'da opencv-python-headless kurulu olmali.")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_gt_mask(mask_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV gerekli. Kaggle'da opencv-python-headless kurulu olmali.")
    if not mask_path:
        return np.zeros(image_shape, dtype=np.uint8)
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Maske bulunamadi: {path}")
    mask = np.load(path)
    if mask.ndim == 2:
        mask = mask > 0
    elif mask.ndim == 3 and mask.shape[0] <= 16:
        mask = np.any(mask > 0, axis=0)
    elif mask.ndim == 3:
        mask = np.any(mask > 0, axis=-1)
    else:
        raise ValueError(f"Beklenmeyen maske boyutu: {path}, shape={mask.shape}")
    if mask.shape != image_shape:
        mask = cv2.resize(mask.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask.astype(np.uint8)


def component_table(binary: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    if cv2 is None:
        raise ImportError("OpenCV gerekli. Kaggle'da opencv-python-headless kurulu olmali.")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    comps: List[Dict[str, float]] = []
    total = float(binary.size)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        values = prob[labels == label]
        comps.append(
            {
                "label": int(label),
                "area": area,
                "area_ratio": float(area / max(total, 1.0)),
                "mean_probability": float(values.mean()) if values.size else 0.0,
            }
        )
    return labels, comps


def apply_morphology(mask: np.ndarray, morphology: str, kernel_size: int) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV gerekli. Kaggle'da opencv-python-headless kurulu olmali.")
    if morphology == "none":
        return mask.astype(np.uint8)
    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    out = mask.astype(np.uint8)
    if morphology in ("open", "open_close"):
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if morphology in ("close", "open_close"):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out.astype(np.uint8)


def sanitize_config(row: pd.Series) -> Dict[str, object]:
    def num(name: str, default: float) -> float:
        value = row.get(name, default)
        if pd.isna(value) or value == "":
            return default
        return float(value)

    top_k = row.get("top_k_components", None)
    if top_k is None or pd.isna(top_k) or top_k == "":
        top_k_clean = None
    else:
        top_k_clean = int(float(top_k))
    return {
        "pixel_threshold": num("pixel_threshold", 0.5),
        "postprocess_mode": str(row.get("postprocess_mode", "raw") or "raw"),
        "min_component_area": int(num("min_component_area", 0)),
        "min_component_mean_probability": num("min_component_mean_probability", 0.0),
        "morphology": str(row.get("morphology", "none") or "none"),
        "morph_kernel_size": int(num("morph_kernel_size", 3)),
        "top_k_components": top_k_clean,
        "top_k_sort_by": str(row.get("top_k_sort_by", "area") or "area"),
    }


def postprocess_probability_map(prob: np.ndarray, config: Dict[str, object]) -> np.ndarray:
    raw = (prob >= float(config["pixel_threshold"])).astype(np.uint8)
    work = raw.copy()
    mode = str(config["postprocess_mode"])
    if mode == "raw":
        return work
    if mode == "morph_area_probability_clean":
        work = apply_morphology(work, str(config["morphology"]), int(config["morph_kernel_size"]))
    labels, comps = component_table(work, prob)
    selected = []
    for comp in comps:
        ok = True
        if mode in ("min_area_clean", "area_probability_clean", "morph_area_probability_clean", "keep_topk_components"):
            ok = ok and comp["area"] >= int(config["min_component_area"])
        if mode in ("probability_gated", "area_probability_clean", "morph_area_probability_clean", "keep_topk_components"):
            ok = ok and comp["mean_probability"] >= float(config["min_component_mean_probability"])
        if ok:
            selected.append(comp)
    if mode == "keep_topk_components" and config["top_k_components"] is not None:
        key = "mean_probability" if config["top_k_sort_by"] == "mean_probability" else "area"
        selected = sorted(selected, key=lambda x: x[key], reverse=True)[: int(config["top_k_components"])]
    out = np.zeros_like(work, dtype=np.uint8)
    for comp in selected:
        out[labels == comp["label"]] = 1
    return out


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    return float((2 * np.logical_and(pred, gt).sum() + EPS) / (pred.sum() + gt.sum() + EPS))


def load_run(run: RunSpec, exp5_root: Path, exp6_root: Path) -> Dict[str, object]:
    base = exp5_root / run.model_dir if run.resolution == 256 else exp6_root / run.model_dir
    selected = pd.read_csv(base / "selected_configs.csv")
    strategy_rows = selected[selected["strategy"].astype(str) == run.strategy]
    if strategy_rows.empty:
        raise ValueError(f"{base}: strategy bulunamadi: {run.strategy}")
    config = sanitize_config(strategy_rows.iloc[0])
    metadata = pd.read_csv(base / "test_metadata.csv")
    metadata["image_id"] = metadata["image_id"].astype(str)
    metadata["case_key"] = metadata.apply(case_key, axis=1)
    probs_npz = np.load(base / "test_prob_maps.npz")
    probs = probs_npz["probs"]
    if "prob_index" in metadata.columns:
        metadata["prob_index"] = metadata["prob_index"].astype(int)
    else:
        metadata["prob_index"] = np.arange(len(metadata), dtype=int)
    by_key = {row.case_key: int(row.prob_index) for row in metadata.itertuples(index=False)}
    return {"base": base, "config": config, "metadata": metadata, "probs": probs, "by_key": by_key}


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.copy().astype(np.float32)
    mask_bool = mask.astype(bool)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray, fallback_shape: Tuple[int, int], pad: int = 40) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = fallback_shape
    if not len(xs):
        side = min(h, w, 420)
        y0 = max(0, h // 2 - side // 2)
        x0 = max(0, w // 2 - side // 2)
        return x0, y0, min(w, x0 + side), min(h, y0 + side)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    min_side = 220
    if x1 - x0 < min_side:
        extra = min_side - (x1 - x0)
        x0 = max(0, x0 - extra // 2)
        x1 = min(w, x1 + extra - extra // 2)
    if y1 - y0 < min_side:
        extra = min_side - (y1 - y0)
        y0 = max(0, y0 - extra // 2)
        y1 = min(h, y1 + extra - extra // 2)
    return x0, y0, x1, y1


def crop(arr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def resize_panel(arr: np.ndarray, size: int = 220, mask: bool = False) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV gerekli. Kaggle'da opencv-python-headless kurulu olmali.")
    interp = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    return cv2.resize(arr, (size, size), interpolation=interp)


def prepare_case(row: pd.Series, dataset_root: Path, runs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    if cv2 is None:
        raise ImportError("OpenCV gerekli. Kaggle'da opencv-python-headless kurulu olmali.")
    key = str(row["case_key"])
    image = load_rgb_image(repair_image_path(row, dataset_root))
    gt = load_gt_mask(mask_path_for_row(row, dataset_root), image.shape[:2])
    result: Dict[str, object] = {"case_key": key, "row": row, "image": image, "gt": gt, "preds": {}, "metrics": {}}
    for spec in RUNS:
        run = runs[spec.key]
        prob_index = run["by_key"].get(key)
        if prob_index is None:
            raise KeyError(f"{spec.label} icin case bulunamadi: {key}")
        prob = np.asarray(run["probs"][prob_index], dtype=np.float32)
        pred_eval = postprocess_probability_map(prob, run["config"])
        gt_eval = cv2.resize(gt, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_orig = cv2.resize(pred_eval, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        result["preds"][spec.key] = pred_orig.astype(np.uint8)
        result["metrics"][spec.key] = {
            "dice": dice_score(pred_eval, gt_eval),
            "pred_area_ratio": float(pred_eval.mean()),
        }
    return result


def choose_cases(base_df: pd.DataFrame, runs: Dict[str, Dict[str, object]], dataset_root: Path, kind: str, n: int) -> List[Dict[str, object]]:
    if kind == "forged":
        candidates = base_df[(base_df["class_name"] == "forged") & (base_df["gt_area_ratio"].astype(float) > 0)].copy()
        candidates = candidates.sort_values(["mask_quartile", "gt_area_ratio"], ascending=[True, True]).head(120)
    else:
        candidates = base_df[base_df["class_name"] == "authentic"].copy().head(220)
    scored = []
    for _, row in candidates.iterrows():
        try:
            case = prepare_case(row, dataset_root, runs)
        except Exception:
            continue
        metrics = case["metrics"]
        if kind == "forged":
            seg_delta = metrics["seg384"]["dice"] - metrics["seg256"]["dice"]
            eff_delta = metrics["eff384"]["dice"] - metrics["eff256"]["dice"]
            score = seg_delta + eff_delta + 0.15 * (metrics["seg384"]["dice"] + metrics["eff384"]["dice"])
            if max(metrics["seg384"]["dice"], metrics["eff384"]["dice"]) < 0.05:
                continue
        else:
            area256 = metrics["seg256"]["pred_area_ratio"] + metrics["eff256"]["pred_area_ratio"]
            area384 = metrics["seg384"]["pred_area_ratio"] + metrics["eff384"]["pred_area_ratio"]
            score = area256 + area384
            if score <= 0:
                continue
        case["selection_score"] = float(score)
        scored.append(case)
    scored = sorted(scored, key=lambda item: item["selection_score"], reverse=True)
    return scored[:n]


def save_case_grid(cases: List[Dict[str, object]], out_path: Path, title: str, authentic: bool = False) -> pd.DataFrame:
    if not cases:
        raise RuntimeError(f"Gorsel icin uygun case bulunamadi: {out_path}")
    cols = ["Original crop", "GT mask", "SegFormer 256", "SegFormer 384", "EffNet 256", "EffNet 384"]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(len(cols) * 2.35, len(cases) * 2.45), squeeze=False)
    manifest = []
    for r, case in enumerate(cases):
        image = case["image"]
        gt = case["gt"]
        if authentic:
            union = np.zeros(gt.shape, dtype=np.uint8)
            for pred in case["preds"].values():
                union = np.maximum(union, pred.astype(np.uint8))
            box = bbox_from_mask(union, gt.shape, pad=60)
        else:
            box = bbox_from_mask(gt, gt.shape, pad=70)
        panels = [
            resize_panel(crop(image, box)),
            resize_panel(crop(overlay_mask(image, gt, (65, 105, 225), 0.60), box)),
        ]
        for key in ("seg256", "seg384", "eff256", "eff384"):
            pred = case["preds"][key]
            panels.append(resize_panel(crop(overlay_mask(image, pred, (230, 70, 70), 0.55), box)))
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11, fontweight="bold")
            if c >= 2:
                key = ("seg256", "seg384", "eff256", "eff384")[c - 2]
                m = case["metrics"][key]
                label = f"Dice {m['dice']:.2f}" if not authentic else f"FP area {100*m['pred_area_ratio']:.1f}%"
                ax.set_xlabel(label, fontsize=9)
        row = case["row"]
        axes[r, 0].set_ylabel(
            f"{row['class_name']} {row['image_id']}\nGT {100*float(row.get('gt_area_ratio', 0)):.2f}%",
            fontsize=9,
        )
        manifest_row = {
            "case_key": case["case_key"],
            "image_id": row["image_id"],
            "class_name": row["class_name"],
            "gt_area_ratio": float(row.get("gt_area_ratio", 0)),
            "selection_score": case["selection_score"],
        }
        for key in ("seg256", "seg384", "eff256", "eff384"):
            manifest_row[f"{key}_dice"] = case["metrics"][key]["dice"]
            manifest_row[f"{key}_pred_area_ratio"] = case["metrics"][key]["pred_area_ratio"]
        manifest.append(manifest_row)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return pd.DataFrame(manifest)


def save_summary_plot(out_dir: Path) -> Optional[Path]:
    comparison_path = first_existing(
        [
            Path("analysis_review/experiment_6_vs_experiment_5_reconstructed_comparison.csv"),
            Path("deney_6/experiments_full/experiment6_smallmask_384/experiment6_vs_experiment5_comparison.csv"),
        ]
    )
    if comparison_path is None:
        return None
    df = pd.read_csv(comparison_path)
    if "model" in df.columns:
        keep = df[
            ((df["model"].astype(str).str.contains("SegFormer")) & (df["exp6_strategy"] == "balanced_final_score"))
            | ((df["model"].astype(str).str.contains("EfficientNet")) & (df["exp6_strategy"] == "low_false_alarm"))
        ].copy()
        labels = keep["model"].astype(str).str.replace(" 384", "", regex=False)
        exp5_q1 = keep["exp5_q1_dice"].astype(float)
        exp6_q1 = keep["exp6_q1_dice"].astype(float)
        exp5_q2 = keep["exp5_q2_dice"].astype(float)
        exp6_q2 = keep["exp6_q2_dice"].astype(float)
    else:
        keep = df[
            ((df["experiment6_model"].astype(str).str.contains("segformer")) & (df["experiment6_strategy"] == "balanced_final_score"))
            | ((df["experiment6_model"].astype(str).str.contains("efficientnet")) & (df["experiment6_strategy"] == "low_false_alarm"))
        ].copy()
        labels = keep["experiment6_model"].astype(str).str.replace("_rgb_384_smallmask", "", regex=False)
        exp5_q1 = keep["exp5_Q1 Dice"].astype(float)
        exp6_q1 = keep["exp6_Q1 Dice"].astype(float)
        exp5_q2 = keep["exp5_Q2 Dice"].astype(float)
        exp6_q2 = keep["exp6_Q2 Dice"].astype(float)

    x = np.arange(len(keep))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(x - 1.5 * width, exp5_q1, width, label="Q1 256", color="#b9c5d8")
    ax.bar(x - 0.5 * width, exp6_q1, width, label="Q1 384", color="#3b6fb6")
    ax.bar(x + 0.5 * width, exp5_q2, width, label="Q2 256", color="#e5b8a0")
    ax.bar(x + 1.5 * width, exp6_q2, width, label="Q2 384", color="#d0673f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, max(0.45, float(np.nanmax([exp5_q1.max(), exp6_q1.max(), exp5_q2.max(), exp6_q2.max()])) + 0.08))
    ax.set_title("Small-mask Dice: 256x256 vs 384x384")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    out_path = out_dir / "smallmask_q1_q2_resolution_summary.png"
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--out-dir", default="analysis_review/smallmask_resolution_visuals")
    parser.add_argument("--num-forged", type=int, default=3)
    parser.add_argument("--num-authentic", type=int, default=2)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_summary_plot(out_dir)

    if args.summary_only:
        print(f"[ok] Summary plot: {summary_path}")
        return

    dataset_root = discover_dataset_root(args.dataset_root)
    experiments_root = discover_experiments_root(args.experiments_root)
    exp5_root = resolve_exp5_root(experiments_root)
    exp6_root = resolve_exp6_root(experiments_root)
    runs = {spec.key: load_run(spec, exp5_root, exp6_root) for spec in RUNS}

    base_df = runs["seg384"]["metadata"].copy()
    base_df["image_path"] = base_df.apply(lambda r: str(repair_image_path(r, dataset_root)), axis=1)
    base_df["mask_path"] = base_df.apply(lambda r: mask_path_for_row(r, dataset_root), axis=1)

    forged_cases = choose_cases(base_df, runs, dataset_root, "forged", args.num_forged)
    forged_manifest = save_case_grid(
        forged_cases,
        out_dir / "forged_smallmask_256_vs_384.png",
        "Same forged small-mask cases: 256x256 vs 384x384 predictions",
        authentic=False,
    )
    forged_manifest.to_csv(out_dir / "forged_smallmask_256_vs_384_cases.csv", index=False)

    authentic_cases = choose_cases(base_df, runs, dataset_root, "authentic", args.num_authentic)
    authentic_manifest = save_case_grid(
        authentic_cases,
        out_dir / "authentic_fp_256_vs_384.png",
        "Same authentic cases: false-positive behavior at 256x256 vs 384x384",
        authentic=True,
    )
    authentic_manifest.to_csv(out_dir / "authentic_fp_256_vs_384_cases.csv", index=False)

    print("[ok] Outputs:")
    for path in [
        summary_path,
        out_dir / "forged_smallmask_256_vs_384.png",
        out_dir / "authentic_fp_256_vs_384.png",
        out_dir / "forged_smallmask_256_vs_384_cases.csv",
        out_dir / "authentic_fp_256_vs_384_cases.csv",
    ]:
        if path is not None:
            print(" -", path)


if __name__ == "__main__":
    main()
