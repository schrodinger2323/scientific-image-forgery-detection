"""
Deney 6 gorsel grid duzeltme script'i.

Bu dosya Kaggle uzerinde calistirilmak icin hazirlanmistir. Mevcut Deney 6
ciktilarindaki GT/raw/final mask panelleri uint8 0/1 olarak cizildigi icin
neredeyse siyah gorunebiliyor. Bu script ayni gorselleri maskeleri float32 0/1
olarak cizerek yeniden uretir ve /kaggle/working altindaki Deney 6 klasorune
kaydeder.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_SIZE = 384
MAX_VISUAL_EXAMPLES = 12
STRATEGY = "balanced_final_score"
BACKUP_OLD_PNGS = True

EXP_ROOT = Path("/kaggle/working/experiments_full/experiment6_smallmask_384")
DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection"),
    Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection"),
    Path("/content/drive/MyDrive/bitirmeProjesi/dataset"),
]

MODEL_NAMES = [
    "efficientnetb0_unet_rgb_384_smallmask",
    "segformer_b0_rgb_384_smallmask",
]

GRID_OUTPUTS = [
    "prediction_examples_best_strategy.png",
    "q1_best_cases.png",
    "q1_worst_cases.png",
    "q1_false_negative_cases.png",
]

FAILURE_GROUPS = [
    "best_cases_forged",
    "low_dice_forged",
    "false_positive_authentic",
    "false_negative_forged",
    "small_mask_failures",
    "large_mask_failures",
]


def find_data_root() -> Optional[Path]:
    for root in DATA_ROOT_CANDIDATES:
        if (root / "train_images").exists():
            return root
    return None


DATA_ROOT = find_data_root()


def as_clean_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if pd.isna(value):
        return default
    return value


def load_image_rgb(path: str, class_name: str, image_id: str) -> np.ndarray:
    image_path = Path(str(path))
    if not image_path.exists() and DATA_ROOT is not None:
        image_path = DATA_ROOT / "train_images" / str(class_name) / f"{image_id}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def canonical_mask_path(mask_path: Any, image_id: str, class_name: str) -> Optional[Path]:
    if str(class_name) != "forged":
        return None
    if mask_path is not None and not pd.isna(mask_path) and str(mask_path).strip():
        candidate = Path(str(mask_path))
        if candidate.exists():
            return candidate
    if DATA_ROOT is not None:
        candidate = DATA_ROOT / "train_masks" / f"{image_id}.npy"
        if candidate.exists():
            return candidate
    return None


def load_mask_array(mask_path: Any, image_shape: Tuple[int, int], image_id: str, class_name: str) -> np.ndarray:
    path = canonical_mask_path(mask_path, image_id, class_name)
    if path is None:
        return np.zeros(image_shape, dtype=np.float32)
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
    return mask.astype(np.float32)


def component_table(binary: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    components: List[Dict[str, Any]] = []
    total_area = float(binary.size)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        values = prob[labels == label]
        components.append(
            {
                "label": int(label),
                "area": area,
                "area_ratio": float(area / max(total_area, 1.0)),
                "mean_probability": float(values.mean()) if values.size else 0.0,
                "max_probability": float(values.max()) if values.size else 0.0,
            }
        )
    return labels, components


def apply_morphology(mask: np.ndarray, morphology: str, kernel_size: int) -> np.ndarray:
    if morphology == "none":
        return mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out = mask.astype(np.uint8)
    if morphology in ("open", "open_close"):
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if morphology in ("close", "open_close"):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out.astype(np.uint8)


def postprocess_probability_map(prob: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    threshold = float(as_clean_value(config.get("pixel_threshold"), 0.5))
    raw = (prob >= threshold).astype(np.uint8)
    work = raw.copy()

    mode = str(as_clean_value(config.get("postprocess_mode"), "raw"))
    min_area = int(as_clean_value(config.get("min_component_area"), 0) or 0)
    min_mean_prob = float(as_clean_value(config.get("min_component_mean_probability"), 0.0) or 0.0)
    morphology = str(as_clean_value(config.get("morphology"), "none"))
    kernel_size = int(as_clean_value(config.get("morph_kernel_size"), as_clean_value(config.get("morph_kernel_size"), 3)) or 3)
    top_k = as_clean_value(config.get("top_k_components"), None)
    sort_by = str(as_clean_value(config.get("top_k_sort_by"), "area"))

    if mode == "raw":
        labels, components = component_table(work, prob)
        return {"raw_mask": raw, "mask": work, "labels": labels, "components": components}

    if mode == "morph_area_probability_clean":
        work = apply_morphology(work, morphology, kernel_size)

    labels, components = component_table(work, prob)
    keep = np.zeros_like(work, dtype=np.uint8)
    selected: List[Dict[str, Any]] = []
    for comp in components:
        ok = True
        if mode in ("min_area_clean", "area_probability_clean", "morph_area_probability_clean", "keep_topk_components"):
            ok = ok and comp["area"] >= min_area
        if mode in ("probability_gated", "area_probability_clean", "morph_area_probability_clean", "keep_topk_components"):
            ok = ok and comp["mean_probability"] >= min_mean_prob
        if ok:
            selected.append(comp)

    if mode == "keep_topk_components" and top_k is not None:
        key = "mean_probability" if sort_by == "mean_probability" else "area"
        selected = sorted(selected, key=lambda item: item[key], reverse=True)[: int(top_k)]

    for comp in selected:
        keep[labels == comp["label"]] = 1
    labels_clean, components_clean = component_table(keep, prob)
    return {"raw_mask": raw, "mask": keep, "labels": labels_clean, "components": components_clean}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask_float = (mask > 0).astype(np.float32)
    return np.repeat(mask_float[..., None], 3, axis=2)


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    out = image.copy()
    mask_bool = mask.astype(bool)
    out[mask_bool] = 0.55 * out[mask_bool] + 0.45 * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 1)


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    out = np.zeros((*gt_bool.shape, 3), dtype=np.float32)
    out[np.logical_and(gt_bool, pred_bool)] = [0.0, 0.85, 0.15]
    out[np.logical_and(~gt_bool, pred_bool)] = [1.0, 0.15, 0.15]
    out[np.logical_and(gt_bool, ~pred_bool)] = [0.15, 0.35, 1.0]
    return out


def load_prob_array(model_dir: Path) -> np.ndarray:
    data = np.load(model_dir / "test_prob_maps.npz", allow_pickle=True)
    if "probs" in data.files:
        return data["probs"].astype(np.float32)
    for key in data.files:
        arr = data[key]
        if arr.ndim == 3:
            return arr.astype(np.float32)
    raise KeyError(f"test_prob_maps.npz icinde 3 boyutlu probability array bulunamadi: {model_dir}")


def load_selected_config(model_dir: Path, strategy: str) -> Dict[str, Any]:
    selected = pd.read_csv(model_dir / "selected_configs.csv")
    row = selected[selected["strategy"].astype(str) == strategy]
    if row.empty:
        row = selected.iloc[[0]]
    return row.iloc[0].to_dict()


def sample_key(class_name: Any, image_id: Any) -> str:
    return f"{str(class_name)}__{str(image_id)}"


def dataframe_keys(df: pd.DataFrame) -> List[str]:
    return [sample_key(row["class_name"], row["image_id"]) for _, row in df.iterrows()]


def make_records_for_keys(model_dir: Path, sample_keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    metadata = pd.read_csv(model_dir / "test_metadata.csv")
    probs = load_prob_array(model_dir)
    needed = {str(key) for key in sample_keys}
    records: Dict[str, Dict[str, Any]] = {}
    for _, row in metadata.iterrows():
        image_id = str(row["image_id"])
        key = str(row["sample_id"]) if "sample_id" in row and not pd.isna(row["sample_id"]) else sample_key(row["class_name"], image_id)
        if key not in needed:
            continue
        image = load_image_rgb(row["image_path"], row["class_name"], image_id)
        h, w = image.shape[:2]
        mask = load_mask_array(row.get("mask_path", ""), (h, w), image_id, row["class_name"])
        mask = cv2.resize(mask.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        prob_index = int(row["prob_index"])
        records[key] = {
            "sample_key": key,
            "image_id": image_id,
            "class_name": str(row["class_name"]),
            "image": image,
            "mask": mask,
            "prob": probs[prob_index],
        }
    missing = needed - set(records)
    if missing:
        print(f"[UYARI] {model_dir.name}: metadata/prob icinde bulunamayan image_id sayisi: {len(missing)}")
    return records


def backup_png(path: Path) -> None:
    if not BACKUP_OLD_PNGS or not path.exists():
        return
    backup_dir = path.parent / "_old_visualization_pngs"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)


def save_prediction_grid(
    records_by_key: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    out_path: Path,
    sample_keys: Sequence[str],
    title: str,
) -> None:
    sample_keys = [str(key) for key in sample_keys if str(key) in records_by_key]
    if not sample_keys:
        print(f"[UYARI] Bos grid atlandi: {out_path}")
        return
    n = len(sample_keys)
    cols = 7
    fig, axes = plt.subplots(n, cols, figsize=(cols * 2.2, n * 2.0))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    subtitles = ["image", "gt", "prob", "raw", "final", "overlay", "error"]
    for r, key in enumerate(sample_keys):
        rec = records_by_key[key]
        out = postprocess_probability_map(rec["prob"], config)
        raw = out["raw_mask"]
        pred = out["mask"]
        gt = rec["mask"]
        image = rec["image"]
        panels = [
            image,
            mask_to_rgb(gt),
            plt.cm.magma(rec["prob"])[..., :3],
            mask_to_rgb(raw),
            mask_to_rgb(pred),
            make_overlay(image, pred, (1.0, 0.1, 0.1)),
            error_map(gt, pred),
        ]
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel, vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(subtitles[c], fontsize=9)
        axes[r, 0].set_ylabel(str(rec["sample_key"]), fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backup_png(out_path)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Kaydedildi: {out_path}")


def main_grid_keys(per_df: pd.DataFrame, grid_name: str) -> List[str]:
    forged = per_df[per_df["image_label"].astype(int) == 1].copy()
    q1 = forged[forged["mask_quartile"].astype(str) == "Q1"].copy()
    if grid_name == "prediction_examples_best_strategy.png":
        df = per_df.sort_values(["image_label", "dice"], ascending=[False, True]).head(MAX_VISUAL_EXAMPLES)
    elif grid_name == "q1_best_cases.png":
        df = q1.sort_values("dice", ascending=False).head(8)
    elif grid_name == "q1_worst_cases.png":
        df = q1.sort_values("dice", ascending=True).head(8)
    elif grid_name == "q1_false_negative_cases.png":
        df = q1[q1["image_pred_label"].astype(int) == 0].sort_values("dice", ascending=True).head(8)
    else:
        raise ValueError(f"Bilinmeyen grid adi: {grid_name}")
    return dataframe_keys(df)


def fix_model_visuals(model_name: str) -> List[Dict[str, Any]]:
    model_dir = EXP_ROOT / model_name
    if not model_dir.exists():
        print(f"[UYARI] Model klasoru bulunamadi, atlandi: {model_dir}")
        return []

    config = load_selected_config(model_dir, STRATEGY)
    per_df = pd.read_csv(model_dir / f"test_per_image_metrics_{STRATEGY}.csv")
    jobs: List[Tuple[Path, List[str], str]] = []

    for grid_name in GRID_OUTPUTS:
        keys = main_grid_keys(per_df, grid_name)
        title = f"{model_name} - {grid_name.replace('.png', '')}"
        jobs.append((model_dir / grid_name, keys, title))

    failure_dir = model_dir / "failure_cases"
    for group in FAILURE_GROUPS:
        csv_path = failure_dir / f"{group}.csv"
        if not csv_path.exists():
            continue
        group_df = pd.read_csv(csv_path)
        keys = dataframe_keys(group_df.head(8))
        jobs.append((failure_dir / f"{group}.png", keys, f"{model_name} - {group}"))

    all_keys = sorted({key for _, keys, _ in jobs for key in keys})
    records_by_key = make_records_for_keys(model_dir, all_keys)
    rows: List[Dict[str, Any]] = []
    for out_path, keys, title in jobs:
        save_prediction_grid(records_by_key, config, out_path, keys, title)
        rows.append(
            {
                "model_name": model_name,
                "output_path": str(out_path),
                "n_requested": len(keys),
                "n_drawn": sum(1 for key in keys if str(key) in records_by_key),
            }
        )
    return rows


def main() -> None:
    if not EXP_ROOT.exists():
        raise FileNotFoundError(f"Deney 6 klasoru bulunamadi: {EXP_ROOT}")
    if DATA_ROOT is None:
        print("[UYARI] Orijinal dataset klasoru bulunamadi. Metadata image_path degerleri erisilebilir degilse gorsel uretimi hata verebilir.")
    else:
        print(f"[INFO] Dataset kok klasoru: {DATA_ROOT}")

    all_rows: List[Dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        print(f"\n========== {model_name} ==========")
        all_rows.extend(fix_model_visuals(model_name))

    log_path = EXP_ROOT / "visualization_gt_fix_log.csv"
    pd.DataFrame(all_rows).to_csv(log_path, index=False)
    print(f"\nTamamlandi. Log dosyasi: {log_path}")


if __name__ == "__main__":
    main()
