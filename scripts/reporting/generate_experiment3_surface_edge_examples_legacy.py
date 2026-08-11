#!/usr/bin/env python
"""Generate forged-focused surface vs edge-enhanced qualitative grids for Experiment 3.

Run this on the machine/Kaggle runtime that has both the ReCodAI-LUC dataset and
the Experiment 3 checkpoint folders. It does not retrain models; it only reloads
the saved two-output U-Net++ checkpoints and re-runs inference on forged test rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "segmentation_models_pytorch is required. Run this in the same Kaggle/Colab "
        "environment used for Experiment 3, or install the project requirements."
    ) from exc


DEFAULT_DATASET_ROOT = Path(
    "/kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection"
    "/kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection"
)
DEFAULT_EXPERIMENTS_ROOT = Path("experiments")
EDGE_MODELS = [
    "unetpp_resnet34_rgb_srm_edge_multitask",
    "unetpp_resnet50_rgb_srm_edge_multitask",
]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SRM_KERNELS = [
    np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
]


@dataclass
class ExperimentSpec:
    name: str
    model_name: str
    encoder_name: str
    input_mode: str
    output_channels: int
    encoder_weights: Optional[str]
    image_size: int
    edge_enhance_weight: float


def load_spec(exp_dir: Path) -> ExperimentSpec:
    with (exp_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    global_cfg = cfg["global"]
    exp_cfg = cfg["experiment"]
    return ExperimentSpec(
        name=exp_cfg["name"],
        model_name=exp_cfg.get("model_name", "UnetPlusPlus"),
        encoder_name=exp_cfg["encoder_name"],
        input_mode=exp_cfg["input_mode"],
        output_channels=int(exp_cfg["output_channels"]),
        encoder_weights=exp_cfg.get("encoder_weights"),
        image_size=int(global_cfg.get("image_size", 256)),
        edge_enhance_weight=float(global_cfg.get("edge_enhance_weight", 0.2)),
    )


def input_channels(input_mode: str) -> int:
    if input_mode == "rgb":
        return 3
    if input_mode == "rgb_srm":
        return 8
    if input_mode == "rgb_srm_ela":
        return 9
    raise ValueError(f"Unknown input_mode: {input_mode}")


def create_model(spec: ExperimentSpec) -> nn.Module:
    model_name = spec.model_name.lower()
    if model_name not in {"unetplusplus", "unet++", "unet"}:
        raise ValueError(f"Unsupported model_name: {spec.model_name}")
    model_cls = smp.UnetPlusPlus if model_name in {"unetplusplus", "unet++"} else smp.Unet
    return model_cls(
        encoder_name=spec.encoder_name,
        encoder_weights=None,
        in_channels=input_channels(spec.input_mode),
        classes=spec.output_channels,
        activation=None,
    )


def split_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if logits.shape[1] == 1:
        return logits[:, :1], None
    return logits[:, :1], logits[:, 1:2]


def resolve_path(raw_path: str, dataset_root: Path) -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path
    parts = path.parts
    if "train_images" in parts:
        idx = parts.index("train_images")
        return dataset_root.joinpath(*parts[idx:])
    if "train_masks" in parts:
        idx = parts.index("train_masks")
        return dataset_root.joinpath(*parts[idx:])
    return path


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found/readable: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def binarize_loaded_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return mask > 0
    if mask.ndim == 3:
        if mask.shape[0] <= 16 and mask.shape[1] > 16 and mask.shape[2] > 16:
            return np.any(mask > 0, axis=0)
        return np.any(mask > 0, axis=-1)
    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def load_mask(path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    raw = np.load(path, allow_pickle=False)
    mask = binarize_loaded_mask(raw).astype(np.uint8)
    h, w = target_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def extract_srm_features(rgb_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    features = []
    for kernel in SRM_KERNELS:
        residual = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        normalized = (residual - float(residual.mean())) / (float(residual.std()) + 1e-6)
        normalized = np.clip(normalized, -5.0, 5.0) / 5.0
        features.append(normalized.astype(np.float32))
    return np.stack(features, axis=-1)


def preprocess_input(rgb_image: np.ndarray, input_mode: str) -> np.ndarray:
    rgb = rgb_image.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    if input_mode == "rgb":
        stacked = rgb
    elif input_mode == "rgb_srm":
        stacked = np.concatenate([rgb, extract_srm_features(rgb_image)], axis=-1)
    else:
        raise ValueError(f"Unsupported input_mode for this figure script: {input_mode}")
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    denom = int(gt_bool.sum()) + int(pred_bool.sum())
    if denom == 0:
        return 1.0
    return float((2 * np.logical_and(gt_bool, pred_bool).sum()) / denom)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[mask > 0] = (1 - alpha) * out[mask > 0] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def infer_record(
    row: pd.Series,
    model: nn.Module,
    spec: ExperimentSpec,
    dataset_root: Path,
    threshold_surface: float,
    threshold_edge: float,
    device: torch.device,
) -> Dict[str, object]:
    image_path = resolve_path(str(row["image_path"]), dataset_root)
    mask_path = resolve_path(str(row["mask_path"]), dataset_root)
    rgb = load_rgb(image_path)
    gt = load_mask(mask_path, rgb.shape[:2])
    rgb_resized = cv2.resize(rgb, (spec.image_size, spec.image_size), interpolation=cv2.INTER_LINEAR)
    gt_resized = cv2.resize(gt, (spec.image_size, spec.image_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    x = torch.from_numpy(preprocess_input(rgb_resized, spec.input_mode))[None].to(device)
    with torch.no_grad():
        logits = model(x)
        mask_logits, edge_logits = split_logits(logits)
        surface_prob = torch.sigmoid(mask_logits)[0, 0].detach().cpu().numpy()
        if edge_logits is None:
            raise RuntimeError(f"{spec.name} does not have an edge output.")
        edge_prob = torch.sigmoid(edge_logits)[0, 0].detach().cpu().numpy()

    edge_enhanced_prob = np.clip(surface_prob * (1.0 + spec.edge_enhance_weight * edge_prob), 0.0, 1.0)
    surface_mask = (surface_prob >= threshold_surface).astype(np.uint8)
    edge_mask = (edge_enhanced_prob >= threshold_edge).astype(np.uint8)
    surface_dice = dice_score(gt_resized, surface_mask)
    edge_dice = dice_score(gt_resized, edge_mask)

    return {
        "sample_id": row["sample_id"],
        "image_id": row["image_id"],
        "rgb": rgb_resized,
        "gt": gt_resized,
        "surface_prob": surface_prob,
        "edge_prob": edge_prob,
        "edge_enhanced_prob": edge_enhanced_prob,
        "surface_mask": surface_mask,
        "edge_mask": edge_mask,
        "surface_overlay": overlay_mask(rgb_resized, surface_mask, (0, 120, 255)),
        "edge_overlay": overlay_mask(rgb_resized, edge_mask, (255, 0, 0)),
        "gt_area": int(gt_resized.sum()),
        "surface_area": int(surface_mask.sum()),
        "edge_area": int(edge_mask.sum()),
        "surface_dice": surface_dice,
        "edge_dice": edge_dice,
        "dice_delta_edge_minus_surface": edge_dice - surface_dice,
        "area_delta_edge_minus_surface": int(edge_mask.sum()) - int(surface_mask.sum()),
        "change_pixels": int(np.logical_xor(surface_mask, edge_mask).sum()),
    }


def plot_grid(records: List[Dict[str, object]], out_path: Path, title: str) -> None:
    columns = [
        ("rgb", None, "Image"),
        ("gt", "gray", "GT"),
        ("surface_prob", "magma", "Surface prob"),
        ("edge_prob", "magma", "Edge prob"),
        ("edge_enhanced_prob", "magma", "Edge-enhanced prob"),
        ("surface_mask", "gray", "Surface mask"),
        ("edge_mask", "gray", "Edge-enhanced mask"),
        ("surface_overlay", None, "Surface overlay"),
        ("edge_overlay", None, "Edge overlay"),
    ]
    fig, axes = plt.subplots(len(records), len(columns), figsize=(2.35 * len(columns), 2.45 * len(records)))
    if len(records) == 1:
        axes = axes[None, :]
    for c, (_, _, label) in enumerate(columns):
        axes[0, c].set_title(label, fontsize=9)
    for r, rec in enumerate(records):
        for c, (key, cmap, _) in enumerate(columns):
            kwargs = {}
            if key.endswith("_prob"):
                kwargs = {"vmin": 0.0, "vmax": 1.0}
            axes[r, c].imshow(rec[key], cmap=cmap, **kwargs)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(
            f"{rec['sample_id']}\nS={rec['surface_dice']:.3f} E={rec['edge_dice']:.3f}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_thresholds(exp_dir: Path) -> Tuple[float, float]:
    metrics = pd.read_csv(exp_dir / "test_metrics.csv")
    surface = metrics.loc[metrics["inference_mode"] == "surface", "threshold"]
    edge = metrics.loc[metrics["inference_mode"] == "edge_enhanced", "threshold"]
    if surface.empty or edge.empty:
        raise ValueError(f"Missing surface/edge_enhanced rows in {exp_dir / 'test_metrics.csv'}")
    return float(surface.iloc[0]), float(edge.iloc[0])


def write_model_figures(
    model_name: str,
    experiments_root: Path,
    dataset_root: Path,
    out_dir: Path,
    max_candidates: int,
    top_k: int,
    device: torch.device,
) -> None:
    exp_dir = experiments_root / model_name
    spec = load_spec(exp_dir)
    threshold_surface, threshold_edge = load_thresholds(exp_dir)

    model = create_model(spec).to(device)
    ckpt = torch.load(exp_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = pd.read_csv(exp_dir / "test_per_image_metrics.csv")
    forged = rows[rows["label"].astype(int) == 1].copy()
    forged["gt_area"] = forged["gt_area"].astype(int)
    forged = forged.sort_values(["gt_area", "dice"], ascending=[True, True])
    if max_candidates > 0:
        forged = forged.head(max_candidates)

    records = []
    for _, row in forged.iterrows():
        try:
            records.append(
                infer_record(row, model, spec, dataset_root, threshold_surface, threshold_edge, device)
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                f"{exc}\nDataset root may be wrong. Pass --dataset-root /path/to/recodai-luc-scientific-image-forgery-detection"
            ) from exc

    ranked_change = sorted(records, key=lambda x: (x["change_pixels"], abs(x["dice_delta_edge_minus_surface"])), reverse=True)
    selected_change = ranked_change[:top_k]
    selected_small = sorted(records, key=lambda x: (x["gt_area"], -x["change_pixels"]))[:top_k]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_cols = [
        "sample_id",
        "image_id",
        "gt_area",
        "surface_area",
        "edge_area",
        "area_delta_edge_minus_surface",
        "change_pixels",
        "surface_dice",
        "edge_dice",
        "dice_delta_edge_minus_surface",
    ]
    pd.DataFrame([{k: rec[k] for k in summary_cols} for rec in records]).sort_values(
        ["change_pixels", "gt_area"], ascending=[False, True]
    ).to_csv(out_dir / f"{model_name}_surface_edge_candidate_summary.csv", index=False)

    plot_grid(
        selected_change,
        out_dir / f"{model_name}_surface_edge_most_changed.png",
        f"{model_name}: forged examples where edge branch changes the mask most",
    )
    plot_grid(
        selected_small,
        out_dir / f"{model_name}_surface_edge_small_masks.png",
        f"{model_name}: small forged masks, surface vs edge-enhanced",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_review/experiment3_surface_edge_examples"))
    parser.add_argument("--models", nargs="+", default=EDGE_MODELS)
    parser.add_argument("--max-candidates", type=int, default=0, help="Forged rows to scan; 0 scans all forged rows.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for model_name in args.models:
        print(f"[info] generating surface/edge examples for {model_name} on {device}")
        write_model_figures(
            model_name=model_name,
            experiments_root=args.experiments_root,
            dataset_root=args.dataset_root,
            out_dir=args.out_dir,
            max_candidates=args.max_candidates,
            top_k=args.top_k,
            device=device,
        )
    print(f"[done] wrote figures and CSV summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
