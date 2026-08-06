"""
Recod.ai/LUC - Deney 4 evaluate-only devam dosyasi.

Bu dosya egitimi tekrar baslatmaz. Mevcut deney klasorlerindeki best_model.pth
checkpoint'lerini yukler, validation uzerinde threshold secer ve test metriklerini
hesaplayip eksik CSV/JSON raporlarini uretir.

Kaggle kullanim onerisi:
1) deney_4 klasorunu /kaggle/working altina ac:
   /kaggle/working/deney_4/experiments_4_full/...
   /kaggle/working/deney_4/experiments_full/...
2) Bu dosyayi notebook'a yukle veya bir hucrede olustur.
3) Calistir:
   %run /kaggle/working/recod_luc_4model_evaluate_existing_checkpoints.py
"""

import os
import sys
import json
import math
import time
import random
import platform
import warnings
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*The secret `HF_TOKEN` does not exist.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")


def is_kaggle_runtime() -> bool:
    return Path("/kaggle/input").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def ensure_package(import_name: str, pip_name: Optional[str] = None) -> None:
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"[install] {pip_name} kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure_package("cv2", "opencv-python-headless")
ensure_package("sklearn", "scikit-learn")
ensure_package("matplotlib")
ensure_package("tqdm")
ensure_package("albumentations")
ensure_package("segmentation_models_pytorch", "segmentation-models-pytorch")
ensure_package("transformers")
ensure_package("scipy")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import albumentations as A
import segmentation_models_pytorch as smp
from transformers import AutoModel, SegformerForSemanticSegmentation

try:
    from torch.amp import autocast
    TORCH_AMP_NEW_API = True
except Exception:
    from torch.cuda.amp import autocast
    TORCH_AMP_NEW_API = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class GlobalConfig:
    seed: int = 42
    image_size: int = 256
    dino_input_size: int = 252
    batch_size: int = 8
    dino_batch_size: int = 4
    num_workers: int = 0
    use_amp: bool = True
    save_prediction_probs: bool = True
    pixel_thresholds: Tuple[float, ...] = tuple(np.round(np.arange(0.10, 0.901, 0.05), 2))
    image_thresholds: Tuple[float, ...] = tuple(np.round(np.arange(0.00, 1.001, 0.01), 2))
    min_component_areas: Tuple[int, ...] = (0, 25, 50, 100, 200, 500)
    min_component_mean_probs: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    component_iou_thresholds: Tuple[float, ...] = (0.10, 0.25)
    primary_component_iou_threshold: float = 0.10
    image_score_methods: Tuple[str, ...] = ("max_probability", "pred_mask_ratio", "topk_mean_probability")
    topk_fraction: float = 0.01


@dataclass
class ExperimentConfig:
    name: str
    model_type: str
    model_family: str
    encoder_or_backbone: str
    input_mode: str = "rgb"
    in_channels: int = 3
    classes: int = 1
    encoder_weights: Optional[str] = "imagenet"
    hf_model_name: Optional[str] = None
    image_size: Optional[int] = None
    eval_size: Optional[int] = None
    batch_size: Optional[int] = None
    freeze_backbone_stage1: bool = False


CFG = GlobalConfig()

EXPERIMENTS = [
    ExperimentConfig(
        name="unetpp_resnet34_rgb_full",
        model_type="unetplusplus",
        model_family="CNN encoder-decoder baseline",
        encoder_or_backbone="resnet34",
        encoder_weights="imagenet",
    ),
    ExperimentConfig(
        name="efficientnetb0_unet_rgb_full",
        model_type="unet",
        model_family="parameter-efficient CNN transfer baseline",
        encoder_or_backbone="efficientnet-b0",
        encoder_weights="imagenet",
    ),
    ExperimentConfig(
        name="segformer_b0_rgb_full",
        model_type="segformer",
        model_family="transformer semantic segmentation",
        encoder_or_backbone="nvidia/segformer-b0-finetuned-ade-512-512",
        hf_model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        encoder_weights=None,
    ),
    ExperimentConfig(
        name="dinov2_lite_decoder_rgb_full",
        model_type="dinov2_lite_decoder",
        model_family="foundation-feature lightweight decoder",
        encoder_or_backbone="facebook/dinov2-small",
        hf_model_name="facebook/dinov2-small",
        encoder_weights=None,
        image_size=CFG.dino_input_size,
        eval_size=CFG.image_size,
        batch_size=CFG.dino_batch_size,
        freeze_backbone_stage1=True,
    ),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(CFG.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


def first_existing(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_paths() -> Tuple[Path, Path, Path, Path]:
    dataset_candidates = [
        Path("/kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection"),
        Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection"),
        Path("/kaggle/input/datasets/koushikkumardinda/scientific-image-forgery-detection/recodai-luc-scientific-image-forgery-detection"),
        Path("/kaggle/input/scientific-image-forgery-detection/recodai-luc-scientific-image-forgery-detection"),
        Path("/content/drive/MyDrive/bitirmeProjesi/dataset"),
    ]
    run_root_candidates = [
        Path("/kaggle/working/deney_4/experiments_4_full"),
        Path("/kaggle/working/experiments_4_full"),
        Path("deney_4/experiments_4_full"),
        Path("experiments_4_full"),
    ]
    comparison_candidates = [
        Path("/kaggle/working/deney_4/experiments_full"),
        Path("/kaggle/working/experiments_full"),
        Path("deney_4/experiments_full"),
        Path("experiments_full"),
    ]
    split_candidates = [
        Path("/kaggle/working/experiments/_shared_splits_seed42"),
        Path("/kaggle/working/_shared_splits_seed42"),
        Path("/kaggle/working/deney_4/experiments/_shared_splits_seed42"),
        Path("experiments/_shared_splits_seed42"),
        Path("deney_4/experiments/_shared_splits_seed42"),
    ]
    dataset_root = first_existing(dataset_candidates)
    if dataset_root is None:
        raise FileNotFoundError("Dataset root bulunamadi. Kaggle dataset path'ini kontrol et.")
    run_root = first_existing(run_root_candidates)
    if run_root is None:
        raise FileNotFoundError("experiments_4_full klasoru bulunamadi. deney_4 ciktisini /kaggle/working altina ac.")
    comparison_root = first_existing(comparison_candidates) or (run_root.parent / "experiments_full")
    comparison_root.mkdir(parents=True, exist_ok=True)
    split_dir = first_existing(split_candidates) or (run_root.parent / "_shared_splits_seed42")
    split_dir.mkdir(parents=True, exist_ok=True)
    return dataset_root, run_root, comparison_root, split_dir


DATASET_ROOT, RUN_ROOT, COMPARISON_ROOT, SHARED_SPLIT_DIR = discover_paths()
print("DATASET_ROOT:", DATASET_ROOT)
print("RUN_ROOT:", RUN_ROOT)
print("COMPARISON_ROOT:", COMPARISON_ROOT)
print("SHARED_SPLIT_DIR:", SHARED_SPLIT_DIR)


def image_id_from_path(path: Path) -> str:
    return path.stem


def find_mask_paths(image_id: str, mask_dir: Path) -> List[Path]:
    exact = mask_dir / f"{image_id}.npy"
    candidates = []
    if exact.exists():
        candidates.append(exact)
    for path in sorted(mask_dir.glob(f"{image_id}*.npy")):
        if path not in candidates:
            candidates.append(path)
    return candidates


def build_dataset_index(dataset_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for class_name, label in [("authentic", 0), ("forged", 1)]:
        for image_path in sorted((dataset_root / "train_images" / class_name).glob("*.png")):
            image_id = image_id_from_path(image_path)
            masks = find_mask_paths(image_id, dataset_root / "train_masks") if label == 1 else []
            rows.append({
                "sample_id": f"{class_name}__{image_id}",
                "image_id": str(image_id),
                "class_name": class_name,
                "image_label": label,
                "label": label,
                "image_path": str(image_path),
                "mask_paths": "|".join(str(p) for p in masks),
                "mask_path": str(masks[0]) if masks else "",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Dataset bos gorunuyor: {dataset_root}")
    return df.sort_values(["class_name", "image_id"]).reset_index(drop=True)


def repair_split_paths(df: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    df = df.copy()
    if "image_label" not in df.columns and "label" in df.columns:
        df["image_label"] = df["label"].astype(int)
    if "label" not in df.columns:
        df["label"] = df["image_label"].astype(int)
    if "class_name" not in df.columns:
        df["class_name"] = np.where(df["image_label"].astype(int) == 1, "forged", "authentic")
    image_paths, mask_paths, first_mask_paths = [], [], []
    for row in df.itertuples(index=False):
        image_id = str(getattr(row, "image_id"))
        class_name = str(getattr(row, "class_name"))
        image_path = dataset_root / "train_images" / class_name / f"{image_id}.png"
        masks = find_mask_paths(image_id, dataset_root / "train_masks") if class_name == "forged" else []
        image_paths.append(str(image_path))
        mask_paths.append("|".join(str(p) for p in masks))
        first_mask_paths.append(str(masks[0]) if masks else "")
    df["image_path"] = image_paths
    df["mask_paths"] = mask_paths
    df["mask_path"] = first_mask_paths
    df["image_id"] = df["image_id"].astype(str)
    df["image_label"] = df["image_label"].astype(int)
    df["label"] = df["image_label"].astype(int)
    return df


def split_files_exist(split_dir: Path) -> bool:
    return all((split_dir / name).exists() for name in ["full.csv", "train.csv", "val.csv", "test.csv"])


def group_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("image_id", as_index=False)
        .agg(image_label=("image_label", "max"), n_rows=("sample_id", "count"))
        .sort_values("image_id")
        .reset_index(drop=True)
    )


def create_shared_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_df = group_level_frame(df)
    groups = group_df["image_id"].to_numpy()
    y = group_df["image_label"].to_numpy()
    try:
        sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        trainval_idx, test_idx = next(sgkf_test.split(group_df, y, groups=groups))
    except Exception:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        trainval_idx, test_idx = next(splitter.split(group_df, y, groups=groups))
    trainval_groups = group_df.iloc[trainval_idx].reset_index(drop=True)
    test_groups = set(group_df.iloc[test_idx]["image_id"].astype(str))
    rel_val_fraction = 0.10 / 0.80
    n_splits_val = max(2, int(round(1 / rel_val_fraction)))
    try:
        sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed + 1)
        train_idx, val_idx = next(sgkf_val.split(trainval_groups, trainval_groups["image_label"], groups=trainval_groups["image_id"]))
    except Exception:
        splitter = GroupShuffleSplit(n_splits=1, test_size=rel_val_fraction, random_state=seed + 1)
        train_idx, val_idx = next(splitter.split(trainval_groups, trainval_groups["image_label"], groups=trainval_groups["image_id"]))
    train_groups = set(trainval_groups.iloc[train_idx]["image_id"].astype(str))
    val_groups = set(trainval_groups.iloc[val_idx]["image_id"].astype(str))
    train_df = df[df["image_id"].isin(train_groups)].reset_index(drop=True)
    val_df = df[df["image_id"].isin(val_groups)].reset_index(drop=True)
    test_df = df[df["image_id"].isin(test_groups)].reset_index(drop=True)
    return train_df, val_df, test_df


if split_files_exist(SHARED_SPLIT_DIR):
    print("[split] Mevcut shared split yukleniyor.")
    full_df = repair_split_paths(pd.read_csv(SHARED_SPLIT_DIR / "full.csv"), DATASET_ROOT)
    train_df = repair_split_paths(pd.read_csv(SHARED_SPLIT_DIR / "train.csv"), DATASET_ROOT)
    val_df = repair_split_paths(pd.read_csv(SHARED_SPLIT_DIR / "val.csv"), DATASET_ROOT)
    test_df = repair_split_paths(pd.read_csv(SHARED_SPLIT_DIR / "test.csv"), DATASET_ROOT)
else:
    print("[split] Shared split bulunamadi; seed=42 ile yeniden olusturuluyor.")
    full_df = build_dataset_index(DATASET_ROOT)
    train_df, val_df, test_df = create_shared_split(full_df, CFG.seed)
    for name, part in [("full", full_df), ("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(SHARED_SPLIT_DIR / f"{name}.csv", index=False)

leakage = {
    "train_val_image_id_overlap": len(set(train_df["image_id"]) & set(val_df["image_id"])),
    "train_test_image_id_overlap": len(set(train_df["image_id"]) & set(test_df["image_id"])),
    "val_test_image_id_overlap": len(set(val_df["image_id"]) & set(test_df["image_id"])),
}
print("[split sizes]", {k: len(v) for k, v in {"train": train_df, "val": val_df, "test": test_df}.items()})
print("[leakage]", leakage)
assert all(v == 0 for v in leakage.values()), "Split leakage tespit edildi."


def load_rgb_image(path: str) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def binarize_loaded_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        binary = mask > 0
    elif mask.ndim == 3:
        if mask.shape[0] <= 16 and mask.shape[1] > 16 and mask.shape[2] > 16:
            binary = np.any(mask > 0, axis=0)
        else:
            binary = np.any(mask > 0, axis=-1)
    else:
        raise ValueError(f"Desteklenmeyen maske shape: {mask.shape}")
    return binary.astype(np.float32)


def load_binary_mask(mask_paths: str, label: int, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if int(label) == 0:
        return np.zeros((h, w), dtype=np.float32)
    paths = [p for p in str(mask_paths).split("|") if p and p != "nan"]
    if not paths:
        return np.zeros((h, w), dtype=np.float32)
    merged = None
    for path in paths:
        raw = np.load(path, allow_pickle=False)
        mask = binarize_loaded_mask(raw)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        merged = mask if merged is None else np.maximum(merged, mask)
    return (merged > 0).astype(np.float32)


def resize_mask_nearest(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def resize_aug(size: int):
    try:
        return A.Resize(size, size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
    except TypeError:
        return A.Resize(size, size, interpolation=cv2.INTER_LINEAR)


def get_eval_transform(size: int) -> A.Compose:
    return A.Compose([resize_aug(size)])


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


class ForgeryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_size: int, eval_size: int):
        self.df = df.reset_index(drop=True)
        self.input_size = int(input_size)
        self.eval_size = int(eval_size)
        self.transform = get_eval_transform(self.input_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image = load_rgb_image(row["image_path"])
        h, w = image.shape[:2]
        mask = load_binary_mask(row.get("mask_paths", row.get("mask_path", "")), int(row["image_label"]), (h, w))
        tr = self.transform(image=image, mask=mask)
        image_t = normalize_image(tr["image"])
        mask_t = (tr["mask"] > 0).astype(np.float32)
        eval_mask = resize_mask_nearest(mask_t, self.eval_size) if self.input_size != self.eval_size else mask_t
        return {
            "image": torch.from_numpy(image_t.transpose(2, 0, 1)).float(),
            "eval_mask": torch.from_numpy(eval_mask[None]).float(),
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "class_name": str(row["class_name"]),
            "image_label": int(row["image_label"]),
            "orig_h": int(h),
            "orig_w": int(w),
        }


class SegFormerBinaryWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class DINOv2LiteDecoder(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-small", freeze_backbone: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if all(not p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:, :]
        b, n, c = tokens.shape
        grid = int(math.sqrt(n))
        tokens = tokens[:, : grid * grid, :]
        feat = tokens.transpose(1, 2).contiguous().view(b, c, grid, grid)
        logits = self.decoder(feat)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(exp: ExperimentConfig) -> nn.Module:
    if exp.model_type == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=exp.encoder_or_backbone,
            encoder_weights=exp.encoder_weights,
            in_channels=exp.in_channels,
            classes=exp.classes,
            activation=None,
        )
    if exp.model_type == "unet":
        return smp.Unet(
            encoder_name=exp.encoder_or_backbone,
            encoder_weights=exp.encoder_weights,
            in_channels=exp.in_channels,
            classes=exp.classes,
            activation=None,
        )
    if exp.model_type == "segformer":
        return SegFormerBinaryWrapper(exp.hf_model_name or "nvidia/segformer-b0-finetuned-ade-512-512")
    if exp.model_type == "dinov2_lite_decoder":
        return DINOv2LiteDecoder(exp.hf_model_name or "facebook/dinov2-small", freeze_backbone=exp.freeze_backbone_stage1)
    raise ValueError(exp.model_type)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def load_checkpoint(path: Path, model: nn.Module) -> Dict[str, Any]:
    # PyTorch 2.6 icin kritik duzeltme: Bu checkpoint'ler kendi egitimimizden geldigi icin weights_only=False kullaniyoruz.
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def autocast_context(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    if TORCH_AMP_NEW_API:
        return autocast(device_type=device.type, enabled=enabled)
    return autocast(enabled=enabled)


def make_loader(df: pd.DataFrame, exp: ExperimentConfig) -> DataLoader:
    ds = ForgeryDataset(df, input_size=exp.image_size or CFG.image_size, eval_size=exp.eval_size or CFG.image_size)
    return DataLoader(
        ds,
        batch_size=exp.batch_size or CFG.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, desc: str) -> Tuple[List[Dict[str, Any]], float]:
    model.eval()
    records: List[Dict[str, Any]] = []
    n_images = 0
    t0 = time.time()
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(DEVICE, non_blocking=True)
        eval_masks = batch["eval_mask"].to(DEVICE, non_blocking=True)
        with autocast_context(DEVICE, CFG.use_amp):
            logits = model(images)
            if logits.shape[-2:] != eval_masks.shape[-2:]:
                logits = F.interpolate(logits, size=eval_masks.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        masks = eval_masks.detach().cpu().numpy()[:, 0]
        for i in range(probs.shape[0]):
            records.append({
                "image_id": str(batch["image_id"][i]),
                "image_path": str(batch["image_path"][i]),
                "class_name": str(batch["class_name"][i]),
                "image_label": int(batch["image_label"][i]),
                "orig_h": int(batch["orig_h"][i]),
                "orig_w": int(batch["orig_w"][i]),
                "prob": probs[i].astype(np.float32),
                "mask": (masks[i] > 0).astype(np.uint8),
            })
        n_images += probs.shape[0]
    elapsed = time.time() - t0
    return records, 1000.0 * elapsed / max(n_images, 1)


def clean_binary_mask(binary: np.ndarray, prob: np.ndarray, min_area: int = 0, min_mean_prob: float = 0.0) -> np.ndarray:
    binary = binary.astype(np.uint8)
    if min_area <= 0 and min_mean_prob <= 0:
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label_id in range(1, n):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        comp = labels == label_id
        mean_prob = float(prob[comp].mean()) if area > 0 else 0.0
        if area >= min_area and mean_prob >= min_mean_prob:
            cleaned[comp] = 1
    return cleaned


def safe_auc(y_true: np.ndarray, y_score: np.ndarray, kind: str) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        if kind == "auprc":
            return float(average_precision_score(y_true, y_score))
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def pixel_metrics_from_records(records: List[Dict[str, Any]], threshold: float, clean: bool = False,
                               min_area: int = 0, min_mean_prob: float = 0.0,
                               forged_only: bool = False, compute_auc: bool = True) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    pred_ratios, gt_ratios = [], []
    all_probs, all_gts = [], []
    for rec in records:
        if forged_only and int(rec["image_label"]) != 1:
            continue
        prob = rec["prob"]
        gt = rec["mask"].astype(np.uint8)
        pred = (prob >= threshold).astype(np.uint8)
        if clean:
            pred = clean_binary_mask(pred, prob, min_area=min_area, min_mean_prob=min_mean_prob)
        tp += int(((pred == 1) & (gt == 1)).sum())
        fp += int(((pred == 1) & (gt == 0)).sum())
        tn += int(((pred == 0) & (gt == 0)).sum())
        fn += int(((pred == 0) & (gt == 1)).sum())
        pred_ratios.append(float(pred.mean()))
        gt_ratios.append(float(gt.mean()))
        if compute_auc:
            all_probs.append(prob.reshape(-1).astype(np.float32))
            all_gts.append(gt.reshape(-1).astype(np.uint8))
    eps = 1e-7
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_true = np.concatenate(all_gts) if all_gts else np.array([])
    return {
        "dice": float((2 * tp) / max(2 * tp + fp + fn, eps)),
        "iou": float(tp / max(tp + fp + fn, eps)),
        "precision": float(tp / max(tp + fp, eps)),
        "recall": float(tp / max(tp + fn, eps)),
        "specificity": float(tn / max(tn + fp, eps)),
        "auprc": safe_auc(y_true, y_prob, "auprc") if len(y_true) else float("nan"),
        "roc_auc": safe_auc(y_true, y_prob, "roc_auc") if len(y_true) else float("nan"),
        "pred_positive_pixel_ratio": float(np.mean(pred_ratios)) if pred_ratios else float("nan"),
        "gt_positive_pixel_ratio": float(np.mean(gt_ratios)) if gt_ratios else float("nan"),
    }


def image_score(prob: np.ndarray, pred: Optional[np.ndarray] = None, method: str = "topk_mean_probability") -> float:
    if method == "max_probability":
        return float(prob.max())
    if method == "pred_mask_ratio":
        if pred is None:
            pred = prob >= 0.5
        return float(pred.mean())
    if method == "topk_mean_probability":
        flat = prob.reshape(-1)
        k = max(1, int(len(flat) * CFG.topk_fraction))
        return float(np.partition(flat, -k)[-k:].mean())
    raise ValueError(method)


def threshold_search_pixel(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    rows = []
    for threshold in CFG.pixel_thresholds:
        m_all = pixel_metrics_from_records(records, threshold, clean=False, forged_only=False, compute_auc=False)
        m_forged = pixel_metrics_from_records(records, threshold, clean=False, forged_only=True, compute_auc=False)
        rows.append({
            "mode": "raw",
            "pixel_threshold": float(threshold),
            "min_component_area": 0,
            "min_component_mean_probability": 0.0,
            "dice_all": m_all["dice"],
            "dice_forged_only": m_forged["dice"],
            "iou_all": m_all["iou"],
            "iou_forged_only": m_forged["iou"],
        })
        for area in CFG.min_component_areas:
            for mean_prob in CFG.min_component_mean_probs:
                if area == 0 and mean_prob == 0:
                    continue
                mc_all = pixel_metrics_from_records(records, threshold, clean=True, min_area=area, min_mean_prob=mean_prob, forged_only=False, compute_auc=False)
                mc_forged = pixel_metrics_from_records(records, threshold, clean=True, min_area=area, min_mean_prob=mean_prob, forged_only=True, compute_auc=False)
                rows.append({
                    "mode": "clean",
                    "pixel_threshold": float(threshold),
                    "min_component_area": int(area),
                    "min_component_mean_probability": float(mean_prob),
                    "dice_all": mc_all["dice"],
                    "dice_forged_only": mc_forged["dice"],
                    "iou_all": mc_all["iou"],
                    "iou_forged_only": mc_forged["iou"],
                })
    df = pd.DataFrame(rows)
    raw_best = df[df["mode"] == "raw"].sort_values(["dice_forged_only", "iou_forged_only"], ascending=False).iloc[0].to_dict()
    clean_best = df[df["mode"] == "clean"].sort_values(["dice_forged_only", "iou_forged_only"], ascending=False).iloc[0].to_dict()
    return df, raw_best, clean_best


def threshold_search_image(records: List[Dict[str, Any]], selected_pixel_threshold: float,
                           clean: bool, min_area: int, min_mean_prob: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    labels = np.array([int(r["image_label"]) for r in records], dtype=np.uint8)
    for method in CFG.image_score_methods:
        scores = []
        for rec in records:
            prob = rec["prob"]
            pred = (prob >= selected_pixel_threshold).astype(np.uint8)
            if clean:
                pred = clean_binary_mask(pred, prob, min_area=min_area, min_mean_prob=min_mean_prob)
            scores.append(image_score(prob, pred, method))
        scores = np.array(scores, dtype=np.float32)
        for thr in CFG.image_thresholds:
            pred_img = (scores >= thr).astype(np.uint8)
            tn = int(((pred_img == 0) & (labels == 0)).sum())
            fp = int(((pred_img == 1) & (labels == 0)).sum())
            rows.append({
                "score_method": method,
                "image_threshold": float(thr),
                "accuracy": float(accuracy_score(labels, pred_img)),
                "precision": float(precision_score(labels, pred_img, zero_division=0)),
                "recall": float(recall_score(labels, pred_img, zero_division=0)),
                "specificity": float(tn / max(tn + fp, 1)),
                "f1": float(f1_score(labels, pred_img, zero_division=0)),
                "roc_auc": safe_auc(labels, scores, "roc_auc"),
            })
    df = pd.DataFrame(rows)
    best = df.sort_values(["f1", "recall", "specificity"], ascending=False).iloc[0].to_dict()
    return df, best


def image_level_metrics(records: List[Dict[str, Any]], pixel_threshold: float, image_threshold: float,
                        score_method: str, clean: bool = False, min_area: int = 0,
                        min_mean_prob: float = 0.0) -> Dict[str, float]:
    y_true, y_score = [], []
    for rec in records:
        prob = rec["prob"]
        pred = (prob >= pixel_threshold).astype(np.uint8)
        if clean:
            pred = clean_binary_mask(pred, prob, min_area=min_area, min_mean_prob=min_mean_prob)
        y_true.append(int(rec["image_label"]))
        y_score.append(image_score(prob, pred, method=score_method))
    y_true = np.array(y_true, dtype=np.uint8)
    y_score = np.array(y_score, dtype=np.float32)
    y_pred = (y_score >= image_threshold).astype(np.uint8)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return {
        "image_accuracy": float(accuracy_score(y_true, y_pred)),
        "image_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "image_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "image_specificity": float(tn / max(tn + fp, 1)),
        "image_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "image_roc_auc": safe_auc(y_true, y_score, "roc_auc"),
        "image_score_method": score_method,
    }


def dice_iou_from_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    iou = (inter + eps) / (np.logical_or(pred, gt).sum() + eps)
    return float(dice), float(iou)


def per_image_metrics(records: List[Dict[str, Any]], pixel_threshold: float, image_threshold: float,
                      score_method: str, clean: bool = False, min_area: int = 0,
                      min_mean_prob: float = 0.0) -> pd.DataFrame:
    rows = []
    for rec in records:
        prob = rec["prob"]
        gt = rec["mask"]
        pred = (prob >= pixel_threshold).astype(np.uint8)
        if clean:
            pred = clean_binary_mask(pred, prob, min_area=min_area, min_mean_prob=min_mean_prob)
        dice, iou = dice_iou_from_binary(pred, gt)
        score = image_score(prob, pred, method=score_method)
        rows.append({
            "image_id": rec["image_id"],
            "image_path": rec["image_path"],
            "class_name": rec["class_name"],
            "image_label": rec["image_label"],
            "gt_area": int(gt.sum()),
            "pred_area": int(pred.sum()),
            "dice": dice,
            "iou": iou,
            "image_score": score,
            "image_pred": int(score >= image_threshold),
            "gt_positive_pixel_ratio": float(gt.mean()),
            "pred_positive_pixel_ratio": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def evaluate_records(records: List[Dict[str, Any]], selected: Dict[str, Any], mode_name: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    clean = mode_name == "clean"
    pixel_threshold = float(selected["pixel_threshold"])
    min_area = int(selected.get("min_component_area", 0))
    min_mean_prob = float(selected.get("min_component_mean_probability", 0.0))
    image_threshold = float(selected["image_threshold"])
    score_method = str(selected["score_method"])
    all_m = pixel_metrics_from_records(records, pixel_threshold, clean=clean, min_area=min_area, min_mean_prob=min_mean_prob, forged_only=False)
    forged_m = pixel_metrics_from_records(records, pixel_threshold, clean=clean, min_area=min_area, min_mean_prob=min_mean_prob, forged_only=True)
    img_m = image_level_metrics(records, pixel_threshold, image_threshold, score_method, clean=clean, min_area=min_area, min_mean_prob=min_mean_prob)
    per_df = per_image_metrics(records, pixel_threshold, image_threshold, score_method, clean=clean, min_area=min_area, min_mean_prob=min_mean_prob)
    metrics = {
        "mode": mode_name,
        "selected_pixel_threshold": pixel_threshold,
        "selected_image_threshold": image_threshold,
        "score_method": score_method,
        "min_component_area": min_area,
        "min_component_mean_probability": min_mean_prob,
        "test_dice_all": all_m["dice"],
        "test_dice_forged_only": forged_m["dice"],
        "test_iou_all": all_m["iou"],
        "test_iou_forged_only": forged_m["iou"],
        "test_precision": all_m["precision"],
        "test_recall": all_m["recall"],
        "test_specificity": all_m["specificity"],
        "test_auprc_all": all_m["auprc"],
        "test_auprc_forged_only": forged_m["auprc"],
        "test_roc_auc_all": all_m["roc_auc"],
        "test_roc_auc_forged_only": forged_m["roc_auc"],
        "pred_positive_pixel_ratio": all_m["pred_positive_pixel_ratio"],
        "gt_positive_pixel_ratio": all_m["gt_positive_pixel_ratio"],
        **img_m,
    }
    return metrics, per_df


def component_iou_matrix(gt_labels: np.ndarray, gt_count: int, pred_labels: np.ndarray, pred_count: int) -> np.ndarray:
    if gt_count == 0 or pred_count == 0:
        return np.zeros((gt_count, pred_count), dtype=np.float32)
    mat = np.zeros((gt_count, pred_count), dtype=np.float32)
    for gi in range(1, gt_count + 1):
        gt_comp = gt_labels == gi
        for pi in range(1, pred_count + 1):
            pred_comp = pred_labels == pi
            inter = np.logical_and(gt_comp, pred_comp).sum()
            union = np.logical_or(gt_comp, pred_comp).sum()
            mat[gi - 1, pi - 1] = inter / max(union, 1)
    return mat


def match_components(gt: np.ndarray, pred: np.ndarray, iou_threshold: float) -> Dict[str, Any]:
    gt_n, gt_labels, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    pred_n, pred_labels, _, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    gt_count = gt_n - 1
    pred_count = pred_n - 1
    iou_mat = component_iou_matrix(gt_labels, gt_count, pred_labels, pred_count)
    matched = 0
    if gt_count > 0 and pred_count > 0:
        rows, cols = linear_sum_assignment(-iou_mat)
        matched = sum(1 for r, c in zip(rows, cols) if iou_mat[r, c] >= iou_threshold)
    return {
        "gt_component_count": int(gt_count),
        "predicted_component_count": int(pred_count),
        "matched_component_count": int(matched),
        "false_positive_component_count": int(pred_count - matched),
        "false_negative_component_count": int(gt_count - matched),
    }


def component_metrics(records: List[Dict[str, Any]], selected: Dict[str, Any], mode_name: str, iou_threshold: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    pixel_threshold = float(selected["pixel_threshold"])
    clean = mode_name == "clean"
    min_area = int(selected.get("min_component_area", 0))
    min_mean_prob = float(selected.get("min_component_mean_probability", 0.0))
    rows = []
    total_tp = total_fp = total_fn = 0
    auth_with_pred = 0
    for rec in records:
        prob = rec["prob"]
        gt = rec["mask"].astype(np.uint8)
        pred = (prob >= pixel_threshold).astype(np.uint8)
        if clean:
            pred = clean_binary_mask(pred, prob, min_area=min_area, min_mean_prob=min_mean_prob)
        comp = match_components(gt, pred, iou_threshold)
        total_tp += comp["matched_component_count"]
        total_fp += comp["false_positive_component_count"]
        total_fn += comp["false_negative_component_count"]
        auth_has_pred = bool(int(rec["image_label"]) == 0 and comp["predicted_component_count"] > 0)
        auth_with_pred += int(auth_has_pred)
        rows.append({
            "image_id": rec["image_id"],
            "class_name": rec["class_name"],
            "image_label": rec["image_label"],
            "component_iou_threshold": float(iou_threshold),
            "authentic_image_has_prediction": auth_has_pred,
            **comp,
        })
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    auth_n = sum(int(r["image_label"]) == 0 for r in records)
    metrics = {
        "component_iou_threshold": float(iou_threshold),
        "component_precision": float(precision),
        "component_recall": float(recall),
        "component_f1": float(f1),
        "authentic_fp_rate": float(auth_with_pred / max(auth_n, 1)),
        "avg_pred_component_count": float(np.mean([r["predicted_component_count"] for r in rows])) if rows else float("nan"),
    }
    return metrics, pd.DataFrame(rows)


def save_prediction_npz(records: List[Dict[str, Any]], out_path: Path) -> None:
    if not CFG.save_prediction_probs:
        return
    probs = np.stack([r["prob"].astype(np.float16) for r in records])
    masks = np.stack([r["mask"].astype(np.uint8) for r in records])
    image_ids = np.array([r["image_id"] for r in records])
    labels = np.array([r["image_label"] for r in records], dtype=np.uint8)
    np.savez_compressed(out_path, probs=probs, masks=masks, image_ids=image_ids, labels=labels)


def best_epoch_from_metrics(metrics_path: Path) -> Tuple[Optional[int], Optional[float]]:
    if not metrics_path.exists():
        return None, None
    df = pd.read_csv(metrics_path)
    if df.empty or "val_forged_dice" not in df.columns:
        return None, None
    idx = df["val_forged_dice"].astype(float).idxmax()
    return int(df.loc[idx, "epoch"]), float(df.loc[idx, "val_forged_dice"])


def evaluate_experiment(exp: ExperimentConfig) -> Optional[Dict[str, Any]]:
    out_dir = RUN_ROOT / exp.name
    best_ckpt = out_dir / "best_model.pth"
    if not best_ckpt.exists():
        print(f"[skip] {exp.name}: best_model.pth yok.")
        return None
    if (out_dir / "summary.json").exists() and (out_dir / "test_metrics.csv").exists():
        print(f"[skip_existing] {exp.name}: test metrikleri zaten var.")
        with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"\n===== EVALUATE ONLY: {exp.name} =====")
    model = build_model(exp).to(DEVICE)
    ckpt = load_checkpoint(best_ckpt, model)
    trainable_params, total_params = count_parameters(model)
    best_epoch, best_val_forged_dice = best_epoch_from_metrics(out_dir / "metrics.csv")
    if best_epoch is None:
        best_epoch = int(ckpt.get("epoch", -1))
        best_val_forged_dice = float(ckpt.get("best_score", float("nan")))

    val_loader = make_loader(val_df, exp)
    test_loader = make_loader(test_df, exp)

    val_records, _ = collect_predictions(model, val_loader, desc=f"{exp.name} val predict")
    save_prediction_npz(val_records, out_dir / "val_predictions_probs.npz")
    pixel_search_df, raw_best, clean_best = threshold_search_pixel(val_records)
    pixel_search_df.to_csv(out_dir / "threshold_search_pixel.csv", index=False)

    raw_img_df, raw_img_best = threshold_search_image(val_records, float(raw_best["pixel_threshold"]), clean=False, min_area=0, min_mean_prob=0.0)
    clean_img_df, clean_img_best = threshold_search_image(
        val_records,
        float(clean_best["pixel_threshold"]),
        clean=True,
        min_area=int(clean_best["min_component_area"]),
        min_mean_prob=float(clean_best["min_component_mean_probability"]),
    )
    pd.concat([raw_img_df.assign(mode="raw"), clean_img_df.assign(mode="clean")], ignore_index=True).to_csv(
        out_dir / "threshold_search_image.csv", index=False
    )
    raw_selected = {**raw_best, **raw_img_best}
    clean_selected = {**clean_best, **clean_img_best}

    val_raw_metrics, _ = evaluate_records(val_records, raw_selected, mode_name="raw")
    val_clean_metrics, _ = evaluate_records(val_records, clean_selected, mode_name="clean")
    pd.DataFrame([val_raw_metrics, val_clean_metrics]).to_csv(out_dir / "val_metrics.csv", index=False)

    test_records, infer_ms = collect_predictions(model, test_loader, desc=f"{exp.name} test predict")
    save_prediction_npz(test_records, out_dir / "test_predictions_probs.npz")
    raw_metrics, _ = evaluate_records(test_records, raw_selected, mode_name="raw")
    clean_metrics, clean_per_df = evaluate_records(test_records, clean_selected, mode_name="clean")

    comp_summary_rows = []
    primary_comp_metrics = {}
    primary_comp_df = pd.DataFrame()
    for iou_thr in CFG.component_iou_thresholds:
        comp_metrics_i, comp_df_i = component_metrics(test_records, clean_selected, mode_name="clean", iou_threshold=iou_thr)
        comp_summary_rows.append(comp_metrics_i)
        if float(iou_thr) == float(CFG.primary_component_iou_threshold):
            primary_comp_metrics = comp_metrics_i
            primary_comp_df = comp_df_i

    pd.DataFrame([raw_metrics]).to_csv(out_dir / "test_metrics_raw.csv", index=False)
    pd.DataFrame([clean_metrics]).to_csv(out_dir / "test_metrics_clean.csv", index=False)
    pd.DataFrame([clean_metrics]).to_csv(out_dir / "test_metrics.csv", index=False)
    clean_per_df.to_csv(out_dir / "test_per_image_metrics.csv", index=False)
    primary_comp_df.to_csv(out_dir / "test_component_metrics.csv", index=False)
    pd.DataFrame(comp_summary_rows).to_csv(out_dir / "test_component_summary_by_iou.csv", index=False)

    summary = {
        "experiment_name": exp.name,
        "model_family": exp.model_family,
        "encoder_or_backbone": exp.encoder_or_backbone,
        "input_mode": exp.input_mode,
        "image_size": exp.image_size or CFG.image_size,
        "eval_size": exp.eval_size or CFG.image_size,
        "best_epoch": best_epoch,
        "best_val_forged_dice": best_val_forged_dice,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "selected_pixel_threshold": clean_metrics["selected_pixel_threshold"],
        "selected_image_threshold": clean_metrics["selected_image_threshold"],
        "score_method": clean_metrics["score_method"],
        "min_component_area": clean_metrics["min_component_area"],
        "min_component_mean_probability": clean_metrics["min_component_mean_probability"],
        "training_time_minutes": float(pd.read_csv(out_dir / "metrics.csv")["epoch_time_minutes"].sum()) if (out_dir / "metrics.csv").exists() else np.nan,
        "inference_time_per_image_ms": infer_ms,
        **clean_metrics,
        **primary_comp_metrics,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# {exp.name}\n\n")
        f.write("Evaluate-only devam kosusu ile mevcut best_model.pth uzerinden uretilmistir.\n\n")
        f.write(pd.DataFrame([summary]).to_markdown(index=False))
        f.write("\n")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[done] {exp.name}: test_dice_forged_only={summary['test_dice_forged_only']:.4f}, component_f1={summary['component_f1']:.4f}")
    return summary


COMPARISON_COLUMNS = [
    "experiment_name", "model_family", "encoder_or_backbone", "input_mode", "image_size",
    "trainable_params", "total_params", "best_epoch", "selected_pixel_threshold",
    "selected_image_threshold", "min_component_area", "test_dice_all", "test_dice_forged_only",
    "test_iou_all", "test_iou_forged_only", "test_precision", "test_recall", "test_specificity",
    "test_auprc_all", "test_auprc_forged_only", "image_accuracy", "image_precision",
    "image_recall", "image_specificity", "image_f1", "image_roc_auc", "component_precision",
    "component_recall", "component_f1", "authentic_fp_rate", "avg_pred_component_count",
    "training_time_minutes", "inference_time_per_image_ms",
]


def write_comparison_table() -> pd.DataFrame:
    rows = []
    for exp in EXPERIMENTS:
        path = RUN_ROOT / exp.name / "summary.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            rows.append({col: summary.get(col, np.nan) for col in COMPARISON_COLUMNS})
    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp
    comp = comp.sort_values(
        ["test_dice_forged_only", "test_iou_forged_only", "component_f1", "authentic_fp_rate"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    comp.to_csv(COMPARISON_ROOT / "model_comparison_full.csv", index=False)
    with open(COMPARISON_ROOT / "model_comparison_full.md", "w", encoding="utf-8") as f:
        f.write("# Full-data Model Comparison\n\n")
        f.write(comp.to_markdown(index=False))
        f.write("\n")
    with open(COMPARISON_ROOT / "report.md", "w", encoding="utf-8") as f:
        f.write("# Full-data Dort Model Karsilastirma Raporu\n\n")
        f.write("Bu rapor egitim tekrarlanmadan, mevcut checkpoint'lerin evaluate-only kosusuyla uretilmistir.\n\n")
        f.write(comp.to_markdown(index=False))
        f.write("\n")
    return comp


def run_statistical_tests() -> pd.DataFrame:
    pairs = [
        ("unetpp_resnet34_rgb_full", "efficientnetb0_unet_rgb_full"),
        ("unetpp_resnet34_rgb_full", "segformer_b0_rgb_full"),
        ("unetpp_resnet34_rgb_full", "dinov2_lite_decoder_rgb_full"),
        ("segformer_b0_rgb_full", "dinov2_lite_decoder_rgb_full"),
    ]
    per_image = {}
    for exp in EXPERIMENTS:
        path = RUN_ROOT / exp.name / "test_per_image_metrics.csv"
        if path.exists():
            df = pd.read_csv(path)[["image_id", "dice", "iou"]].rename(
                columns={"dice": f"{exp.name}_dice", "iou": f"{exp.name}_iou"}
            )
            df["image_id"] = df["image_id"].astype(str)
            per_image[exp.name] = df
    rows = []
    for a, b in pairs:
        if a not in per_image or b not in per_image:
            continue
        merged = per_image[a].merge(per_image[b], on="image_id", how="inner")
        for metric in ["dice", "iou"]:
            x = merged[f"{a}_{metric}"].to_numpy(dtype=float)
            y = merged[f"{b}_{metric}"].to_numpy(dtype=float)
            diff = x - y
            try:
                t_stat, t_p = ttest_rel(x, y, nan_policy="omit")
            except Exception:
                t_stat, t_p = np.nan, np.nan
            try:
                w_stat, w_p = wilcoxon(diff)
            except Exception:
                w_stat, w_p = np.nan, np.nan
            rows.append({
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "n_images": int(len(merged)),
                "mean_a": float(np.nanmean(x)),
                "mean_b": float(np.nanmean(y)),
                "mean_diff_a_minus_b": float(np.nanmean(diff)),
                "paired_t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
                "paired_t_pvalue": float(t_p) if not np.isnan(t_p) else np.nan,
                "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
                "wilcoxon_pvalue": float(w_p) if not np.isnan(w_p) else np.nan,
            })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(COMPARISON_ROOT / "statistical_tests.csv", index=False)
    return stats_df


def main() -> None:
    print("Evaluate-only devam basliyor. Egitim yapilmayacak.")
    print("Python:", sys.version.split()[0], "Torch:", torch.__version__, "Platform:", platform.platform())
    summaries = []
    for exp in EXPERIMENTS:
        try:
            summary = evaluate_experiment(exp)
            if summary is not None:
                summaries.append(summary)
        except Exception as exc:
            print(f"[ERROR] {exp.name}: {exc}")
            import traceback
            with open(RUN_ROOT / "evaluate_only_errors.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n===== {exp.name} =====\n")
                f.write(traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    comp = write_comparison_table()
    stats = run_statistical_tests()
    print("\nComparison:")
    print(comp)
    print("\nStatistical tests:")
    print(stats)
    print("\nCikti klasoru:", RUN_ROOT)
    print("Comparison klasoru:", COMPARISON_ROOT)


if __name__ == "__main__":
    main()
