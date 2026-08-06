# %% [markdown]
# # Recod.ai/LUC - Deney 6 Small-mask Improvement Experiment
#
# Bu notebook, Deney 5'te acik kalan kucuk sahtecilik bolgesi problemini hedefler.
# Yeni model ailesi aramaz; Deney 5 sonunda anlamli kalan EfficientNetB0-UNet ve
# SegFormer-B0 modellerini 384x384 cozumurlukte yeniden egitir, validation setinde
# threshold/post-processing secimini yapar ve test setinde sabit config ile raporlar.
#
# Temel kural: Test seti threshold, image-level threshold veya post-processing parametresi
# seciminde kullanilmaz.

# %% [markdown]
# ## 1. Install / Imports

# %%
import os
import sys
import json
import math
import time
import random
import platform
import warnings
import traceback
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*The secret `HF_TOKEN` does not exist.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")


def ensure_package(import_name: str, pip_name: Optional[str] = None) -> None:
    """Colab/Kaggle ortaminda eksik paketleri sessizce kurar."""
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
ensure_package("scipy")
ensure_package("tabulate")
ensure_package("albumentations")
ensure_package("segmentation_models_pytorch", "segmentation-models-pytorch")
ensure_package("transformers")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

import albumentations as A
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation

try:
    from torch.amp import GradScaler, autocast
    TORCH_AMP_NEW_API = True
except Exception:
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AMP_NEW_API = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EPS = 1e-7


# %% [markdown]
# ## 2. Global Config

# %%
@dataclass
class ExperimentConfig:
    name: str
    model_type: str
    model_family: str
    encoder_or_backbone: str
    input_mode: str = "rgb"
    image_size: int = 384
    classes: int = 1
    encoder_weights: Optional[str] = "imagenet"
    hf_model_name: Optional[str] = None
    pretrained: bool = True
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    use_pos_weight: bool = True
    use_small_mask_oversampling: bool = False
    use_small_mask_loss_weight: bool = False
    enabled: bool = True


@dataclass
class GlobalConfig:
    seed: int = 42
    image_size: int = 384
    num_workers: int = 2
    effective_batch_size: int = 8
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    use_amp: bool = True
    run_robustness: bool = False
    save_prediction_probs: bool = True
    max_visual_examples: int = 12
    max_curve_pixels: int = 2_000_000
    pixel_thresholds_full: Tuple[float, ...] = tuple(np.round(np.arange(0.10, 0.901, 0.05), 2))
    pixel_thresholds_grid: Tuple[float, ...] = (0.45, 0.55, 0.65, 0.75, 0.85)
    image_thresholds: Tuple[float, ...] = tuple(np.round(np.arange(0.01, 0.991, 0.01), 2))
    min_component_areas: Tuple[int, ...] = (0, 25, 100, 500)
    min_component_mean_probs: Tuple[float, ...] = (0.0, 0.2, 0.4)
    morphologies: Tuple[str, ...] = ("none", "open", "close", "open_close")
    morph_kernel_sizes: Tuple[int, ...] = (3, 5)
    top_k_components_values: Tuple[Optional[int], ...] = (None, 1, 2, 3)
    component_iou_thresholds: Tuple[float, ...] = (0.10, 0.25, 0.50)
    image_score_types: Tuple[str, ...] = (
        "max_probability",
        "top1_mean_probability",
        "top5_mean_probability",
        "pred_mask_ratio_raw",
        "pred_mask_ratio_clean",
        "max_component_mean_probability",
        "max_component_area_ratio",
    )


CFG = GlobalConfig()

EXPERIMENTS = [
    ExperimentConfig(
        name="efficientnetb0_unet_rgb_384_smallmask",
        model_type="unet",
        model_family="parameter-efficient CNN transfer baseline",
        encoder_or_backbone="efficientnet-b0",
        encoder_weights="imagenet",
    ),
    ExperimentConfig(
        name="segformer_b0_rgb_384_smallmask",
        model_type="segformer",
        model_family="transformer semantic segmentation",
        encoder_or_backbone="nvidia/segformer-b0-finetuned-ade-512-512",
        hf_model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        encoder_weights=None,
    ),
    ExperimentConfig(
        name="efficientnetb0_unet_rgb_384_smallmask_oversampling",
        model_type="unet",
        model_family="parameter-efficient CNN transfer baseline + Q1/Q2 oversampling",
        encoder_or_backbone="efficientnet-b0",
        encoder_weights="imagenet",
        use_small_mask_oversampling=True,
        enabled=False,  # 6C icin gerekirse True yapin.
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


# %% [markdown]
# ## 3. Path Discovery

# %%
def first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_paths() -> Tuple[Path, Path, Path, Optional[Path]]:
    dataset_candidates = [
        Path("/kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection"),
        Path("/content/drive/MyDrive/bitirmeProjesi/dataset"),
    ]
    split_candidates = [
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/_shared_splits_seed42"),
        Path("/kaggle/working/experiments_full/_shared_splits_seed42"),
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments/_shared_splits_seed42"),
        Path("/kaggle/working/experiments/_shared_splits_seed42"),
        Path("deney_4/_shared_splits_seed42"),
        Path("experiments_full/_shared_splits_seed42"),
        Path("experiments/_shared_splits_seed42"),
    ]
    output_candidates = [
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/experiment6_smallmask_384"),
        Path("/kaggle/working/experiments_full/experiment6_smallmask_384"),
    ]
    exp5_candidates = [
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/experiment5_calibration_postprocessing"),
        Path("/kaggle/working/experiments_full/experiment5_calibration_postprocessing"),
        Path("/kaggle/working/deney_4/experiments_4_full/experiment5_calibration_postprocessing"),
        Path("deney_5/experiments_4_full/experiment5_calibration_postprocessing"),
    ]

    dataset_root = first_existing(dataset_candidates)
    if dataset_root is None:
        raise FileNotFoundError("Dataset path bulunamadi. Kaggle veya Google Drive dataset yolunu kontrol edin.")

    split_dir = first_existing(split_candidates)
    if split_dir is None:
        raise FileNotFoundError("Shared split klasoru bulunamadi. Deney 6 yeni split olusturmaz.")
    for filename in ["full.csv", "train.csv", "val.csv", "test.csv"]:
        if not (split_dir / filename).exists():
            raise FileNotFoundError(f"Split dosyasi eksik: {split_dir / filename}")

    output_root = output_candidates[0] if output_candidates[0].parent.exists() else output_candidates[1]
    output_root.mkdir(parents=True, exist_ok=True)
    exp5_root = first_existing(exp5_candidates)
    return dataset_root, split_dir, output_root, exp5_root


DATASET_ROOT, SPLIT_DIR, OUTPUT_ROOT, EXP5_ROOT = discover_paths()
print("DATASET_ROOT:", DATASET_ROOT)
print("SPLIT_DIR:", SPLIT_DIR)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("EXP5_ROOT:", EXP5_ROOT if EXP5_ROOT else "Bulunamadi; Deney 5 karsilastirmasi uyarili calisacak.")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


save_json(
    {
        "global_config": asdict(CFG),
        "experiments": [asdict(exp) for exp in EXPERIMENTS],
        "dataset_root": str(DATASET_ROOT),
        "split_dir": str(SPLIT_DIR),
        "output_root": str(OUTPUT_ROOT),
        "experiment5_root": str(EXP5_ROOT) if EXP5_ROOT else None,
    },
    OUTPUT_ROOT / "experiment6_config.json",
)


# %% [markdown]
# ## 4. Split Loading and Leakage Check

# %%
def canonical_image_path(row: pd.Series, dataset_root: Path) -> str:
    class_name = str(row.get("class_name", "forged" if int(row.get("image_label", row.get("label", 0))) == 1 else "authentic"))
    image_id = str(row["image_id"])
    return str(dataset_root / "train_images" / class_name / f"{image_id}.png")


def canonical_mask_path(row: pd.Series, dataset_root: Path) -> str:
    if int(row.get("image_label", row.get("label", 0))) == 0:
        return ""
    image_id = str(row["image_id"])
    return str(dataset_root / "train_masks" / f"{image_id}.npy")


def normalize_split_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()
    if "image_label" not in df.columns:
        df["image_label"] = df["label"].astype(int)
    if "label" not in df.columns:
        df["label"] = df["image_label"].astype(int)
    if "class_name" not in df.columns:
        df["class_name"] = np.where(df["image_label"].astype(int) == 1, "forged", "authentic")
    df["image_id"] = df["image_id"].astype(str)
    df["image_path"] = df.apply(lambda r: canonical_image_path(r, DATASET_ROOT), axis=1)
    df["mask_path"] = df.apply(lambda r: canonical_mask_path(r, DATASET_ROOT), axis=1)
    df["split"] = split_name
    return df.reset_index(drop=True)


full_df = normalize_split_df(pd.read_csv(SPLIT_DIR / "full.csv"), "full")
train_df = normalize_split_df(pd.read_csv(SPLIT_DIR / "train.csv"), "train")
val_df = normalize_split_df(pd.read_csv(SPLIT_DIR / "val.csv"), "val")
test_df = normalize_split_df(pd.read_csv(SPLIT_DIR / "test.csv"), "test")


def split_counts(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    labels = df["image_label"].astype(int)
    return {
        "split": split_name,
        "count": int(len(df)),
        "authentic": int((labels == 0).sum()),
        "forged": int((labels == 1).sum()),
    }


def overlap_count(a: pd.DataFrame, b: pd.DataFrame) -> int:
    return len(set(a["image_id"].astype(str)).intersection(set(b["image_id"].astype(str))))


split_summary_rows = [
    split_counts(train_df, "train"),
    split_counts(val_df, "val"),
    split_counts(test_df, "test"),
    {"split": "leak_train_val", "count": overlap_count(train_df, val_df), "authentic": 0, "forged": 0},
    {"split": "leak_train_test", "count": overlap_count(train_df, test_df), "authentic": 0, "forged": 0},
    {"split": "leak_val_test", "count": overlap_count(val_df, test_df), "authentic": 0, "forged": 0},
]
split_summary = pd.DataFrame(split_summary_rows)
split_summary.to_csv(OUTPUT_ROOT / "split_summary.csv", index=False)
print(split_summary)

if any(split_summary[split_summary["split"].str.startswith("leak_")]["count"].astype(int) != 0):
    raise RuntimeError("Leakage kontrolu basarisiz: splitler arasinda image_id overlap var.")


# %% [markdown]
# ## 5. Mask Area Analysis

# %%
def load_image_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask_array(mask_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    if not mask_path:
        return np.zeros(image_shape, dtype=np.float32)
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Forged maske bulunamadi: {path}")
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


def add_mask_area_info(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name} mask area"):
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Goruntu okunamadi: {row['image_path']}")
        h, w = image.shape[:2]
        if int(row["image_label"]) == 1:
            mask = load_mask_array(row["mask_path"], (h, w))
            gt_area = int(mask.sum())
        else:
            gt_area = 0
        out = row.to_dict()
        out.update(
            {
                "height": int(h),
                "width": int(w),
                "image_area": int(h * w),
                "gt_area": gt_area,
                "gt_area_ratio": float(gt_area / max(h * w, 1)),
                "is_forged": int(row["image_label"]) == 1,
            }
        )
        rows.append(out)
    out_df = pd.DataFrame(rows)
    forged = out_df[out_df["image_label"].astype(int) == 1].copy()
    if len(forged) >= 4:
        q1, q2, q3 = np.quantile(forged["gt_area"].astype(float), [0.25, 0.50, 0.75])
        bins = [-1, q1, q2, q3, np.inf]
        labels = ["Q1", "Q2", "Q3", "Q4"]
        out_df["mask_quartile"] = ""
        out_df.loc[forged.index, "mask_quartile"] = pd.cut(
            forged["gt_area"].astype(float), bins=bins, labels=labels, include_lowest=True
        ).astype(str)
    else:
        out_df["mask_quartile"] = ""
    out_df.to_csv(OUTPUT_ROOT / f"mask_area_summary_{split_name}.csv", index=False)
    return out_df


train_df = add_mask_area_info(train_df, "train")
val_df = add_mask_area_info(val_df, "val")
test_df = add_mask_area_info(test_df, "test")


def quartile_summary(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    forged = df[df["image_label"].astype(int) == 1]
    thresholds = {}
    if len(forged) >= 4:
        q = np.quantile(forged["gt_area"].astype(float), [0.25, 0.50, 0.75])
        thresholds = {"q1_max_area": float(q[0]), "q2_max_area": float(q[1]), "q3_max_area": float(q[2])}
    counts = forged["mask_quartile"].value_counts().to_dict() if "mask_quartile" in forged.columns else {}
    return {"split": split_name, "thresholds": thresholds, "counts": {str(k): int(v) for k, v in counts.items()}}


mask_quartiles = {
    "train": quartile_summary(train_df, "train"),
    "val": quartile_summary(val_df, "val"),
    "test": quartile_summary(test_df, "test"),
}
save_json(mask_quartiles, OUTPUT_ROOT / "mask_area_quartiles.json")
print(json.dumps(mask_quartiles, indent=2, ensure_ascii=False))


# %% [markdown]
# ## 6. Dataset and Augmentations

# %%
def get_train_transforms(image_size: int) -> A.Compose:
    try:
        noise_aug = A.GaussNoise(std_range=(0.02, 0.10), p=0.15)
    except TypeError:
        noise_aug = A.GaussNoise(var_limit=(5.0, 25.0), p=0.15)
    try:
        compression_aug = A.ImageCompression(quality_range=(75, 100), p=0.20)
    except TypeError:
        compression_aug = A.ImageCompression(quality_lower=75, quality_upper=100, p=0.20)
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.10,
                rotate_limit=12,
                border_mode=cv2.BORDER_REFLECT_101,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.45,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.35),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            noise_aug,
            compression_aug,
            A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )


def get_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )


class ForgeryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, split: str, augment: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.split = split
        self.transforms = get_train_transforms(image_size) if augment else get_eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image = load_image_rgb(row["image_path"])
        mask = load_mask_array(row["mask_path"], image.shape[:2]) if int(row["image_label"]) == 1 else np.zeros(image.shape[:2], np.float32)
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"].astype(np.float32)
        mask = (augmented["mask"] > 0.5).astype(np.float32)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask[None, :, :]).float()

        loss_weight = 1.0
        if self.split == "train":
            quartile = str(row.get("mask_quartile", ""))
            if quartile == "Q1":
                loss_weight = 2.0
            elif quartile == "Q2":
                loss_weight = 1.5

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": str(row["image_id"]),
            "image_label": torch.tensor(int(row["image_label"]), dtype=torch.long),
            "loss_weight": torch.tensor(loss_weight, dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def make_sampler(df: pd.DataFrame, use_small_mask_oversampling: bool) -> Optional[WeightedRandomSampler]:
    if not use_small_mask_oversampling:
        return None
    weights = []
    for _, row in df.iterrows():
        weight = 1.0
        if int(row["image_label"]) == 1:
            quartile = str(row.get("mask_quartile", ""))
            if quartile == "Q1":
                weight = 3.0
            elif quartile == "Q2":
                weight = 2.0
        weights.append(weight)
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def auto_batch_size(exp: ExperimentConfig) -> int:
    if DEVICE.type != "cuda":
        return 2 if exp.model_type == "segformer" else 4
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if exp.model_type == "segformer":
        return 4 if total_gb >= 14 else 2
    return 8 if total_gb >= 14 else 4


def make_loaders(exp: ExperimentConfig) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    batch_size = auto_batch_size(exp)
    sampler = make_sampler(train_df, exp.use_small_mask_oversampling)
    train_ds = ForgeryDataset(train_df, exp.image_size, "train", augment=True)
    val_ds = ForgeryDataset(val_df, exp.image_size, "val", augment=False)
    test_ds = ForgeryDataset(test_df, exp.image_size, "test", augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=CFG.num_workers,
        pin_memory=DEVICE.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=DEVICE.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=DEVICE.type == "cuda")
    grad_accum = max(1, math.ceil(CFG.effective_batch_size / batch_size))
    return train_loader, val_loader, test_loader, grad_accum


# %% [markdown]
# ## 7. Model Builders

# %%
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


def build_model(exp: ExperimentConfig) -> nn.Module:
    if exp.model_type == "unet":
        return smp.Unet(
            encoder_name=exp.encoder_or_backbone,
            encoder_weights=exp.encoder_weights if exp.pretrained else None,
            in_channels=3,
            classes=exp.classes,
            activation=None,
        )
    if exp.model_type == "segformer":
        return SegFormerBinaryWrapper(exp.hf_model_name or exp.encoder_or_backbone)
    raise ValueError(f"Bilinmeyen model_type: {exp.model_type}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def write_model_summary(model: nn.Module, exp: ExperimentConfig, out_dir: Path) -> None:
    counts = count_parameters(model)
    text = [
        f"model_name: {exp.name}",
        f"model_type: {exp.model_type}",
        f"encoder_or_backbone: {exp.encoder_or_backbone}",
        f"image_size: {exp.image_size}",
        f"pretrained: {exp.pretrained}",
        f"total_parameters: {counts['total_parameters']}",
        f"trainable_parameters: {counts['trainable_parameters']}",
        "",
        str(model),
    ]
    (out_dir / "model_summary.txt").write_text("\n".join(text), encoding="utf-8")


# %% [markdown]
# ## 8. Losses and Metrics

# %%
class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none").flatten(1).mean(dim=1)
        dice = self.dice(logits, targets)
        loss = 0.5 * bce + 0.5 * dice
        if sample_weights is not None:
            loss = loss * sample_weights.to(loss.device)
        return loss.mean()


def compute_pos_weight(df: pd.DataFrame) -> Tuple[float, float]:
    total_pixels = float((df["height"] * df["width"]).sum())
    pos_pixels = float(df["gt_area"].sum())
    pos_ratio = pos_pixels / max(total_pixels, 1.0)
    raw_weight = (total_pixels - pos_pixels) / max(pos_pixels, 1.0)
    return pos_ratio, float(np.clip(raw_weight, 1.0, 20.0))


def binary_dice_np(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float((2 * inter + EPS) / (denom + EPS))


def binary_iou_np(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(gt, pred).sum()
    return float((inter + EPS) / (union + EPS))


def suffix_for_iou(thr: float) -> str:
    return f"{int(round(thr * 100)):03d}"


def safe_auc(y_true: Sequence[int], y_score: Sequence[float], kind: str) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score) if kind == "roc" else average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": i, "bin_min": lo, "bin_max": hi, "n": 0, "confidence": np.nan, "accuracy": np.nan})
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (n / max(len(y_true), 1)) * abs(acc - conf)
        rows.append({"bin": i, "bin_min": lo, "bin_max": hi, "n": n, "confidence": conf, "accuracy": acc})
    return float(ece), pd.DataFrame(rows)


# %% [markdown]
# ## 9. Training Loop

# %%
def amp_context():
    if TORCH_AMP_NEW_API:
        return autocast(device_type="cuda", enabled=CFG.use_amp and DEVICE.type == "cuda")
    return autocast(enabled=CFG.use_amp and DEVICE.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    grad_accum: int,
    use_small_mask_loss_weight: bool,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        sample_weights = batch["loss_weight"].to(DEVICE) if use_small_mask_loss_weight else None
        with amp_context():
            logits = model(images)
            loss = criterion(logits, masks, sample_weights) / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu().item() * grad_accum))
    return {"train_loss": float(np.mean(losses)) if losses else float("nan")}


@torch.no_grad()
def validate_epoch(model: nn.Module, loader: DataLoader, criterion: BCEDiceLoss, df: pd.DataFrame) -> Dict[str, float]:
    model.eval()
    losses, per_image = [], []
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        with amp_context():
            logits = model(images)
            loss = criterion(logits, masks)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        gt = masks.detach().cpu().numpy()[:, 0]
        preds = probs >= 0.5
        for i, idx in enumerate(batch["idx"].numpy()):
            row = df.iloc[int(idx)]
            if int(row["image_label"]) == 1:
                per_image.append(
                    {
                        "dice": binary_dice_np(gt[i], preds[i]),
                        "quartile": str(row.get("mask_quartile", "")),
                    }
                )
        losses.append(float(loss.detach().cpu().item()))
    forged_dice = float(np.mean([r["dice"] for r in per_image])) if per_image else 0.0
    q1_dice = float(np.mean([r["dice"] for r in per_image if r["quartile"] == "Q1"])) if any(r["quartile"] == "Q1" for r in per_image) else 0.0
    q2_dice = float(np.mean([r["dice"] for r in per_image if r["quartile"] == "Q2"])) if any(r["quartile"] == "Q2" for r in per_image) else 0.0
    small_score = 0.50 * forged_dice + 0.30 * q1_dice + 0.20 * q2_dice
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_forged_dice_t050": forged_dice,
        "val_q1_dice_t050": q1_dice,
        "val_q2_dice_t050": q2_dice,
        "val_smallmask_score_t050": small_score,
    }


def plot_training_curves(metrics_df: pd.DataFrame, path: Path) -> None:
    if metrics_df.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="train")
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.25)
    plt.subplot(1, 2, 2)
    plt.plot(metrics_df["epoch"], metrics_df["val_forged_dice_t050"], label="forged dice")
    plt.plot(metrics_df["epoch"], metrics_df["val_q1_dice_t050"], label="Q1 dice")
    plt.plot(metrics_df["epoch"], metrics_df["val_smallmask_score_t050"], label="monitor")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def train_model(exp: ExperimentConfig, out_dir: Path) -> Tuple[Optional[nn.Module], pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(asdict(exp), out_dir / "config.json")
    save_json(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        },
        out_dir / "environment_info.json",
    )

    model = build_model(exp).to(DEVICE)
    write_model_summary(model, exp, out_dir)
    train_loader, val_loader, _, grad_accum = make_loaders(exp)
    pos_ratio, pos_weight_value = compute_pos_weight(train_df)
    pos_weight = torch.tensor([pos_weight_value], device=DEVICE) if exp.use_pos_weight else None
    criterion = BCEDiceLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.learning_rate, weight_decay=exp.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CFG.scheduler_factor, patience=CFG.scheduler_patience
    )
    scaler = GradScaler(enabled=CFG.use_amp and DEVICE.type == "cuda")

    print(f"\n[{exp.name}] batch={train_loader.batch_size}, grad_accum={grad_accum}, pos_ratio={pos_ratio:.6f}, pos_weight={pos_weight_value:.3f}")
    best_score = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    start = time.time()
    for epoch in range(1, exp.epochs + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scaler, grad_accum, exp.use_small_mask_loss_weight)
        val_m = validate_epoch(model, val_loader, criterion, val_df)
        score = val_m["val_smallmask_score_t050"]
        scheduler.step(score)
        row = {
            "epoch": epoch,
            **train_m,
            **val_m,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_accumulation_steps": grad_accum,
            "batch_size": train_loader.batch_size,
            "pos_weight": pos_weight_value if exp.use_pos_weight else 1.0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
        print(
            f"[{exp.name}] epoch {epoch:02d}: loss={row['train_loss']:.4f}/{row['val_loss']:.4f}, "
            f"forged={row['val_forged_dice_t050']:.4f}, q1={row['val_q1_dice_t050']:.4f}, monitor={score:.4f}"
        )
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(exp), "epoch": epoch}, out_dir / "last_model.pth")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(exp), "epoch": epoch}, out_dir / "best_model.pth")
        else:
            no_improve += 1
        if no_improve >= CFG.early_stopping_patience:
            print(f"[{exp.name}] Early stopping: best_epoch={best_epoch}, best_monitor={best_score:.4f}")
            break

    history_df = pd.DataFrame(history)
    history_df["training_time_sec_total"] = time.time() - start
    history_df.to_csv(out_dir / "metrics.csv", index=False)
    plot_training_curves(history_df, out_dir / "training_curves.png")
    ckpt = torch.load(out_dir / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history_df


# %% [markdown]
# ## 10. Prediction Generation

# %%
@torch.no_grad()
def generate_predictions(model: nn.Module, loader: DataLoader, df: pd.DataFrame, out_dir: Path, split_name: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    model.eval()
    records, probs_all = [], []
    start = time.time()
    for batch in tqdm(loader, desc=f"predict {split_name}"):
        images = batch["image"].to(DEVICE, non_blocking=True)
        with amp_context():
            logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0].astype(np.float32)
        probs_all.append(probs.astype(np.float16))
        for i, idx in enumerate(batch["idx"].numpy()):
            row = df.iloc[int(idx)].to_dict()
            mask = load_mask_array(row["mask_path"], (int(row["height"]), int(row["width"]))) if int(row["image_label"]) == 1 else np.zeros((int(row["height"]), int(row["width"])), np.float32)
            mask = cv2.resize(mask.astype(np.uint8), (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            row.update({"mask": mask, "prob_index": len(records)})
            records.append(row)
    probs_arr = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, CFG.image_size, CFG.image_size), dtype=np.float16)
    if CFG.save_prediction_probs:
        np.savez_compressed(out_dir / f"{split_name}_prob_maps.npz", probs=probs_arr)
        meta = pd.DataFrame([{k: v for k, v in r.items() if k != "mask"} for r in records])
        meta.to_csv(out_dir / f"{split_name}_metadata.csv", index=False)
    elapsed = time.time() - start
    print(f"[{split_name}] prediction time: {elapsed:.2f}s, per image: {elapsed / max(len(records), 1):.4f}s")
    return records, probs_arr.astype(np.float32)


def attach_probs(records: List[Dict[str, Any]], probs: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    for rec in records:
        row = dict(rec)
        row["prob"] = probs[int(rec["prob_index"])].astype(np.float32)
        out.append(row)
    return out


# %% [markdown]
# ## 11. Validation Grid Search

# %%
def component_table(binary: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    comps = []
    total_area = float(binary.size)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        ys, xs = np.where(labels == label)
        values = prob[labels == label]
        comps.append(
            {
                "label": int(label),
                "area": area,
                "area_ratio": float(area / max(total_area, 1.0)),
                "mean_probability": float(values.mean()) if values.size else 0.0,
                "max_probability": float(values.max()) if values.size else 0.0,
                "bbox_x": int(xs.min()) if xs.size else 0,
                "bbox_y": int(ys.min()) if ys.size else 0,
                "bbox_w": int(stats[label, cv2.CC_STAT_WIDTH]),
                "bbox_h": int(stats[label, cv2.CC_STAT_HEIGHT]),
            }
        )
    return labels, comps


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
    threshold = float(config.get("pixel_threshold", 0.5))
    raw = (prob >= threshold).astype(np.uint8)
    work = raw.copy()
    mode = str(config.get("postprocess_mode", "raw"))
    min_area = int(config.get("min_component_area", 0) or 0)
    min_mean_prob = float(config.get("min_component_mean_probability", 0.0) or 0.0)
    morphology = str(config.get("morphology", "none"))
    kernel_size = int(config.get("morph_kernel_size", 3) or 3)
    top_k = config.get("top_k_components", None)
    sort_by = str(config.get("top_k_sort_by", "area"))

    if mode == "raw":
        labels, comps = component_table(work, prob)
        return {"raw_mask": raw, "mask": work, "labels": labels, "components": comps}

    if mode == "morph_area_probability_clean":
        work = apply_morphology(work, morphology, kernel_size)

    labels, comps = component_table(work, prob)
    keep = np.zeros_like(work, dtype=np.uint8)
    selected = []
    for comp in comps:
        ok = True
        if mode in ("min_area_clean", "area_probability_clean", "morph_area_probability_clean", "keep_topk_components"):
            ok = ok and comp["area"] >= min_area
        if mode in ("probability_gated", "area_probability_clean", "morph_area_probability_clean", "keep_topk_components"):
            ok = ok and comp["mean_probability"] >= min_mean_prob
        if ok:
            selected.append(comp)

    if mode == "keep_topk_components" and top_k is not None and not (isinstance(top_k, float) and np.isnan(top_k)):
        key = "mean_probability" if sort_by == "mean_probability" else "area"
        selected = sorted(selected, key=lambda x: x[key], reverse=True)[: int(top_k)]

    for comp in selected:
        keep[labels == comp["label"]] = 1
    labels_clean, comps_clean = component_table(keep, prob)
    return {"raw_mask": raw, "mask": keep, "labels": labels_clean, "components": comps_clean}


def image_score_from_outputs(prob: np.ndarray, raw_mask: np.ndarray, clean_mask: np.ndarray, comps: List[Dict[str, Any]], score_type: str) -> float:
    flat = prob.reshape(-1)
    if score_type == "max_probability":
        return float(flat.max())
    if score_type == "top1_mean_probability":
        k = max(1, int(0.01 * flat.size))
        return float(np.partition(flat, -k)[-k:].mean())
    if score_type == "top5_mean_probability":
        k = max(1, int(0.05 * flat.size))
        return float(np.partition(flat, -k)[-k:].mean())
    if score_type == "pred_mask_ratio_raw":
        return float(raw_mask.mean())
    if score_type == "pred_mask_ratio_clean":
        return float(clean_mask.mean())
    if score_type == "max_component_mean_probability":
        return float(max([c["mean_probability"] for c in comps], default=0.0))
    if score_type == "max_component_area_ratio":
        return float(max([c["area_ratio"] for c in comps], default=0.0))
    raise ValueError(f"Bilinmeyen image_score_type: {score_type}")


def build_postprocess_configs() -> List[Dict[str, Any]]:
    configs = []
    for pt in CFG.pixel_thresholds_grid:
        configs.append({"pixel_threshold": pt, "postprocess_mode": "raw", "min_component_area": 0, "min_component_mean_probability": 0.0, "morphology": "none", "morph_kernel_size": 3, "top_k_components": None, "top_k_sort_by": "area"})
        for area in CFG.min_component_areas:
            configs.append({"pixel_threshold": pt, "postprocess_mode": "min_area_clean", "min_component_area": area, "min_component_mean_probability": 0.0, "morphology": "none", "morph_kernel_size": 3, "top_k_components": None, "top_k_sort_by": "area"})
        for mean_prob in CFG.min_component_mean_probs:
            configs.append({"pixel_threshold": pt, "postprocess_mode": "probability_gated", "min_component_area": 0, "min_component_mean_probability": mean_prob, "morphology": "none", "morph_kernel_size": 3, "top_k_components": None, "top_k_sort_by": "area"})
        for area in CFG.min_component_areas:
            for mean_prob in CFG.min_component_mean_probs:
                configs.append({"pixel_threshold": pt, "postprocess_mode": "area_probability_clean", "min_component_area": area, "min_component_mean_probability": mean_prob, "morphology": "none", "morph_kernel_size": 3, "top_k_components": None, "top_k_sort_by": "area"})
                for morph in CFG.morphologies:
                    for kernel in CFG.morph_kernel_sizes:
                        configs.append({"pixel_threshold": pt, "postprocess_mode": "morph_area_probability_clean", "min_component_area": area, "min_component_mean_probability": mean_prob, "morphology": morph, "morph_kernel_size": kernel, "top_k_components": None, "top_k_sort_by": "area"})
                for top_k in CFG.top_k_components_values:
                    configs.append({"pixel_threshold": pt, "postprocess_mode": "keep_topk_components", "min_component_area": area, "min_component_mean_probability": mean_prob, "morphology": "none", "morph_kernel_size": 3, "top_k_components": top_k, "top_k_sort_by": "area"})
    # Tekrarlari temizle.
    seen, unique = set(), []
    for cfg in configs:
        key = tuple((k, cfg[k]) for k in sorted(cfg))
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def pixel_metrics_from_records(records: List[Dict[str, Any]], preds: List[np.ndarray], probs: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
    gt_all = np.concatenate([rec["mask"].reshape(-1).astype(np.uint8) for rec in records])
    pred_all = np.concatenate([pred.reshape(-1).astype(np.uint8) for pred in preds])
    tp = int(((gt_all == 1) & (pred_all == 1)).sum())
    fp = int(((gt_all == 0) & (pred_all == 1)).sum())
    fn = int(((gt_all == 1) & (pred_all == 0)).sum())
    tn = int(((gt_all == 0) & (pred_all == 0)).sum())
    dice_all = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou_all = (tp + EPS) / (tp + fp + fn + EPS)
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    specificity = (tn + EPS) / (tn + fp + EPS)

    forged_idx = [i for i, r in enumerate(records) if int(r["image_label"]) == 1]
    if forged_idx:
        gt_f = np.concatenate([records[i]["mask"].reshape(-1).astype(np.uint8) for i in forged_idx])
        pred_f = np.concatenate([preds[i].reshape(-1).astype(np.uint8) for i in forged_idx])
        tp_f = int(((gt_f == 1) & (pred_f == 1)).sum())
        fp_f = int(((gt_f == 0) & (pred_f == 1)).sum())
        fn_f = int(((gt_f == 1) & (pred_f == 0)).sum())
        dice_f = (2 * tp_f + EPS) / (2 * tp_f + fp_f + fn_f + EPS)
        iou_f = (tp_f + EPS) / (tp_f + fp_f + fn_f + EPS)
    else:
        dice_f, iou_f = 0.0, 0.0

    out = {
        "dice_all": float(dice_all),
        "dice_forged_only": float(dice_f),
        "iou_all": float(iou_all),
        "iou_forged_only": float(iou_f),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "predicted_positive_pixel_ratio": float(pred_all.mean()),
        "gt_positive_pixel_ratio": float(gt_all.mean()),
    }
    if probs is not None:
        prob_all = np.concatenate([p.reshape(-1).astype(np.float32) for p in probs])
        if len(prob_all) > CFG.max_curve_pixels:
            rng = np.random.default_rng(CFG.seed)
            idx = rng.choice(len(prob_all), size=CFG.max_curve_pixels, replace=False)
            gt_auc, prob_auc = gt_all[idx], prob_all[idx]
        else:
            gt_auc, prob_auc = gt_all, prob_all
        out["auprc_all"] = safe_auc(gt_auc, prob_auc, "pr")
        out["roc_auc_all"] = safe_auc(gt_auc, prob_auc, "roc")
        if forged_idx:
            prob_f = np.concatenate([probs[i].reshape(-1).astype(np.float32) for i in forged_idx])
            if len(prob_f) > CFG.max_curve_pixels:
                rng = np.random.default_rng(CFG.seed)
                idx = rng.choice(len(prob_f), size=CFG.max_curve_pixels, replace=False)
                gt_ff, prob_ff = gt_f[idx], prob_f[idx]
            else:
                gt_ff, prob_ff = gt_f, prob_f
            out["auprc_forged_only"] = safe_auc(gt_ff, prob_ff, "pr")
        else:
            out["auprc_forged_only"] = float("nan")
    return out


def per_image_metrics(records: List[Dict[str, Any]], preds: List[np.ndarray], scores: Sequence[float], image_threshold: float) -> pd.DataFrame:
    rows = []
    for rec, pred, score in zip(records, preds, scores):
        gt = rec["mask"].astype(np.uint8)
        pred = pred.astype(np.uint8)
        rows.append(
            {
                "sample_id": str(rec.get("sample_id", f"{rec['class_name']}__{rec['image_id']}")),
                "image_id": rec["image_id"],
                "image_path": rec["image_path"],
                "class_name": rec["class_name"],
                "image_label": int(rec["image_label"]),
                "gt_area": int(gt.sum()),
                "gt_area_ratio": float(gt.mean()),
                "pred_area": int(pred.sum()),
                "pred_area_ratio": float(pred.mean()),
                "dice": binary_dice_np(gt, pred),
                "iou": binary_iou_np(gt, pred),
                "image_score": float(score),
                "image_pred_label": int(score >= image_threshold),
                "mask_quartile": str(rec.get("mask_quartile", "")),
            }
        )
    return pd.DataFrame(rows)


def summarize_per_image(per_df: pd.DataFrame) -> Dict[str, float]:
    forged = per_df[per_df["image_label"].astype(int) == 1]
    return {
        "mean_dice_forged": float(forged["dice"].mean()) if len(forged) else 0.0,
        "median_dice_forged": float(forged["dice"].median()) if len(forged) else 0.0,
        "mean_iou_forged": float(forged["iou"].mean()) if len(forged) else 0.0,
        "median_iou_forged": float(forged["iou"].median()) if len(forged) else 0.0,
        "dice_lt_005_count": int((forged["dice"] < 0.05).sum()) if len(forged) else 0,
        "dice_lt_010_count": int((forged["dice"] < 0.10).sum()) if len(forged) else 0,
        "dice_lt_025_count": int((forged["dice"] < 0.25).sum()) if len(forged) else 0,
    }


def small_mask_metrics(per_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    forged = per_df[per_df["image_label"].astype(int) == 1]
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = forged[forged["mask_quartile"].astype(str) == q]
        rows.append(
            {
                "mask_quartile": q,
                "n": int(len(sub)),
                "mean_dice": float(sub["dice"].mean()) if len(sub) else 0.0,
                "median_dice": float(sub["dice"].median()) if len(sub) else 0.0,
                "mean_iou": float(sub["iou"].mean()) if len(sub) else 0.0,
                "median_iou": float(sub["iou"].median()) if len(sub) else 0.0,
                "dice_lt_005_count": int((sub["dice"] < 0.05).sum()) if len(sub) else 0,
                "dice_lt_010_count": int((sub["dice"] < 0.10).sum()) if len(sub) else 0,
                "mean_pred_area_ratio": float(sub["pred_area_ratio"].mean()) if len(sub) else 0.0,
                "mean_gt_area_ratio": float(sub["gt_area_ratio"].mean()) if len(sub) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def component_iou_matrix(gt_labels: np.ndarray, gt_count: int, pred_labels: np.ndarray, pred_count: int) -> np.ndarray:
    mat = np.zeros((gt_count, pred_count), dtype=np.float32)
    for gi in range(1, gt_count + 1):
        g = gt_labels == gi
        for pi in range(1, pred_count + 1):
            p = pred_labels == pi
            inter = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            mat[gi - 1, pi - 1] = 0.0 if union == 0 else inter / union
    return mat


def match_components(gt: np.ndarray, pred: np.ndarray, iou_threshold: float) -> Dict[str, Any]:
    gt_count, gt_labels = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    pred_count, pred_labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    gt_n, pred_n = gt_count - 1, pred_count - 1
    if gt_n == 0 or pred_n == 0:
        return {
            "gt_component_count": int(gt_n),
            "pred_component_count": int(pred_n),
            "matched_component_count": 0,
            "component_fp": int(pred_n),
            "component_fn": int(gt_n),
            "avg_matched_iou": 0.0,
        }
    iou_mat = component_iou_matrix(gt_labels, gt_n, pred_labels, pred_n)
    gi, pi = linear_sum_assignment(-iou_mat)
    matched_ious = [float(iou_mat[g, p]) for g, p in zip(gi, pi) if iou_mat[g, p] >= iou_threshold]
    matched = len(matched_ious)
    return {
        "gt_component_count": int(gt_n),
        "pred_component_count": int(pred_n),
        "matched_component_count": int(matched),
        "component_fp": int(pred_n - matched),
        "component_fn": int(gt_n - matched),
        "avg_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def component_metrics(records: List[Dict[str, Any]], preds: List[np.ndarray], iou_threshold: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows, tp, fp, fn = [], 0, 0, 0
    for rec, pred in zip(records, preds):
        comp = match_components(rec["mask"].astype(np.uint8), pred.astype(np.uint8), iou_threshold)
        row = {"image_id": rec["image_id"], "image_label": int(rec["image_label"]), "mask_quartile": str(rec.get("mask_quartile", "")), **comp}
        rows.append(row)
        tp += comp["matched_component_count"]
        fp += comp["component_fp"]
        fn += comp["component_fn"]
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    f1 = (2 * precision * recall + EPS) / (precision + recall + EPS)
    suffix = suffix_for_iou(iou_threshold)
    return (
        {
            f"component_precision_iou{suffix}": float(precision),
            f"component_recall_iou{suffix}": float(recall),
            f"component_f1_iou{suffix}": float(f1),
            f"component_tp_iou{suffix}": int(tp),
            f"component_fp_iou{suffix}": int(fp),
            f"component_fn_iou{suffix}": int(fn),
            f"avg_matched_iou_iou{suffix}": float(np.mean([r["avg_matched_iou"] for r in rows])) if rows else 0.0,
            "avg_pred_component_count": float(np.mean([r["pred_component_count"] for r in rows])) if rows else 0.0,
            "avg_gt_component_count": float(np.mean([r["gt_component_count"] for r in rows])) if rows else 0.0,
        },
        pd.DataFrame(rows),
    )


def authentic_false_alarm_metrics(records: List[Dict[str, Any]], preds: List[np.ndarray], comps_per_image: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    counts, areas = [], []
    alarm = 0
    for rec, pred, comps in zip(records, preds, comps_per_image):
        if int(rec["image_label"]) == 0:
            c = len(comps)
            counts.append(c)
            areas.append(float(pred.mean()))
            alarm += int(c > 0)
    n = len(counts)
    return {
        "authentic_fp_rate": float(alarm / max(n, 1)),
        "authentic_mean_pred_component_count": float(np.mean(counts)) if counts else 0.0,
        "authentic_median_pred_component_count": float(np.median(counts)) if counts else 0.0,
        "authentic_mean_pred_area_ratio": float(np.mean(areas)) if areas else 0.0,
    }


def apply_config(records: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    preds, raw_preds, scores, comps_per_image = [], [], [], []
    for rec in records:
        out = postprocess_probability_map(rec["prob"], config)
        preds.append(out["mask"].astype(np.uint8))
        raw_preds.append(out["raw_mask"].astype(np.uint8))
        comps_per_image.append(out["components"])
        scores.append(image_score_from_outputs(rec["prob"], out["raw_mask"], out["mask"], out["components"], config["image_score_type"]))
    return {"preds": preds, "raw_preds": raw_preds, "scores": np.asarray(scores, dtype=np.float32), "components_per_image": comps_per_image}


def choose_best_image_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best = {"image_f1": -1.0}
    best_thr = 0.5
    for thr in CFG.image_thresholds:
        pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["image_f1"]:
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            best = {
                "image_accuracy": accuracy_score(y_true, pred),
                "image_precision": precision_score(y_true, pred, zero_division=0),
                "image_recall": recall_score(y_true, pred, zero_division=0),
                "image_sensitivity": recall_score(y_true, pred, zero_division=0),
                "image_specificity": tn / max(tn + fp, 1),
                "image_f1": f1,
                "image_tp": int(tp),
                "image_fn": int(fn),
                "image_tn": int(tn),
                "image_fp": int(fp),
            }
            best_thr = float(thr)
    best["image_roc_auc"] = safe_auc(y_true, scores, "roc")
    best["image_auprc"] = safe_auc(y_true, scores, "pr")
    return best_thr, {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in best.items()}


def evaluate_records(records: List[Dict[str, Any]], config: Dict[str, Any], save_prefix: Optional[Path] = None) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs = apply_config(records, config)
    y_true = np.array([int(r["image_label"]) for r in records], dtype=int)
    image_threshold = float(config.get("image_threshold", 0.5))
    per_df = per_image_metrics(records, outputs["preds"], outputs["scores"], image_threshold)
    pix = pixel_metrics_from_records(records, outputs["preds"], probs=[r["prob"] for r in records])
    per_summary = summarize_per_image(per_df)
    small_df = small_mask_metrics(per_df)
    image_pred = (outputs["scores"] >= image_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, image_pred, labels=[0, 1]).ravel()
    image_metrics = {
        "image_accuracy": float(accuracy_score(y_true, image_pred)),
        "image_precision": float(precision_score(y_true, image_pred, zero_division=0)),
        "image_recall": float(recall_score(y_true, image_pred, zero_division=0)),
        "image_sensitivity": float(recall_score(y_true, image_pred, zero_division=0)),
        "image_specificity": float(tn / max(tn + fp, 1)),
        "image_f1": float(f1_score(y_true, image_pred, zero_division=0)),
        "image_roc_auc": safe_auc(y_true, outputs["scores"], "roc"),
        "image_auprc": safe_auc(y_true, outputs["scores"], "pr"),
        "image_tp": int(tp),
        "image_fn": int(fn),
        "image_tn": int(tn),
        "image_fp": int(fp),
    }
    try:
        image_metrics["image_brier"] = float(brier_score_loss(y_true, np.clip(outputs["scores"], 0, 1)))
    except Exception:
        image_metrics["image_brier"] = float("nan")
    ece, calib_df = ece_score(y_true, np.clip(outputs["scores"], 0, 1), n_bins=10)
    image_metrics["image_ece_10bin"] = ece

    comp_all, comp_details = {}, None
    for thr in CFG.component_iou_thresholds:
        comp_m, comp_df = component_metrics(records, outputs["preds"], float(thr))
        comp_all.update(comp_m)
        if abs(thr - 0.10) < 1e-6:
            comp_details = comp_df
    auth_m = authentic_false_alarm_metrics(records, outputs["preds"], outputs["components_per_image"])

    q = {f"q{row.mask_quartile[-1]}_dice": row.mean_dice for row in small_df.itertuples() if str(row.mask_quartile).startswith("Q")}
    metrics = {**pix, **per_summary, **image_metrics, **comp_all, **auth_m, **q}
    if save_prefix is not None:
        per_df.to_csv(save_prefix.with_name(save_prefix.name + "_per_image.csv"), index=False)
        small_df.to_csv(save_prefix.with_name(save_prefix.name + "_small_bins.csv"), index=False)
        comp_details.to_csv(save_prefix.with_name(save_prefix.name + "_component_details.csv"), index=False)
        calib_df.to_csv(save_prefix.with_name(save_prefix.name + "_calibration.csv"), index=False)
    return metrics, per_df, small_df, comp_details if comp_details is not None else pd.DataFrame(), calib_df


def validation_grid_search(records: List[Dict[str, Any]], out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    y_true = np.array([int(r["image_label"]) for r in records], dtype=int)
    for base_cfg in tqdm(build_postprocess_configs(), desc="validation grid"):
        # Segmentasyon ciktilari image score type'tan bagimsiz oldugu icin once bir kez hesaplanir.
        temp_cfg = dict(base_cfg)
        temp_cfg["image_score_type"] = "max_probability"
        out = apply_config(records, temp_cfg)
        pix = pixel_metrics_from_records(records, out["preds"])
        per_df = per_image_metrics(records, out["preds"], np.zeros(len(records)), 0.5)
        small_df = small_mask_metrics(per_df)
        q1 = float(small_df.loc[small_df["mask_quartile"] == "Q1", "mean_dice"].iloc[0])
        q2 = float(small_df.loc[small_df["mask_quartile"] == "Q2", "mean_dice"].iloc[0])
        comp010, _ = component_metrics(records, out["preds"], 0.10)
        auth_m = authentic_false_alarm_metrics(records, out["preds"], out["components_per_image"])
        for score_type in CFG.image_score_types:
            scores = np.array(
                [
                    image_score_from_outputs(rec["prob"], raw, pred, comps, score_type)
                    for rec, raw, pred, comps in zip(records, out["raw_preds"], out["preds"], out["components_per_image"])
                ],
                dtype=np.float32,
            )
            best_thr, img_m = choose_best_image_threshold(y_true, scores)
            row = {
                **base_cfg,
                "image_score_type": score_type,
                "image_threshold": best_thr,
                "val_dice_all": pix["dice_all"],
                "val_forged_dice": pix["dice_forged_only"],
                "val_forged_iou": pix["iou_forged_only"],
                "val_pixel_precision": pix["precision"],
                "val_pixel_recall": pix["recall"],
                "val_pixel_specificity": pix["specificity"],
                "val_small_q1_dice": q1,
                "val_small_q2_dice": q2,
                "val_component_f1_iou010": comp010["component_f1_iou010"],
                "val_image_f1": img_m["image_f1"],
                "val_image_recall": img_m["image_recall"],
                "val_image_specificity": img_m["image_specificity"],
                "val_authentic_fp_rate": auth_m["authentic_fp_rate"],
            }
            row["balanced_score"] = (
                0.30 * row["val_forged_dice"]
                + 0.25 * row["val_small_q1_dice"]
                + 0.20 * row["val_component_f1_iou010"]
                + 0.15 * row["val_image_f1"]
                + 0.10 * (1 - row["val_authentic_fp_rate"])
            )
            row["small_object_score"] = (
                0.40 * row["val_small_q1_dice"]
                + 0.25 * row["val_small_q2_dice"]
                + 0.20 * row["val_component_f1_iou010"]
                + 0.15 * (1 - row["val_authentic_fp_rate"])
            )
            rows.append(row)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(out_dir / "val_grid_search_all.csv", index=False)
    top = grid_df.sort_values(["balanced_score", "val_forged_dice"], ascending=[False, False]).head(50)
    top.to_csv(out_dir / "val_grid_search_top50.csv", index=False)
    return grid_df, top


def select_strategy_configs(grid_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    selected = []

    def take(strategy: str, df: pd.DataFrame, by: List[str], asc: List[bool]) -> None:
        if df.empty:
            return
        row = df.sort_values(by, ascending=asc).iloc[0].to_dict()
        row["strategy"] = strategy
        selected.append(row)

    take("best_forged_dice", grid_df, ["val_forged_dice", "val_small_q1_dice"], [False, False])
    take("best_small_mask_q1_dice", grid_df, ["val_small_q1_dice", "val_small_q2_dice"], [False, False])
    take("balanced_final_score", grid_df, ["balanced_score", "val_small_q1_dice"], [False, False])
    low = grid_df[grid_df["val_authentic_fp_rate"] <= 0.25]
    if not low.empty:
        take("low_false_alarm", low, ["val_component_f1_iou010", "val_forged_dice"], [False, False])
    else:
        take("low_false_alarm", grid_df.sort_values("val_authentic_fp_rate").head(10), ["val_forged_dice"], [False])
    take("small_object_practical", grid_df, ["small_object_score", "val_forged_dice"], [False, False])

    selected_df = pd.DataFrame(selected)
    cols = ["strategy"] + [c for c in selected_df.columns if c != "strategy"]
    selected_df = selected_df[cols]
    selected_df.to_csv(out_dir / "selected_configs.csv", index=False)
    return selected_df


# %% [markdown]
# ## 12. Test Evaluation

# %%
def config_from_selected_row(row: pd.Series) -> Dict[str, Any]:
    keys = [
        "pixel_threshold",
        "image_score_type",
        "image_threshold",
        "postprocess_mode",
        "min_component_area",
        "min_component_mean_probability",
        "morphology",
        "morph_kernel_size",
        "top_k_components",
        "top_k_sort_by",
    ]
    cfg = {k: row[k] for k in keys if k in row.index}
    if pd.isna(cfg.get("top_k_components", None)):
        cfg["top_k_components"] = None
    return cfg


def evaluate_selected_strategies(model_name: str, records: List[Dict[str, Any]], selected_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in selected_df.iterrows():
        strategy = str(row["strategy"])
        cfg = config_from_selected_row(row)
        metrics, per_df, small_df, comp_df, calib_df = evaluate_records(records, cfg)
        result = {"model_name": model_name, "strategy": strategy, **cfg, **metrics}
        result["final_score"] = (
            0.25 * result.get("dice_forged_only", 0.0)
            + 0.25 * result.get("q1_dice", 0.0)
            + 0.15 * result.get("q2_dice", 0.0)
            + 0.15 * result.get("component_f1_iou010", 0.0)
            + 0.10 * result.get("image_f1", 0.0)
            + 0.10 * (1 - result.get("authentic_fp_rate", 1.0))
        )
        rows.append(result)
        per_df.to_csv(out_dir / f"test_per_image_metrics_{strategy}.csv", index=False)
        comp_df.to_csv(out_dir / f"test_component_details_{strategy}.csv", index=False)
        small_df.to_csv(out_dir / f"small_mask_bin_metrics_{strategy}.csv", index=False)
        calib_df.to_csv(out_dir / f"image_level_calibration_{strategy}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_results_by_strategy.csv", index=False)
    return df


# %% [markdown]
# ## 13. Small-mask Analysis / 14. Component-aware / 15. Calibration
#
# Bu metrikler `evaluate_records` ve `evaluate_selected_strategies` fonksiyonlari icinde
# uretilir. Ayrica asagidaki grafik ve failure-case bloklarinda kullanilir.

# %% [markdown]
# ## 16. Experiment 5 Comparison

# %%
EXP5_MODEL_MAP = {
    "efficientnetb0_unet_rgb_384_smallmask": "efficientnetb0_unet_rgb_full",
    "efficientnetb0_unet_rgb_384_smallmask_oversampling": "efficientnetb0_unet_rgb_full",
    "segformer_b0_rgb_384_smallmask": "segformer_b0_rgb_full",
}


def load_experiment5_results() -> pd.DataFrame:
    if EXP5_ROOT is None:
        print("[warn] Deney 5 klasoru bulunamadi. Karsilastirma bos uretilecek.")
        return pd.DataFrame()
    candidates = [
        EXP5_ROOT / "test_results_all_strategies.csv",
        EXP5_ROOT / "experiment5_all_results.csv",
    ]
    rows = []
    for path in candidates:
        if path.exists():
            rows.append(pd.read_csv(path))
    if not rows:
        for model_dir in EXP5_ROOT.iterdir():
            path = model_dir / "test_results_by_strategy.csv"
            if path.exists():
                rows.append(pd.read_csv(path))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compare_with_experiment5(exp6_results: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    exp5 = load_experiment5_results()
    rows = []
    if exp5.empty:
        md = "# Deney 6 vs Deney 5 Karsilastirmasi\n\nDeney 5 sonuc CSV'leri bulunamadigi icin otomatik karsilastirma yapilamadi.\n"
        return pd.DataFrame(), md
    for _, r6 in exp6_results.iterrows():
        old_model = EXP5_MODEL_MAP.get(r6["model_name"])
        if old_model is None:
            continue
        candidates = exp5[exp5["model_name"].astype(str) == old_model].copy()
        if "strategy" in candidates.columns and str(r6["strategy"]) in set(candidates["strategy"].astype(str)):
            candidates = candidates[candidates["strategy"].astype(str) == str(r6["strategy"])]
        elif "strategy" in candidates.columns and "balanced_final_score" in set(candidates["strategy"].astype(str)):
            candidates = candidates[candidates["strategy"].astype(str) == "balanced_final_score"]
        if candidates.empty:
            continue
        r5 = candidates.iloc[0]
        row = {
            "experiment6_model": r6["model_name"],
            "experiment6_strategy": r6["strategy"],
            "experiment5_model": old_model,
            "experiment5_strategy": r5.get("strategy", ""),
        }
        metric_map = {
            "forged_dice": ("dice_forged_only", "dice_forged_only"),
            "forged_iou": ("iou_forged_only", "iou_forged_only"),
            "Q1 Dice": ("q1_dice", "q1_dice"),
            "Q2 Dice": ("q2_dice", "q2_dice"),
            "Q3 Dice": ("q3_dice", "q3_dice"),
            "Q4 Dice": ("q4_dice", "q4_dice"),
            "component F1 @0.10": ("component_f1_iou010", "component_f1_iou010"),
            "authentic FP rate": ("authentic_fp_rate", "authentic_fp_rate"),
            "image F1": ("image_f1", "image_f1"),
            "training time": ("training_time_sec_total", "training_time_sec_total"),
            "inference time per image": ("inference_time_per_image_sec", "inference_time_per_image_sec"),
        }
        for label, (c6, c5) in metric_map.items():
            v6 = float(r6[c6]) if c6 in r6.index and pd.notna(r6[c6]) else np.nan
            v5 = float(r5[c5]) if c5 in r5.index and pd.notna(r5[c5]) else np.nan
            row[f"exp6_{label}"] = v6
            row[f"exp5_{label}"] = v5
            row[f"delta_{label}"] = v6 - v5 if pd.notna(v6) and pd.notna(v5) else np.nan
        criteria = {
            "Q1 Dice artisi": row.get("delta_Q1 Dice", np.nan) > 0,
            "Q2 Dice artisi": row.get("delta_Q2 Dice", np.nan) > 0,
            "Dice <0.05 azalmasi": False,
            "Forged Dice ciddi dusmedi": row.get("delta_forged_dice", np.nan) >= -0.02,
            "Authentic FP kabul edilebilir": row.get("exp6_authentic FP rate", 1.0) <= 0.35,
            "Component F1 dusmedi": row.get("delta_component F1 @0.10", np.nan) >= -0.02,
        }
        row["success_criteria_met"] = int(sum(bool(v) for v in criteria.values()))
        row["success"] = row["success_criteria_met"] >= 2
        row["criteria_json"] = json.dumps(criteria, ensure_ascii=False)
        rows.append(row)
    comp = pd.DataFrame(rows)
    lines = ["# Deney 6 vs Deney 5 Karsilastirmasi", ""]
    if comp.empty:
        lines.append("Eslesen Deney 5 satiri bulunamadi.")
    else:
        lines.append("Deney 6 basari kosulu: alti kriterden en az ikisinin saglanmasi.")
        lines.append("")
        show_cols = [
            "experiment6_model",
            "experiment6_strategy",
            "delta_Q1 Dice",
            "delta_Q2 Dice",
            "delta_forged_dice",
            "delta_component F1 @0.10",
            "exp6_authentic FP rate",
            "success_criteria_met",
            "success",
        ]
        lines.append(comp[show_cols].to_markdown(index=False))
    return comp, "\n".join(lines) + "\n"


# %% [markdown]
# ## 17. Visualization

# %%
def ensure_plots_dir(out_dir: Path) -> Path:
    path = out_dir / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    if df.empty or y not in df.columns:
        return
    plt.figure(figsize=(max(8, len(df) * 0.8), 4))
    labels = df[x].astype(str).tolist()
    plt.bar(np.arange(len(df)), df[y].astype(float).values)
    plt.xticks(np.arange(len(df)), labels, rotation=45, ha="right")
    plt.ylabel(y); plt.title(title); plt.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def save_scatter(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(6, 5))
    for _, row in df.iterrows():
        plt.scatter(row[x], row[y], s=60)
        plt.text(row[x], row[y], str(row.get("model_strategy", row.get("strategy", "")))[:28], fontsize=8)
    plt.xlabel(x); plt.ylabel(y); plt.title(title); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def save_main_plots(all_results: pd.DataFrame, output_root: Path) -> None:
    if all_results.empty:
        return
    df = all_results.copy()
    df["model_strategy"] = df["model_name"].astype(str) + "\n" + df["strategy"].astype(str)
    plt.figure(figsize=(max(9, len(df) * 0.9), 4.5))
    x = np.arange(len(df))
    plt.bar(x - 0.18, df["q1_dice"].astype(float).values, width=0.36, label="Q1")
    plt.bar(x + 0.18, df["q2_dice"].astype(float).values, width=0.36, label="Q2")
    plt.xticks(x, df["model_strategy"].astype(str).tolist(), rotation=45, ha="right")
    plt.ylabel("Mean Dice"); plt.title("Q1/Q2 Dice comparison"); plt.grid(axis="y", alpha=0.25); plt.legend()
    plt.tight_layout(); plt.savefig(output_root / "bar_q1_q2_dice_comparison.png", dpi=180); plt.close()
    save_bar(df, "model_strategy", "dice_forged_only", "Forged Dice comparison", output_root / "bar_forged_dice_comparison.png")
    save_bar(df, "model_strategy", "component_f1_iou010", "Component F1 @0.10 comparison", output_root / "bar_component_f1_comparison.png")
    save_bar(df, "model_strategy", "authentic_fp_rate", "Authentic FP rate comparison", output_root / "bar_authentic_fp_rate_comparison.png")
    save_scatter(df, "q1_dice", "authentic_fp_rate", "Q1 Dice vs Authentic FP", output_root / "scatter_q1_dice_vs_authfp.png")
    save_scatter(df, "dice_forged_only", "q1_dice", "Forged Dice vs Q1 Dice", output_root / "scatter_forged_dice_vs_q1_dice.png")
    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        vals = [row.get(f"q{i}_dice", np.nan) for i in [1, 2, 3, 4]]
        plt.plot(["Q1", "Q2", "Q3", "Q4"], vals, marker="o", label=f"{row['model_name']}:{row['strategy']}")
    plt.ylabel("Mean Dice"); plt.title("Small-mask quartile dice curves"); plt.grid(alpha=0.25); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(output_root / "small_mask_quartile_dice_curves.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 5))
    for model_name in df["model_name"].astype(str).unique():
        grid_path = output_root / model_name / "val_grid_search_all.csv"
        if not grid_path.exists():
            continue
        grid = pd.read_csv(grid_path)
        if grid.empty:
            continue
        summary = grid.groupby("pixel_threshold", as_index=False).agg(q1=("val_small_q1_dice", "max"), forged=("val_forged_dice", "max"))
        plt.plot(summary["pixel_threshold"], summary["q1"], marker="o", label=f"{model_name} Q1")
        plt.plot(summary["pixel_threshold"], summary["forged"], marker="x", linestyle="--", label=f"{model_name} forged")
    plt.xlabel("Pixel threshold"); plt.ylabel("Validation score"); plt.title("Threshold tradeoff small-mask"); plt.grid(alpha=0.25); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(output_root / "threshold_tradeoff_smallmask.png", dpi=180); plt.close()

    save_image_level_curves_from_outputs(df, output_root)


def save_image_level_curves_from_outputs(df: pd.DataFrame, output_root: Path) -> None:
    curve_rows = []
    for _, row in df.iterrows():
        model_name = str(row["model_name"])
        strategy = str(row["strategy"])
        per_path = output_root / model_name / f"test_per_image_metrics_{strategy}.csv"
        calib_path = output_root / model_name / f"image_level_calibration_{strategy}.csv"
        if not per_path.exists():
            continue
        per = pd.read_csv(per_path)
        if per.empty or per["image_label"].nunique() < 2:
            continue
        label = f"{model_name}:{strategy}"
        y = per["image_label"].astype(int).values
        s = per["image_score"].astype(float).values
        curve_rows.append((label, y, s, calib_path))

    if curve_rows:
        plt.figure(figsize=(6, 5))
        for label, y, s, _ in curve_rows:
            fpr, tpr, _ = roc_curve(y, s)
            auc = safe_auc(y, s, "roc")
            plt.plot(fpr, tpr, label=f"{label} ({auc:.3f})")
        plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Image-level ROC")
        plt.grid(alpha=0.25); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(output_root / "roc_curves_image_level.png", dpi=180); plt.close()

        plt.figure(figsize=(6, 5))
        for label, y, s, _ in curve_rows:
            precision, recall, _ = precision_recall_curve(y, s)
            ap = safe_auc(y, s, "pr")
            plt.plot(recall, precision, label=f"{label} ({ap:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Image-level PR")
        plt.grid(alpha=0.25); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(output_root / "pr_curves_image_level.png", dpi=180); plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        for label, _, _, calib_path in curve_rows:
            if not calib_path.exists():
                continue
            calib = pd.read_csv(calib_path).dropna(subset=["confidence", "accuracy"])
            plt.plot(calib["confidence"], calib["accuracy"], marker="o", label=label)
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability diagram")
        plt.grid(alpha=0.25); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(output_root / "reliability_diagram_image_level.png", dpi=180); plt.close()

        n = len(curve_rows)
        fig, axes = plt.subplots(1, n, figsize=(max(4, 3.2 * n), 3.2))
        if n == 1:
            axes = [axes]
        for ax, (label, y, s, _) in zip(axes, curve_rows):
            # Per-image CSV zaten validation'da secilmis image_threshold ile pred label tasir.
            per_path = output_root / label.split(":")[0] / f"test_per_image_metrics_{label.split(':', 1)[1]}.csv"
            per = pd.read_csv(per_path)
            cm = confusion_matrix(per["image_label"].astype(int), per["image_pred_label"].astype(int), labels=[0, 1])
            ax.imshow(cm, cmap="Blues")
            ax.set_title(label[:34], fontsize=8)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xlabel("Pred"); ax.set_ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", color="black")
        plt.tight_layout(); plt.savefig(output_root / "confusion_matrices_image_level.png", dpi=180); plt.close()


def plot_validation_curves(grid_df: pd.DataFrame, out_dir: Path) -> None:
    plots = ensure_plots_dir(out_dir)
    if grid_df.empty:
        return
    summary = grid_df.groupby("pixel_threshold", as_index=False).agg(
        q1=("val_small_q1_dice", "max"),
        q2=("val_small_q2_dice", "max"),
        forged=("val_forged_dice", "max"),
    )
    plt.figure(figsize=(7, 4))
    plt.plot(summary["pixel_threshold"], summary["q1"], marker="o", label="Q1 Dice")
    plt.plot(summary["pixel_threshold"], summary["q2"], marker="o", label="Q2 Dice")
    plt.plot(summary["pixel_threshold"], summary["forged"], marker="o", label="Forged Dice")
    plt.xlabel("Pixel threshold"); plt.ylabel("Best validation score"); plt.grid(alpha=0.25); plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "validation_threshold_curve.png", dpi=180)
    plt.savefig(out_dir / "threshold_tradeoff_smallmask.png", dpi=180)
    plt.close()

    area_df = grid_df.groupby("min_component_area", as_index=False).agg(
        auth_fp=("val_authentic_fp_rate", "min"),
        comp=("val_component_f1_iou010", "max"),
    )
    plt.figure(figsize=(7, 4))
    plt.plot(area_df["min_component_area"], area_df["auth_fp"], marker="o", label="auth FP")
    plt.plot(area_df["min_component_area"], area_df["comp"], marker="o", label="component F1")
    plt.xlabel("min_component_area"); plt.grid(alpha=0.25); plt.legend()
    plt.tight_layout(); plt.savefig(plots / "authfp_component_vs_min_area.png", dpi=180); plt.close()


def denormalize_image(tensor_or_image: np.ndarray) -> np.ndarray:
    img = tensor_or_image.astype(np.float32)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    out = image.copy()
    m = mask.astype(bool)
    out[m] = 0.55 * out[m] + 0.45 * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 1)


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    out = np.zeros((*gt.shape, 3), dtype=np.float32)
    out[np.logical_and(gt, pred)] = [0.0, 0.85, 0.15]   # TP
    out[np.logical_and(~gt, pred)] = [1.0, 0.15, 0.15]  # FP
    out[np.logical_and(gt, ~pred)] = [0.15, 0.35, 1.0]  # FN
    return out


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask_float = (mask > 0).astype(np.float32)
    return np.repeat(mask_float[..., None], 3, axis=2)


def sample_key_from_mapping(row: Dict[str, Any]) -> str:
    if row.get("sample_id", None) not in (None, "") and not pd.isna(row.get("sample_id")):
        return str(row["sample_id"])
    return f"{row.get('class_name')}__{row.get('image_id')}"


def save_prediction_grid(records: List[Dict[str, Any]], config: Dict[str, Any], out_path: Path, indices: Sequence[int], title: str) -> None:
    if not indices:
        return
    n = len(indices)
    cols = 7
    fig, axes = plt.subplots(n, cols, figsize=(cols * 2.2, n * 2.0))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, idx in enumerate(indices):
        rec = records[int(idx)]
        img = load_image_rgb(rec["image_path"])
        img = cv2.resize(img, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        gt = rec["mask"].astype(np.uint8)
        out = postprocess_probability_map(rec["prob"], config)
        raw, pred, prob = out["raw_mask"], out["mask"], rec["prob"]
        panels = [
            img,
            mask_to_rgb(gt),
            plt.cm.magma(prob)[..., :3],
            mask_to_rgb(raw),
            mask_to_rgb(pred),
            make_overlay(img, pred, (1.0, 0.1, 0.1)),
            error_map(gt, pred),
        ]
        subtitles = ["image", "gt", "prob", "raw", "final", "overlay", "error"]
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel, vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(subtitles[c], fontsize=9)
        axes[r, 0].set_ylabel(sample_key_from_mapping(rec), fontsize=8)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


# %% [markdown]
# ## 18. Failure Case Analysis

# %%
def save_failure_cases(model_name: str, records: List[Dict[str, Any]], config: Dict[str, Any], per_df: pd.DataFrame, out_dir: Path) -> None:
    failure_dir = out_dir / "failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)
    forged = per_df[per_df["image_label"].astype(int) == 1].copy()
    authentic = per_df[per_df["image_label"].astype(int) == 0].copy()
    groups = {
        "best_cases_forged": forged.sort_values("dice", ascending=False).head(12),
        "low_dice_forged": forged.sort_values("dice", ascending=True).head(12),
        "false_positive_authentic": authentic[authentic["image_pred_label"].astype(int) == 1].sort_values("pred_area", ascending=False).head(12),
        "false_negative_forged": forged[forged["image_pred_label"].astype(int) == 0].sort_values("dice", ascending=True).head(12),
        "small_mask_failures": forged[forged["mask_quartile"].astype(str).isin(["Q1", "Q2"])].sort_values("dice", ascending=True).head(12),
        "large_mask_failures": forged[forged["mask_quartile"].astype(str) == "Q4"].sort_values("dice", ascending=True).head(12),
    }
    id_to_idx = {sample_key_from_mapping(r): i for i, r in enumerate(records)}
    for name, df in groups.items():
        df.to_csv(failure_dir / f"{name}.csv", index=False)
        indices = [id_to_idx[key] for key in [sample_key_from_mapping(row.to_dict()) for _, row in df.iterrows()] if key in id_to_idx]
        save_prediction_grid(records, config, failure_dir / f"{name}.png", indices[:8], f"{model_name} - {name}")


def save_case_grids(records: List[Dict[str, Any]], config: Dict[str, Any], per_df: pd.DataFrame, out_dir: Path) -> None:
    id_to_idx = {sample_key_from_mapping(r): i for i, r in enumerate(records)}
    forged = per_df[per_df["image_label"].astype(int) == 1]
    q1 = forged[forged["mask_quartile"].astype(str) == "Q1"]
    best = [id_to_idx[key] for key in [sample_key_from_mapping(row.to_dict()) for _, row in q1.sort_values("dice", ascending=False).head(8).iterrows()] if key in id_to_idx]
    worst = [id_to_idx[key] for key in [sample_key_from_mapping(row.to_dict()) for _, row in q1.sort_values("dice", ascending=True).head(8).iterrows()] if key in id_to_idx]
    fn = [id_to_idx[key] for key in [sample_key_from_mapping(row.to_dict()) for _, row in q1[q1["image_pred_label"].astype(int) == 0].sort_values("dice").head(8).iterrows()] if key in id_to_idx]
    sample = [id_to_idx[key] for key in [sample_key_from_mapping(row.to_dict()) for _, row in per_df.sort_values(["image_label", "dice"], ascending=[False, True]).head(CFG.max_visual_examples).iterrows()] if key in id_to_idx]
    save_prediction_grid(records, config, out_dir / "prediction_examples_best_strategy.png", sample, "Prediction examples - balanced strategy")
    save_prediction_grid(records, config, out_dir / "q1_best_cases.png", best, "Q1 best cases")
    save_prediction_grid(records, config, out_dir / "q1_worst_cases.png", worst, "Q1 worst cases")
    save_prediction_grid(records, config, out_dir / "q1_false_negative_cases.png", fn, "Q1 false negative cases")


# %% [markdown]
# ## 19. Optional Robustness

# %%
def distort_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "jpeg90":
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if mode == "jpeg70":
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if mode == "gaussian_blur":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if mode == "gaussian_noise":
        noise = np.random.default_rng(CFG.seed).normal(0, 8, size=image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


class RobustnessDataset(ForgeryDataset):
    def __init__(self, df: pd.DataFrame, image_size: int, corruption: str):
        super().__init__(df, image_size, "test", augment=False)
        self.corruption = corruption

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image = distort_image(load_image_rgb(row["image_path"]), self.corruption)
        mask = load_mask_array(row["mask_path"], image.shape[:2]) if int(row["image_label"]) == 1 else np.zeros(image.shape[:2], np.float32)
        augmented = self.transforms(image=image, mask=mask)
        image_tensor = torch.from_numpy(augmented["image"].astype(np.float32).transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy((augmented["mask"] > 0.5).astype(np.float32)[None]).float()
        return {"image": image_tensor, "mask": mask_tensor, "image_id": str(row["image_id"]), "image_label": torch.tensor(int(row["image_label"])), "loss_weight": torch.tensor(1.0), "idx": torch.tensor(idx)}


def run_robustness(model: nn.Module, exp: ExperimentConfig, best_config: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    rows = []
    for corruption in ["jpeg90", "jpeg70", "gaussian_blur", "gaussian_noise"]:
        ds = RobustnessDataset(test_df, exp.image_size, corruption)
        loader = DataLoader(ds, batch_size=auto_batch_size(exp), shuffle=False, num_workers=CFG.num_workers, pin_memory=DEVICE.type == "cuda")
        records, probs = generate_predictions(model, loader, test_df, out_dir, f"robustness_{corruption}")
        records = attach_probs(records, probs)
        metrics, _, _, _, _ = evaluate_records(records, best_config)
        rows.append(
            {
                "corruption": corruption,
                "forged_dice": metrics.get("dice_forged_only", 0.0),
                "forged_iou": metrics.get("iou_forged_only", 0.0),
                "Q1 Dice": metrics.get("q1_dice", 0.0),
                "Q2 Dice": metrics.get("q2_dice", 0.0),
                "component F1": metrics.get("component_f1_iou010", 0.0),
                "authentic FP rate": metrics.get("authentic_fp_rate", 0.0),
                "image F1": metrics.get("image_f1", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_ROOT / "robustness_metrics_experiment6_best.csv", index=False)
    return df


# %% [markdown]
# ## 20. Report Generation

# %%
def write_model_report(out_dir: Path, exp: ExperimentConfig, test_results: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    lines = [
        f"# {exp.name} Raporu",
        "",
        "## Model",
        f"- Model tipi: {exp.model_type}",
        f"- Backbone/encoder: {exp.encoder_or_backbone}",
        f"- Girdi cozumurlugu: {exp.image_size}x{exp.image_size}",
        f"- Small-mask oversampling: {exp.use_small_mask_oversampling}",
        f"- Small-mask loss weighting: {exp.use_small_mask_loss_weight}",
        "",
        "## Validation Secimleri",
        selected_df.to_markdown(index=False) if not selected_df.empty else "Secim yok.",
        "",
        "## Test Sonuclari",
        test_results.to_markdown(index=False) if not test_results.empty else "Test sonucu yok.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "model_name": exp.name,
        "best_strategy_by_final_score": test_results.sort_values("final_score", ascending=False).iloc[0].to_dict() if not test_results.empty else None,
    }
    save_json(summary, out_dir / "summary.json")


def write_final_report(all_results: pd.DataFrame, ranking: pd.DataFrame, comparison_md: str) -> None:
    lines = [
        "# Deney 6 Small-mask Improvement Raporu",
        "",
        "## 1. Deneyin amaci",
        "Deney 6, 256x256 Deney 5 sonucunda zayif kalan kucuk sahtecilik bolgelerini 384x384 egitim ve validation-only post-processing secimi ile iyilestirmeyi hedefler.",
        "",
        "## 2. Deney 5'ten gelen problem",
        "Q1 en kucuk maske grubunda modeller anlamli Dice uretmekte zorlanmistir. Bu nedenle final skor Q1 ve Q2 Dice'a yuksek agirlik verir.",
        "",
        "## 3. Kullanilan modeller",
        "- EfficientNetB0-UNet RGB 384",
        "- SegFormer-B0 RGB 384",
        "- Opsiyonel EfficientNetB0-UNet RGB 384 + small-mask oversampling",
        "",
        "## 4. 384x384 egitim protokolu",
        "AdamW, BCEWithLogitsLoss + DiceLoss, ReduceLROnPlateau, early stopping ve CUDA varsa AMP kullanilmistir.",
        "",
        "## 5. Kucuk maske gruplarinin tanimi",
        "Forged goruntuler gt_area degerine gore split bazinda Q1/Q2/Q3/Q4 ceyreklerine ayrilmistir.",
        "",
        "## 6. Egitim sonuclari",
        "Her modelin epoch metrikleri kendi `metrics.csv` dosyasina yazilmistir.",
        "",
        "## 7. Validation threshold ve post-processing secimi",
        "Tum pixel threshold, image score ve post-processing secimleri validation setinde yapilmis; testte yeniden secim yapilmamistir.",
        "",
        "## 8. Test sonuclari",
        all_results.to_markdown(index=False) if not all_results.empty else "Test sonucu yok.",
        "",
        "## 9. Q1/Q2 kucuk maske analizi",
        ranking[["model_name", "strategy", "q1_dice", "q2_dice", "dice_forged_only", "authentic_fp_rate", "experiment6_final_score"]].to_markdown(index=False) if not ranking.empty else "Siralama yok.",
        "",
        "## 10. Deney 5 ile karsilastirma",
        comparison_md,
        "",
        "## 11. Component-aware sonuclar",
        ranking[["model_name", "strategy", "component_f1_iou010", "authentic_fp_rate"]].to_markdown(index=False) if not ranking.empty else "Component sonucu yok.",
        "",
        "## 12. Image-level calibration",
        "Her strateji icin Brier score, ECE ve reliability binleri model klasorlerinde kaydedilmistir.",
        "",
        "## 13. Failure case analizi",
        "Best/low dice/false positive/false negative/small-mask/large-mask failure case CSV ve PNG gridleri model klasorlerinde uretildi.",
        "",
        "## 14. Deney 6 basari kriterleri",
        "Basari, Deney 5'e gore Q1/Q2 artisi, dusuk Dice sayisinin azalmasi, forged Dice'in ciddi dusmemesi, authentic FP'nin kabul edilebilir kalmasi ve component F1'in korunmasi uzerinden degerlendirilir.",
        "",
        "## 15. Final karar",
    ]
    if ranking.empty:
        lines.append("Final aday secilemedi; hata loglarini kontrol edin.")
    else:
        best = ranking.iloc[0]
        lines.append(
            f"Deney 6 final aday skoru ile en guclu aday `{best['model_name']}` / `{best['strategy']}` oldu. "
            "Deney 5 karsilastirmasindaki basari kriterleriyle birlikte final pipeline karari verilmelidir."
        )
    lines.extend(
        [
            "",
            "## 16. Sonraki adim",
            "Deney 6 Q1/Q2 kazanci saglarsa final robustness ve genisletilmis failure case analizine gecilebilir. Saglamazsa Deney 7 DINOv2 limited unfreeze halen anlamli bir sonraki deneydir.",
            "",
        ]
    )
    (OUTPUT_ROOT / "experiment6_report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "best_candidate": ranking.iloc[0].to_dict() if not ranking.empty else None,
        "n_results": int(len(all_results)),
        "output_root": str(OUTPUT_ROOT),
    }
    save_json(summary, OUTPUT_ROOT / "experiment6_summary.json")


# %% [markdown]
# ## Main Execution

# %%
def run_experiment(exp: ExperimentConfig) -> pd.DataFrame:
    model_out = OUTPUT_ROOT / exp.name
    plots_dir = ensure_plots_dir(model_out)
    print(f"\n========== {exp.name} ==========")
    model, history_df = train_model(exp, model_out)
    if model is None:
        return pd.DataFrame()

    _, val_loader, test_loader, _ = make_loaders(exp)

    val_records_raw, val_probs = generate_predictions(model, val_loader, val_df, model_out, "val")
    val_records = attach_probs(val_records_raw, val_probs)
    test_start = time.time()
    test_records_raw, test_probs = generate_predictions(model, test_loader, test_df, model_out, "test")
    inference_time = (time.time() - test_start) / max(len(test_records_raw), 1)
    test_records = attach_probs(test_records_raw, test_probs)

    grid_df, _ = validation_grid_search(val_records, model_out)
    selected_df = select_strategy_configs(grid_df, model_out)
    plot_validation_curves(grid_df, model_out)

    test_results = evaluate_selected_strategies(exp.name, test_records, selected_df, model_out)
    test_results["training_time_sec_total"] = float(history_df["training_time_sec_total"].iloc[-1]) if not history_df.empty else np.nan
    test_results["inference_time_per_image_sec"] = float(inference_time)
    test_results.to_csv(model_out / "test_results_by_strategy.csv", index=False)

    # Model ici kisa plotlar.
    save_bar(test_results, "strategy", "dice_forged_only", f"{exp.name} forged Dice", plots_dir / "bar_test_forged_dice_by_strategy.png")
    save_bar(test_results, "strategy", "component_f1_iou010", f"{exp.name} component F1", plots_dir / "bar_component_f1_by_strategy.png")
    save_bar(test_results, "strategy", "authentic_fp_rate", f"{exp.name} authentic FP", plots_dir / "bar_authentic_fp_rate_by_strategy.png")
    save_bar(test_results, "strategy", "image_f1", f"{exp.name} image F1", plots_dir / "bar_image_f1_by_strategy.png")

    # Balanced strategy gorselleri ve failure case analizi.
    balanced = selected_df[selected_df["strategy"].astype(str) == "balanced_final_score"]
    if balanced.empty:
        balanced = selected_df.head(1)
    if not balanced.empty:
        cfg = config_from_selected_row(balanced.iloc[0])
        metrics, per_df, _, _, _ = evaluate_records(test_records, cfg)
        save_case_grids(test_records, cfg, per_df, model_out)
        save_failure_cases(exp.name, test_records, cfg, per_df, model_out)
    write_model_report(model_out, exp, test_results, selected_df)

    if CFG.run_robustness:
        best_row = test_results.sort_values("final_score", ascending=False).iloc[0]
        best_cfg = config_from_selected_row(best_row)
        run_robustness(model, exp, best_cfg, model_out)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return test_results


all_results = []
errors = []
for exp in EXPERIMENTS:
    if not exp.enabled:
        print(f"[skip] {exp.name} disabled. 6C icin config.enabled=True yapabilirsiniz.")
        continue
    try:
        result = run_experiment(exp)
        if not result.empty:
            all_results.append(result)
    except Exception as exc:
        msg = f"{exp.name}: {repr(exc)}\n{traceback.format_exc()}"
        errors.append(msg)
        (OUTPUT_ROOT / "experiment_errors.log").write_text("\n\n".join(errors), encoding="utf-8")
        print("[error]", msg)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

all_results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
all_results_df.to_csv(OUTPUT_ROOT / "experiment6_all_results.csv", index=False)
all_results_df.to_csv(OUTPUT_ROOT / "test_results_all_strategies.csv", index=False)

if not all_results_df.empty:
    ranking = all_results_df.copy()
    ranking["experiment6_final_score"] = (
        0.25 * ranking["dice_forged_only"]
        + 0.25 * ranking.get("q1_dice", 0.0)
        + 0.15 * ranking.get("q2_dice", 0.0)
        + 0.15 * ranking["component_f1_iou010"]
        + 0.10 * ranking["image_f1"]
        + 0.10 * (1 - ranking["authentic_fp_rate"])
    )
    ranking = ranking.sort_values(
        ["experiment6_final_score", "q1_dice", "q2_dice", "dice_forged_only", "authentic_fp_rate"],
        ascending=[False, False, False, False, True],
    )
else:
    ranking = pd.DataFrame()
ranking.to_csv(OUTPUT_ROOT / "experiment6_final_candidate_ranking.csv", index=False)

comparison_df, comparison_md = compare_with_experiment5(all_results_df)
comparison_df.to_csv(OUTPUT_ROOT / "experiment6_vs_experiment5_comparison.csv", index=False)
(OUTPUT_ROOT / "experiment6_vs_experiment5_comparison.md").write_text(comparison_md, encoding="utf-8")

save_main_plots(all_results_df, OUTPUT_ROOT)
write_final_report(all_results_df, ranking, comparison_md)

print("\nDeney 6 tamamlandi.")
print("OUTPUT_ROOT:", OUTPUT_ROOT)
if not ranking.empty:
    print(ranking[["model_name", "strategy", "experiment6_final_score", "q1_dice", "q2_dice", "dice_forged_only", "authentic_fp_rate"]].head(10))
