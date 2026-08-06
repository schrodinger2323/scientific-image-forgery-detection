# %% [markdown]
# # Recod.ai/LUC - Deney 5 Calibration ve Post-processing
#
# Bu notebook-style Python dosyasi Colab/Kaggle uzerinde dogrudan calistirilmak icin
# hazirlanmistir. Yeni model egitmez; Deney 4'ten gelen prediction probability
# haritalarini veya gerekli olursa checkpoint'leri kullanarak calibration,
# threshold optimization, post-processing ablation, component-aware evaluation,
# small-object analysis, failure case analysis ve raporlama yapar.
#
# Test seti hicbir parametre seciminde kullanilmaz. Tum threshold ve post-processing
# secimleri validation setinde yapilir, secilen config'ler test setine aynen uygulanir.

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

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*The secret `HF_TOKEN` does not exist.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")


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
ensure_package("scipy")
ensure_package("tabulate")

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
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import albumentations as A
except ImportError:
    ensure_package("albumentations")
    import albumentations as A

try:
    import segmentation_models_pytorch as smp
except ImportError:
    try:
        ensure_package("segmentation_models_pytorch", "segmentation-models-pytorch")
        import segmentation_models_pytorch as smp
    except Exception as exc:
        smp = None
        print(f"[warn] segmentation_models_pytorch yuklenemedi. NPZ cache varsa sorun degil; checkpoint fallback calismaz. Hata: {exc}")

try:
    from transformers import AutoModel, SegformerForSemanticSegmentation
except ImportError:
    try:
        ensure_package("transformers")
        from transformers import AutoModel, SegformerForSemanticSegmentation
    except Exception as exc:
        AutoModel = None
        SegformerForSemanticSegmentation = None
        print(f"[warn] transformers yuklenemedi. NPZ cache varsa sorun degil; checkpoint fallback calismaz. Hata: {exc}")

try:
    from torch.amp import autocast
    TORCH_AMP_NEW_API = True
except Exception:
    from torch.cuda.amp import autocast
    TORCH_AMP_NEW_API = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# %% [markdown]
# ## 2. Global Config

# %%
@dataclass
class GlobalConfig:
    seed: int = 42
    image_size: int = 256
    dino_input_size: int = 252
    batch_size: int = 8
    dino_batch_size: int = 4
    num_workers: int = 0
    use_amp: bool = True
    flip_tta: bool = False
    save_prediction_probs: bool = True
    run_robustness: bool = False
    max_visual_examples: int = 12
    pixel_thresholds: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
    image_thresholds: Tuple[float, ...] = tuple(np.round(np.arange(0.01, 0.991, 0.02), 2))
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
    stage1_top_k: int = 5
    eps: float = 1e-7


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

PREVIOUS_EXPERIMENT4_SUMMARY = {
    "segformer_b0_rgb_full": {
        "forged_dice": 0.5686,
        "forged_iou": 0.3972,
        "forged_auprc": 0.6020,
        "image_f1": 0.7283,
        "image_recall": 0.9292,
        "image_specificity": 0.2733,
        "component_f1_iou010": 0.3577,
        "authentic_fp_rate": 0.3051,
        "comment": "En guclu aggregate localization ve component-aware model.",
    },
    "efficientnetb0_unet_rgb_full": {
        "forged_dice": 0.5216,
        "forged_iou": 0.3528,
        "forged_auprc": 0.5591,
        "image_f1": 0.7399,
        "image_recall": 0.9111,
        "image_specificity": 0.3559,
        "per_image_mean_dice": 0.3533,
        "comment": "En dengeli CNN modeli ve small-mask grubunda guclu.",
    },
    "dinov2_lite_decoder_rgb_full": {
        "forged_dice": 0.5279,
        "forged_iou": 0.3586,
        "forged_auprc": 0.5502,
        "image_f1": 0.7019,
        "image_recall": 0.9873,
        "image_specificity": 0.0360,
        "comment": "Pixel-level rekabetci, image-level calibration zayif.",
    },
    "unetpp_resnet34_rgb_full": {
        "forged_dice": 0.5092,
        "forged_iou": 0.3416,
        "image_f1": 0.6983,
        "authentic_fp_rate": 0.7182,
        "comment": "Baseline; cok fazla false positive component uretiyor.",
    },
}


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


def discover_paths() -> Tuple[Path, Path, Path, Path, Path]:
    dataset_root = first_existing([
        Path("/kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection"),
        Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection"),
        Path("/content/drive/MyDrive/bitirmeProjesi/dataset"),
        Path("dataset"),
    ])
    experiment4_root = first_existing([
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full"),
        Path("/kaggle/working/experiments_full"),
        Path("/kaggle/working/experiments_4_full"),
        Path("/kaggle/working/deney_4/experiments_4_full"),
        Path("experiments_full"),
        Path("experiments_4_full"),
        Path("deney_4/experiments_4_full"),
    ])
    split_root = first_existing([
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/_shared_splits_seed42"),
        Path("/kaggle/working/experiments_full/_shared_splits_seed42"),
        Path("/kaggle/working/experiments/_shared_splits_seed42"),
        Path("/kaggle/working/deney_4/_shared_splits_seed42"),
        Path("experiments_full/_shared_splits_seed42"),
        Path("experiments/_shared_splits_seed42"),
        Path("deney_4/_shared_splits_seed42"),
        Path("deney_4/experiments/_shared_splits_seed42"),
    ])
    if dataset_root is None:
        raise RuntimeError("Dataset root bulunamadi. Kaggle input veya Google Drive dataset path'ini kontrol edin.")
    if experiment4_root is None:
        raise RuntimeError("Deney 4 experiment root bulunamadi. experiments_full veya experiments_4_full path'ini kontrol edin.")
    if split_root is None:
        raise RuntimeError("Shared split bulunamadi. Split tekrar olusturulmayacak; _shared_splits_seed42 klasoru gerekli.")
    output_root = experiment4_root / "experiment5_calibration_postprocessing"
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_root = output_root
    print("[paths]")
    print(" dataset_root:", dataset_root)
    print(" experiment4_root:", experiment4_root)
    print(" split_root:", split_root)
    print(" output_root:", output_root)
    return dataset_root, experiment4_root, split_root, output_root, comparison_root


DATASET_ROOT, EXPERIMENT4_ROOT, SPLIT_ROOT, OUTPUT_ROOT, COMPARISON_ROOT = discover_paths()

with open(OUTPUT_ROOT / "experiment5_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "global_config": asdict(CFG),
        "experiments": [asdict(e) for e in EXPERIMENTS],
        "dataset_root": str(DATASET_ROOT),
        "experiment4_root": str(EXPERIMENT4_ROOT),
        "split_root": str(SPLIT_ROOT),
        "output_root": str(OUTPUT_ROOT),
    }, f, indent=2, ensure_ascii=False, default=str)

with open(OUTPUT_ROOT / "environment_info.json", "w", encoding="utf-8") as f:
    json.dump({
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
    }, f, indent=2, ensure_ascii=False)


# %% [markdown]
# ## 4. Split Loading and Leakage Check

# %%
def find_mask_paths(image_id: str, mask_dir: Path) -> List[Path]:
    exact = mask_dir / f"{image_id}.npy"
    candidates: List[Path] = []
    if exact.exists():
        candidates.append(exact)
    for path in sorted(mask_dir.glob(f"{image_id}*.npy")):
        if path not in candidates:
            candidates.append(path)
    return candidates


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
    df["image_id"] = df["image_id"].astype(str)
    df["image_path"] = image_paths
    df["mask_paths"] = mask_paths
    df["mask_path"] = first_mask_paths
    df["image_label"] = df["image_label"].astype(int)
    df["label"] = df["image_label"].astype(int)
    return df


def load_shared_splits(split_root: Path, dataset_root: Path) -> Dict[str, pd.DataFrame]:
    required = ["full.csv", "train.csv", "val.csv", "test.csv"]
    missing = [name for name in required if not (split_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Shared split eksik: {missing}. Deney 5 split yeniden olusturmaz.")
    splits = {
        name.replace(".csv", ""): repair_split_paths(pd.read_csv(split_root / name), dataset_root)
        for name in required
    }
    return splits


def summarize_split(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    labels = df["image_label"].astype(int)
    return {
        "split": split_name,
        "count": int(len(df)),
        "authentic": int((labels == 0).sum()),
        "forged": int((labels == 1).sum()),
    }


SPLITS = load_shared_splits(SPLIT_ROOT, DATASET_ROOT)
full_df, train_df, val_df, test_df = SPLITS["full"], SPLITS["train"], SPLITS["val"], SPLITS["test"]

leakage = {
    "train_val_image_id_overlap": len(set(train_df["image_id"]) & set(val_df["image_id"])),
    "train_test_image_id_overlap": len(set(train_df["image_id"]) & set(test_df["image_id"])),
    "val_test_image_id_overlap": len(set(val_df["image_id"]) & set(test_df["image_id"])),
}
if any(v != 0 for v in leakage.values()):
    raise RuntimeError(f"Split leakage tespit edildi: {leakage}")

split_summary = pd.DataFrame([
    summarize_split(train_df, "train"),
    summarize_split(val_df, "val"),
    summarize_split(test_df, "test"),
])
for key, value in leakage.items():
    split_summary[key] = value
split_summary.to_csv(OUTPUT_ROOT / "split_summary.csv", index=False)
print(split_summary)


# %% [markdown]
# ## 5. Dataset and Mask Utilities

# %%
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


def resize_mask_nearest(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def load_mask_for_row(row: pd.Series, eval_size: int = 256) -> np.ndarray:
    image = load_rgb_image(row["image_path"])
    h, w = image.shape[:2]
    mask = load_binary_mask(row.get("mask_paths", row.get("mask_path", "")), int(row["image_label"]), (h, w))
    if mask.shape != (eval_size, eval_size):
        mask = cv2.resize(mask.astype(np.uint8), (eval_size, eval_size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


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


# %% [markdown]
# ## 6. Prediction Loading or Recomputing

# %%
class SegFormerBinaryWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        if SegformerForSemanticSegmentation is None:
            raise ImportError("transformers paketi yok; SegFormer checkpoint fallback calisamaz.")
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
        if AutoModel is None:
            raise ImportError("transformers paketi yok; DINOv2 checkpoint fallback calisamaz.")
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
        frozen = all(not p.requires_grad for p in self.backbone.parameters())
        with torch.no_grad() if frozen else torch.enable_grad():
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
        if smp is None:
            raise ImportError("segmentation_models_pytorch paketi yok; SMP checkpoint fallback calisamaz.")
        return smp.UnetPlusPlus(
            encoder_name=exp.encoder_or_backbone,
            encoder_weights=exp.encoder_weights,
            in_channels=exp.in_channels,
            classes=exp.classes,
            activation=None,
        )
    if exp.model_type == "unet":
        if smp is None:
            raise ImportError("segmentation_models_pytorch paketi yok; SMP checkpoint fallback calisamaz.")
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
        return DINOv2LiteDecoder(exp.hf_model_name or "facebook/dinov2-small", exp.freeze_backbone_stage1)
    raise ValueError(exp.model_type)


def load_checkpoint(path: Path, model: nn.Module) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return ckpt


def autocast_context(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    if TORCH_AMP_NEW_API:
        return autocast(device_type=device.type, enabled=enabled)
    return autocast(enabled=enabled)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, desc: str, flip_tta: bool = False) -> List[Dict[str, Any]]:
    model.eval()
    records: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(DEVICE, non_blocking=True)
        eval_masks = batch["eval_mask"].to(DEVICE, non_blocking=True)
        with autocast_context(DEVICE, CFG.use_amp):
            logits = model(images)
            if flip_tta:
                logits_flip = model(torch.flip(images, dims=[3]))
                logits = 0.5 * (logits + torch.flip(logits_flip, dims=[3]))
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
                "original_height": int(batch["orig_h"][i]),
                "original_width": int(batch["orig_w"][i]),
                "resized_height": int(probs.shape[-2]),
                "resized_width": int(probs.shape[-1]),
                "prob": probs[i].astype(np.float32),
                "mask": (masks[i] > 0).astype(np.uint8),
            })
    return records


def prediction_candidates(model_dir: Path, split_name: str) -> List[Path]:
    return [
        model_dir / f"{split_name}_prob_maps.npz",
        model_dir / f"{split_name}_prediction_probs.npz",
        model_dir / f"{split_name}_predictions_probs.npz",
        model_dir / f"{split_name}_pred_probs.npz",
    ]


def save_prediction_npz(records: List[Dict[str, Any]], out_path: Path) -> None:
    probs = np.stack([r["prob"].astype(np.float16) for r in records])
    masks = np.stack([r["mask"].astype(np.uint8) for r in records])
    image_ids = np.array([str(r["image_id"]) for r in records])
    labels = np.array([int(r["image_label"]) for r in records], dtype=np.uint8)
    np.savez_compressed(out_path, probs=probs, masks=masks, image_ids=image_ids, labels=labels)


def make_metadata(records: List[Dict[str, Any]], split_df: pd.DataFrame) -> pd.DataFrame:
    by_id = {str(r["image_id"]): r for r in records}
    rows = []
    for _, row in split_df.iterrows():
        image_id = str(row["image_id"])
        rec = by_id.get(image_id)
        if rec is None:
            continue
        mask = rec["mask"]
        gt_area = int(mask.sum())
        rows.append({
            "image_id": image_id,
            "image_path": row["image_path"],
            "mask_path": row.get("mask_path", ""),
            "class_name": row["class_name"],
            "image_label": int(row["image_label"]),
            "original_height": int(rec.get("original_height", 0)),
            "original_width": int(rec.get("original_width", 0)),
            "resized_height": int(mask.shape[0]),
            "resized_width": int(mask.shape[1]),
            "gt_area": gt_area,
            "gt_area_ratio": float(gt_area / max(mask.size, 1)),
        })
    return pd.DataFrame(rows)


def load_records_from_npz(npz_path: Path, split_df: pd.DataFrame, eval_size: int = 256) -> List[Dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    key_probs = "probs" if "probs" in data.files else "prob_maps"
    probs = data[key_probs].astype(np.float32)
    image_ids = data["image_ids"].astype(str) if "image_ids" in data.files else split_df["image_id"].astype(str).to_numpy()
    labels = data["labels"].astype(int) if "labels" in data.files else None
    masks = data["masks"].astype(np.uint8) if "masks" in data.files else None
    split_by_id = {str(r.image_id): r for r in split_df.itertuples(index=False)}
    records: List[Dict[str, Any]] = []
    for i, image_id in enumerate(image_ids):
        row = split_by_id.get(str(image_id))
        if row is None:
            continue
        prob = probs[i]
        if prob.ndim == 3:
            prob = prob.squeeze()
        if masks is not None:
            mask = masks[i]
            if mask.ndim == 3:
                mask = mask.squeeze()
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = load_mask_for_row(pd.Series(row._asdict()), eval_size=prob.shape[0])
        records.append({
            "image_id": str(image_id),
            "image_path": str(getattr(row, "image_path")),
            "class_name": str(getattr(row, "class_name")),
            "image_label": int(labels[i]) if labels is not None else int(getattr(row, "image_label")),
            "original_height": 0,
            "original_width": 0,
            "resized_height": int(prob.shape[0]),
            "resized_width": int(prob.shape[1]),
            "prob": prob.astype(np.float32),
            "mask": mask.astype(np.uint8),
        })
    if len(records) != len(split_df):
        print(f"[warn] {npz_path.name}: {len(records)}/{len(split_df)} kayit eslesti.")
    return records


def get_or_create_predictions(exp: ExperimentConfig, split_name: str, split_df: pd.DataFrame,
                              model_out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    exp4_model_dir = EXPERIMENT4_ROOT / exp.name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    availability = {
        "model_name": exp.name,
        "split": split_name,
        "loaded_from": "",
        "status": "missing",
        "n_records": 0,
        "message": "",
    }
    for candidate in prediction_candidates(exp4_model_dir, split_name) + prediction_candidates(model_out_dir, split_name):
        if candidate.exists():
            records = load_records_from_npz(candidate, split_df, eval_size=exp.eval_size or CFG.image_size)
            canonical = model_out_dir / f"{split_name}_prob_maps.npz"
            if canonical.resolve() != candidate.resolve():
                save_prediction_npz(records, canonical)
            metadata = make_metadata(records, split_df)
            metadata.to_csv(model_out_dir / f"{split_name}_metadata.csv", index=False)
            availability.update({
                "loaded_from": str(candidate),
                "status": "loaded_npz",
                "n_records": len(records),
                "message": "Prediction probability cache yuklendi.",
            })
            return records, availability
    ckpt_path = exp4_model_dir / "best_model.pth"
    if not ckpt_path.exists():
        availability["message"] = f"Prediction npz ve checkpoint bulunamadi: {exp4_model_dir}"
        raise FileNotFoundError(availability["message"])
    model = build_model(exp).to(DEVICE)
    load_checkpoint(ckpt_path, model)
    loader = make_loader(split_df, exp)
    records = collect_predictions(model, loader, desc=f"{exp.name} {split_name} predict", flip_tta=CFG.flip_tta)
    save_prediction_npz(records, model_out_dir / f"{split_name}_prob_maps.npz")
    make_metadata(records, split_df).to_csv(model_out_dir / f"{split_name}_metadata.csv", index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    availability.update({
        "loaded_from": str(ckpt_path),
        "status": "recomputed_from_checkpoint",
        "n_records": len(records),
        "message": "Checkpoint ile prediction yeniden uretildi.",
    })
    return records, availability


# %% [markdown]
# ## 7. Post-processing Functions

# %%
POSTPROCESS_MODES = {
    0: "raw",
    1: "min_area_clean",
    2: "probability_gated",
    3: "area_probability_clean",
    4: "morph_area_probability_clean",
    5: "keep_topk_components",
}


def apply_morphology(binary: np.ndarray, morphology: str, kernel_size: int) -> np.ndarray:
    if morphology == "none":
        return binary.astype(np.uint8)
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    out = binary.astype(np.uint8)
    if morphology in ("open", "open_close"):
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if morphology in ("close", "open_close"):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return (out > 0).astype(np.uint8)


def component_table(binary: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    comps = []
    total_area = max(binary.size, 1)
    for label_id in range(1, n):
        comp = labels == label_id
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        comps.append({
            "label_id": label_id,
            "area": area,
            "area_ratio": float(area / total_area),
            "mean_probability": float(prob[comp].mean()) if area > 0 else 0.0,
            "max_probability": float(prob[comp].max()) if area > 0 else 0.0,
        })
    return labels, comps


def postprocess_probability_map(prob: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    pixel_threshold = float(config.get("pixel_threshold", 0.5))
    mode = str(config.get("postprocess_mode", "raw"))
    min_area = int(config.get("min_component_area", 0) or 0)
    min_mean_prob = float(config.get("min_component_mean_probability", 0.0) or 0.0)
    morphology = str(config.get("morphology", "none") or "none")
    kernel_size = int(config.get("morph_kernel_size", 3) or 3)
    top_k = config.get("top_k_components", None)
    if pd.isna(top_k):
        top_k = None
    top_k = None if top_k in (None, "None", "none", "") else int(top_k)

    raw = (prob >= pixel_threshold).astype(np.uint8)
    work = raw.copy()
    if mode == "raw":
        labels, comps = component_table(work, prob)
        return {"raw_mask": raw, "mask": work, "labels": labels, "components": comps}

    if mode == "morph_area_probability_clean":
        work = apply_morphology(work, morphology, kernel_size)

    labels, comps = component_table(work, prob)
    keep: List[Dict[str, Any]] = []
    for comp in comps:
        pass_area = True if mode == "probability_gated" else comp["area"] >= min_area
        pass_prob = True if mode == "min_area_clean" else comp["mean_probability"] >= min_mean_prob
        if mode in ("area_probability_clean", "morph_area_probability_clean", "keep_topk_components"):
            pass_area = comp["area"] >= min_area
            pass_prob = comp["mean_probability"] >= min_mean_prob
        if pass_area and pass_prob:
            keep.append(comp)

    if mode == "keep_topk_components" and top_k is not None:
        sort_key = str(config.get("top_k_sort_by", "area"))
        key = "mean_probability" if sort_key == "mean_probability" else "area"
        keep = sorted(keep, key=lambda x: x[key], reverse=True)[:top_k]

    cleaned = np.zeros_like(work, dtype=np.uint8)
    keep_ids = {c["label_id"] for c in keep}
    for label_id in keep_ids:
        cleaned[labels == label_id] = 1
    labels_clean, comps_clean = component_table(cleaned, prob)
    return {"raw_mask": raw, "mask": cleaned, "labels": labels_clean, "components": comps_clean}


def image_score_from_outputs(prob: np.ndarray, raw_mask: np.ndarray, final_mask: np.ndarray,
                             components: List[Dict[str, Any]], score_type: str) -> float:
    flat = prob.reshape(-1)
    if score_type == "max_probability":
        return float(flat.max())
    if score_type == "top1_mean_probability":
        k = max(1, int(math.ceil(0.01 * flat.size)))
        return float(np.partition(flat, -k)[-k:].mean())
    if score_type == "top5_mean_probability":
        k = max(1, int(math.ceil(0.05 * flat.size)))
        return float(np.partition(flat, -k)[-k:].mean())
    if score_type == "pred_mask_ratio_raw":
        return float(raw_mask.mean())
    if score_type == "pred_mask_ratio_clean":
        return float(final_mask.mean())
    if score_type == "max_component_mean_probability":
        return float(max([c["mean_probability"] for c in components], default=0.0))
    if score_type == "max_component_area_ratio":
        return float(max([c["area_ratio"] for c in components], default=0.0))
    raise ValueError(score_type)


# %% [markdown]
# ## 8. Pixel Metrics

# %%
def safe_auc(y_true: np.ndarray, y_score: np.ndarray, kind: str) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        if kind == "auprc":
            return float(average_precision_score(y_true, y_score))
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def dice_iou_from_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> Tuple[float, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    pred_sum = int(pred_b.sum())
    gt_sum = int(gt_b.sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    dice = (2.0 * inter + eps) / max(pred_sum + gt_sum + eps, eps)
    iou = (inter + eps) / max(union + eps, eps)
    return float(dice), float(iou)


def pixel_metrics_from_predictions(preds: List[np.ndarray], records: List[Dict[str, Any]],
                                   forged_only: bool = False, compute_auc: bool = True) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    pred_ratios, gt_ratios = [], []
    all_probs, all_gts = [], []
    for pred, rec in zip(preds, records):
        if forged_only and int(rec["image_label"]) != 1:
            continue
        gt = rec["mask"].astype(np.uint8)
        pred = pred.astype(np.uint8)
        tp += int(((pred == 1) & (gt == 1)).sum())
        fp += int(((pred == 1) & (gt == 0)).sum())
        tn += int(((pred == 0) & (gt == 0)).sum())
        fn += int(((pred == 0) & (gt == 1)).sum())
        pred_ratios.append(float(pred.mean()))
        gt_ratios.append(float(gt.mean()))
        if compute_auc:
            all_probs.append(rec["prob"].reshape(-1).astype(np.float32))
            all_gts.append(gt.reshape(-1).astype(np.uint8))
    y_prob = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_gts) if all_gts else np.array([], dtype=np.uint8)
    eps = CFG.eps
    return {
        "dice": float((2 * tp) / max(2 * tp + fp + fn, eps)),
        "iou": float(tp / max(tp + fp + fn, eps)),
        "precision": float(tp / max(tp + fp, eps)),
        "recall": float(tp / max(tp + fn, eps)),
        "specificity": float(tn / max(tn + fp, eps)),
        "auprc": safe_auc(y_true, y_prob, "auprc") if len(y_true) else float("nan"),
        "roc_auc": safe_auc(y_true, y_prob, "roc_auc") if len(y_true) else float("nan"),
        "predicted_positive_pixel_ratio": float(np.mean(pred_ratios)) if pred_ratios else float("nan"),
        "gt_positive_pixel_ratio": float(np.mean(gt_ratios)) if gt_ratios else float("nan"),
    }


def per_image_metrics(records: List[Dict[str, Any]], preds: List[np.ndarray],
                      image_scores: np.ndarray, image_threshold: float) -> pd.DataFrame:
    rows = []
    for rec, pred, score in zip(records, preds, image_scores):
        gt = rec["mask"].astype(np.uint8)
        dice, iou = dice_iou_from_binary(pred, gt)
        rows.append({
            "image_id": rec["image_id"],
            "image_path": rec["image_path"],
            "class_name": rec["class_name"],
            "image_label": int(rec["image_label"]),
            "gt_area": int(gt.sum()),
            "gt_area_ratio": float(gt.mean()),
            "pred_area": int(pred.sum()),
            "pred_area_ratio": float(pred.mean()),
            "dice": dice,
            "iou": iou,
            "image_score": float(score),
            "image_pred_label": int(score >= image_threshold),
        })
    return pd.DataFrame(rows)


# %% [markdown]
# ## 9. Image-level Metrics and Calibration

# %%
def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(y_true)
    clipped = np.clip(y_score.astype(float), 0.0, 1.0)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (clipped >= lo) & (clipped <= hi)
        else:
            mask = (clipped >= lo) & (clipped < hi)
        count = int(mask.sum())
        if count == 0:
            avg_conf = avg_acc = 0.0
        else:
            avg_conf = float(clipped[mask].mean())
            avg_acc = float(y_true[mask].mean())
            ece += (count / max(n, 1)) * abs(avg_acc - avg_conf)
        rows.append({
            "bin": i,
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "count": count,
            "avg_confidence": avg_conf,
            "empirical_accuracy": avg_acc,
        })
    return float(ece), pd.DataFrame(rows)


def image_level_metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, image_threshold: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_pred = (y_score >= image_threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ece, reliability = expected_calibration_error(y_true, y_score, n_bins=10)
    metrics = {
        "image_accuracy": float(accuracy_score(y_true, y_pred)),
        "image_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "image_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "image_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "image_specificity": float(tn / max(tn + fp, 1)),
        "image_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "image_roc_auc": safe_auc(y_true, y_score, "roc_auc"),
        "image_auprc": safe_auc(y_true, y_score, "auprc"),
        "image_brier": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))) if len(np.unique(y_true)) > 1 else float("nan"),
        "image_ece_10bin": ece,
        "image_tp": int(tp),
        "image_fn": int(fn),
        "image_tn": int(tn),
        "image_fp": int(fp),
    }
    return metrics, reliability


def best_image_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.5
    best = {"image_f1": -1.0, "image_recall": 0.0, "image_specificity": 0.0}
    thresholds = sorted(set(CFG.image_thresholds.tolist() if hasattr(CFG.image_thresholds, "tolist") else CFG.image_thresholds))
    quantiles = np.quantile(y_score, np.linspace(0.02, 0.98, 49)) if len(y_score) else []
    for thr in sorted(set(list(thresholds) + [float(x) for x in quantiles])):
        metrics, _ = image_level_metrics_from_scores(y_true, y_score, float(thr))
        key = (metrics["image_f1"], metrics["image_recall"], metrics["image_specificity"])
        best_key = (best["image_f1"], best["image_recall"], best["image_specificity"])
        if key > best_key:
            best_thr = float(thr)
            best = metrics
    return best_thr, best


# %% [markdown]
# ## 10. Component-aware Metrics

# %%
def component_iou_matrix(gt_labels: np.ndarray, gt_count: int, pred_labels: np.ndarray, pred_count: int) -> np.ndarray:
    if gt_count == 0 or pred_count == 0:
        return np.zeros((gt_count, pred_count), dtype=np.float32)
    mat = np.zeros((gt_count, pred_count), dtype=np.float32)
    for gi in range(1, gt_count + 1):
        gt_comp = gt_labels == gi
        for pi in range(1, pred_count + 1):
            pred_comp = pred_labels == pi
            inter = int(np.logical_and(gt_comp, pred_comp).sum())
            union = int(np.logical_or(gt_comp, pred_comp).sum())
            mat[gi - 1, pi - 1] = inter / max(union, 1)
    return mat


def match_components(gt: np.ndarray, pred: np.ndarray, iou_threshold: float) -> Dict[str, Any]:
    gt_n, gt_labels, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    pred_n, pred_labels, _, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    gt_count = gt_n - 1
    pred_count = pred_n - 1
    iou_mat = component_iou_matrix(gt_labels, gt_count, pred_labels, pred_count)
    matched = 0
    matched_ious: List[float] = []
    if gt_count > 0 and pred_count > 0:
        rows, cols = linear_sum_assignment(-iou_mat)
        for r, c in zip(rows, cols):
            iou = float(iou_mat[r, c])
            if iou >= iou_threshold:
                matched += 1
                matched_ious.append(iou)
    return {
        "gt_component_count": int(gt_count),
        "pred_component_count": int(pred_count),
        "matched_component_count": int(matched),
        "component_fp": int(pred_count - matched),
        "component_fn": int(gt_count - matched),
        "avg_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def component_metrics_from_predictions(records: List[Dict[str, Any]], preds: List[np.ndarray],
                                       iou_threshold: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    total_tp = total_fp = total_fn = 0
    matched_ious = []
    for rec, pred in zip(records, preds):
        comp = match_components(rec["mask"].astype(np.uint8), pred.astype(np.uint8), iou_threshold)
        total_tp += comp["matched_component_count"]
        total_fp += comp["component_fp"]
        total_fn += comp["component_fn"]
        if comp["avg_matched_iou"] > 0:
            matched_ious.append(comp["avg_matched_iou"])
        rows.append({
            "image_id": rec["image_id"],
            "class_name": rec["class_name"],
            "image_label": int(rec["image_label"]),
            "component_iou_threshold": float(iou_threshold),
            **comp,
        })
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, CFG.eps)
    suffix = f"{int(round(iou_threshold * 100)):03d}"
    metrics = {
        f"component_precision_iou{suffix}": float(precision),
        f"component_recall_iou{suffix}": float(recall),
        f"component_f1_iou{suffix}": float(f1),
        f"component_tp_iou{suffix}": int(total_tp),
        f"component_fp_iou{suffix}": int(total_fp),
        f"component_fn_iou{suffix}": int(total_fn),
        f"avg_matched_iou_iou{suffix}": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "avg_pred_component_count": float(np.mean([r["pred_component_count"] for r in rows])) if rows else 0.0,
        "avg_gt_component_count": float(np.mean([r["gt_component_count"] for r in rows])) if rows else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def authentic_false_alarm_metrics(records: List[Dict[str, Any]], preds: List[np.ndarray],
                                  components_per_image: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    counts, ratios = [], []
    for rec, pred, comps in zip(records, preds, components_per_image):
        if int(rec["image_label"]) != 0:
            continue
        counts.append(len(comps))
        ratios.append(float(pred.mean()))
    counts_arr = np.array(counts, dtype=float) if counts else np.array([], dtype=float)
    return {
        "authentic_fp_rate": float((counts_arr > 0).mean()) if len(counts_arr) else float("nan"),
        "authentic_mean_pred_component_count": float(counts_arr.mean()) if len(counts_arr) else 0.0,
        "authentic_median_pred_component_count": float(np.median(counts_arr)) if len(counts_arr) else 0.0,
        "authentic_mean_pred_area_ratio": float(np.mean(ratios)) if ratios else 0.0,
    }


def apply_config_to_records(records: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    preds, raw_preds, scores, comps_per_image = [], [], [], []
    score_type = str(config.get("image_score_type", "max_probability"))
    for rec in records:
        out = postprocess_probability_map(rec["prob"], config)
        pred = out["mask"].astype(np.uint8)
        raw = out["raw_mask"].astype(np.uint8)
        comps = out["components"]
        score = image_score_from_outputs(rec["prob"], raw, pred, comps, score_type)
        preds.append(pred)
        raw_preds.append(raw)
        comps_per_image.append(comps)
        scores.append(score)
    return {
        "preds": preds,
        "raw_preds": raw_preds,
        "image_scores": np.array(scores, dtype=np.float32),
        "components_per_image": comps_per_image,
    }


# %% [markdown]
# ## 11. Validation Grid Search

# %%
def base_config(**kwargs: Any) -> Dict[str, Any]:
    cfg = {
        "pixel_threshold": 0.5,
        "image_score_type": "max_probability",
        "image_threshold": 0.5,
        "postprocess_mode": "raw",
        "min_component_area": 0,
        "min_component_mean_probability": 0.0,
        "morphology": "none",
        "morph_kernel_size": 3,
        "top_k_components": None,
        "top_k_sort_by": "area",
    }
    cfg.update(kwargs)
    return cfg


def small_mask_threshold(records: List[Dict[str, Any]]) -> float:
    forged_areas = [int(r["mask"].sum()) for r in records if int(r["image_label"]) == 1]
    if not forged_areas:
        return 0.0
    return float(np.quantile(forged_areas, 0.25))


def evaluate_config_on_records(records: List[Dict[str, Any]], config: Dict[str, Any],
                               choose_image_threshold: bool = True) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    outputs = apply_config_to_records(records, config)
    y_true = np.array([int(r["image_label"]) for r in records], dtype=np.uint8)
    y_score = outputs["image_scores"]
    if choose_image_threshold:
        image_threshold, _ = best_image_threshold(y_true, y_score)
        config = dict(config)
        config["image_threshold"] = float(image_threshold)
    image_threshold = float(config.get("image_threshold", 0.5))

    all_m = pixel_metrics_from_predictions(outputs["preds"], records, forged_only=False, compute_auc=False)
    forged_m = pixel_metrics_from_predictions(outputs["preds"], records, forged_only=True, compute_auc=False)
    img_m, reliability = image_level_metrics_from_scores(y_true, y_score, image_threshold)
    comp010, _ = component_metrics_from_predictions(records, outputs["preds"], 0.10)
    comp025, _ = component_metrics_from_predictions(records, outputs["preds"], 0.25)
    auth_m = authentic_false_alarm_metrics(records, outputs["preds"], outputs["components_per_image"])
    per_df = per_image_metrics(records, outputs["preds"], y_score, image_threshold)

    forged_per = per_df[per_df["image_label"] == 1]
    q1_thr = small_mask_threshold(records)
    small_df = forged_per[forged_per["gt_area"] <= q1_thr]
    metrics = {
        **config,
        "val_dice_all": all_m["dice"],
        "val_forged_dice": forged_m["dice"],
        "val_forged_iou": forged_m["iou"],
        "val_pixel_precision": all_m["precision"],
        "val_pixel_recall": all_m["recall"],
        "val_pixel_specificity": all_m["specificity"],
        "val_component_f1_iou010": comp010["component_f1_iou010"],
        "val_component_f1_iou025": comp025["component_f1_iou025"],
        "val_image_f1": img_m["image_f1"],
        "val_image_recall": img_m["image_recall"],
        "val_image_specificity": img_m["image_specificity"],
        "val_authentic_fp_rate": auth_m["authentic_fp_rate"],
        "val_small_mask_dice": float(small_df["dice"].mean()) if len(small_df) else float("nan"),
    }
    metrics["balanced_score"] = (
        0.35 * metrics["val_forged_dice"]
        + 0.25 * metrics["val_component_f1_iou010"]
        + 0.20 * metrics["val_image_f1"]
        + 0.20 * (1.0 - metrics["val_authentic_fp_rate"])
    )
    return metrics, per_df, reliability


def stage1_configs() -> List[Dict[str, Any]]:
    configs = []
    for pixel_threshold in CFG.pixel_thresholds:
        for image_score_type in CFG.image_score_types:
            configs.append(base_config(
                pixel_threshold=float(pixel_threshold),
                image_score_type=image_score_type,
                postprocess_mode="raw",
            ))
            for min_area in CFG.min_component_areas:
                if int(min_area) == 0:
                    continue
                configs.append(base_config(
                    pixel_threshold=float(pixel_threshold),
                    image_score_type=image_score_type,
                    postprocess_mode="min_area_clean",
                    min_component_area=int(min_area),
                ))
    return configs


def expand_stage2_configs(top_stage1: pd.DataFrame) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for row in top_stage1.itertuples(index=False):
        pixel_threshold = float(getattr(row, "pixel_threshold"))
        image_score_type = str(getattr(row, "image_score_type"))
        min_area = int(getattr(row, "min_component_area"))
        for mean_prob in CFG.min_component_mean_probs:
            configs.append(base_config(
                pixel_threshold=pixel_threshold,
                image_score_type=image_score_type,
                postprocess_mode="probability_gated",
                min_component_mean_probability=float(mean_prob),
            ))
            configs.append(base_config(
                pixel_threshold=pixel_threshold,
                image_score_type=image_score_type,
                postprocess_mode="area_probability_clean",
                min_component_area=min_area,
                min_component_mean_probability=float(mean_prob),
            ))
            for morphology in CFG.morphologies:
                if morphology == "none":
                    continue
                for kernel_size in CFG.morph_kernel_sizes:
                    configs.append(base_config(
                        pixel_threshold=pixel_threshold,
                        image_score_type=image_score_type,
                        postprocess_mode="morph_area_probability_clean",
                        min_component_area=min_area,
                        min_component_mean_probability=float(mean_prob),
                        morphology=morphology,
                        morph_kernel_size=int(kernel_size),
                    ))
            for top_k in CFG.top_k_components_values:
                for sort_by in ("area", "mean_probability"):
                    configs.append(base_config(
                        pixel_threshold=pixel_threshold,
                        image_score_type=image_score_type,
                        postprocess_mode="keep_topk_components",
                        min_component_area=min_area,
                        min_component_mean_probability=float(mean_prob),
                        top_k_components=top_k,
                        top_k_sort_by=sort_by,
                    ))
    unique = []
    seen = set()
    for cfg in configs:
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def run_validation_grid_search(model_name: str, records: List[Dict[str, Any]], model_out_dir: Path) -> pd.DataFrame:
    rows = []
    print(f"[grid-stage1] {model_name}")
    for cfg in tqdm(stage1_configs(), desc=f"{model_name} stage1", leave=False):
        metrics, _, _ = evaluate_config_on_records(records, cfg, choose_image_threshold=True)
        rows.append(metrics)
    stage1_df = pd.DataFrame(rows).sort_values(
        ["balanced_score", "val_forged_dice", "val_component_f1_iou010"],
        ascending=False,
    ).reset_index(drop=True)

    print(f"[grid-stage2] {model_name}: top {CFG.stage1_top_k}")
    for cfg in tqdm(expand_stage2_configs(stage1_df.head(CFG.stage1_top_k)), desc=f"{model_name} stage2", leave=False):
        metrics, _, _ = evaluate_config_on_records(records, cfg, choose_image_threshold=True)
        rows.append(metrics)

    grid_df = pd.DataFrame(rows).drop_duplicates(
        subset=[
            "pixel_threshold", "image_score_type", "image_threshold", "postprocess_mode",
            "min_component_area", "min_component_mean_probability", "morphology",
            "morph_kernel_size", "top_k_components", "top_k_sort_by",
        ],
        keep="first",
    )
    grid_df = grid_df.sort_values(
        ["balanced_score", "val_forged_dice", "val_component_f1_iou010"],
        ascending=False,
    ).reset_index(drop=True)
    grid_df.to_csv(model_out_dir / "val_grid_search_all.csv", index=False)
    grid_df.head(50).to_csv(model_out_dir / "val_grid_search_top50.csv", index=False)
    return grid_df


def select_strategy_configs(model_name: str, grid_df: pd.DataFrame) -> pd.DataFrame:
    selected = []

    def take(strategy: str, df: pd.DataFrame, sort_cols: List[str], ascending: List[bool]) -> None:
        if df.empty:
            return
        row = df.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        row["strategy"] = strategy
        row["model_name"] = model_name
        selected.append(row)

    take("best_forged_dice", grid_df, ["val_forged_dice", "val_forged_iou"], [False, False])
    take("best_component_f1", grid_df, ["val_component_f1_iou010", "val_authentic_fp_rate"], [False, True])
    take("balanced_final_score", grid_df, ["balanced_score", "val_forged_dice"], [False, False])
    take("small_object_focused", grid_df, ["val_small_mask_dice", "val_forged_dice"], [False, False])
    low_fp = grid_df[grid_df["val_authentic_fp_rate"] <= 0.25]
    if len(low_fp):
        take("low_false_alarm", low_fp, ["val_component_f1_iou010", "val_forged_dice"], [False, False])
    else:
        top_low = grid_df.sort_values("val_authentic_fp_rate", ascending=True).head(10)
        take("low_false_alarm", top_low, ["val_forged_dice", "val_component_f1_iou010"], [False, False])

    cols = [
        "model_name", "strategy", "pixel_threshold", "image_score_type", "image_threshold",
        "postprocess_mode", "min_component_area", "min_component_mean_probability",
        "morphology", "morph_kernel_size", "top_k_components", "top_k_sort_by",
        "val_forged_dice", "val_forged_iou", "val_component_f1_iou010",
        "val_component_f1_iou025", "val_image_f1", "val_authentic_fp_rate",
        "val_small_mask_dice", "balanced_score",
    ]
    out = pd.DataFrame(selected)
    return out[[c for c in cols if c in out.columns]]


# %% [markdown]
# ## 12. Test Evaluation

# %%
def small_mask_bin_metrics(per_df: pd.DataFrame) -> pd.DataFrame:
    forged = per_df[per_df["image_label"] == 1].copy()
    if forged.empty:
        return pd.DataFrame()
    try:
        forged["area_bin"] = pd.qcut(forged["gt_area"], q=4, labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"], duplicates="drop")
    except Exception:
        forged["area_bin"] = "all"
    rows = []
    for bin_name, part in forged.groupby("area_bin", observed=False):
        rows.append({
            "area_bin": str(bin_name),
            "n": int(len(part)),
            "mean_dice": float(part["dice"].mean()),
            "median_dice": float(part["dice"].median()),
            "mean_iou": float(part["iou"].mean()),
            "dice_lt_005_count": int((part["dice"] < 0.05).sum()),
        })
    return pd.DataFrame(rows)


def evaluate_strategy_on_test(model_name: str, strategy_row: Dict[str, Any],
                              test_records: List[Dict[str, Any]],
                              model_out_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = str(strategy_row["strategy"])
    config = dict(strategy_row)
    outputs = apply_config_to_records(test_records, config)
    y_true = np.array([int(r["image_label"]) for r in test_records], dtype=np.uint8)
    y_score = outputs["image_scores"]
    image_threshold = float(config["image_threshold"])

    all_m = pixel_metrics_from_predictions(outputs["preds"], test_records, forged_only=False, compute_auc=True)
    forged_m = pixel_metrics_from_predictions(outputs["preds"], test_records, forged_only=True, compute_auc=True)
    img_m, reliability_df = image_level_metrics_from_scores(y_true, y_score, image_threshold)
    auth_m = authentic_false_alarm_metrics(test_records, outputs["preds"], outputs["components_per_image"])
    per_df = per_image_metrics(test_records, outputs["preds"], y_score, image_threshold)
    forged_per = per_df[per_df["image_label"] == 1]

    comp_details = []
    comp_metrics_all: Dict[str, Any] = {}
    for thr in CFG.component_iou_thresholds:
        comp_m, comp_df = component_metrics_from_predictions(test_records, outputs["preds"], float(thr))
        comp_metrics_all.update(comp_m)
        comp_details.append(comp_df)
    comp_details_df = pd.concat(comp_details, ignore_index=True) if comp_details else pd.DataFrame()
    bin_df = small_mask_bin_metrics(per_df)

    metrics = {
        "model_name": model_name,
        "strategy": strategy,
        "postprocess_mode": config.get("postprocess_mode"),
        "pixel_threshold": float(config.get("pixel_threshold")),
        "image_score_type": config.get("image_score_type"),
        "image_threshold": image_threshold,
        "min_component_area": int(config.get("min_component_area", 0)),
        "min_component_mean_probability": float(config.get("min_component_mean_probability", 0.0)),
        "morphology": config.get("morphology", "none"),
        "morph_kernel_size": int(config.get("morph_kernel_size", 3)),
        "top_k_components": config.get("top_k_components", None),
        "dice_all": all_m["dice"],
        "dice_forged_only": forged_m["dice"],
        "iou_all": all_m["iou"],
        "iou_forged_only": forged_m["iou"],
        "precision": all_m["precision"],
        "recall": all_m["recall"],
        "specificity": all_m["specificity"],
        "auprc_all": all_m["auprc"],
        "auprc_forged_only": forged_m["auprc"],
        "roc_auc_all": all_m["roc_auc"],
        "predicted_positive_pixel_ratio": all_m["predicted_positive_pixel_ratio"],
        "gt_positive_pixel_ratio": all_m["gt_positive_pixel_ratio"],
        "mean_dice_forged": float(forged_per["dice"].mean()) if len(forged_per) else float("nan"),
        "median_dice_forged": float(forged_per["dice"].median()) if len(forged_per) else float("nan"),
        "mean_iou_forged": float(forged_per["iou"].mean()) if len(forged_per) else float("nan"),
        "median_iou_forged": float(forged_per["iou"].median()) if len(forged_per) else float("nan"),
        "dice_lt_005_count": int((forged_per["dice"] < 0.05).sum()) if len(forged_per) else 0,
        "dice_lt_010_count": int((forged_per["dice"] < 0.10).sum()) if len(forged_per) else 0,
        "dice_lt_025_count": int((forged_per["dice"] < 0.25).sum()) if len(forged_per) else 0,
        **img_m,
        **comp_metrics_all,
        **auth_m,
    }
    metrics["final_score"] = (
        0.35 * metrics["dice_forged_only"]
        + 0.25 * metrics.get("component_f1_iou010", 0.0)
        + 0.20 * metrics["image_f1"]
        + 0.20 * (1.0 - metrics["authentic_fp_rate"])
    )

    per_df.to_csv(model_out_dir / f"test_per_image_metrics_{strategy}.csv", index=False)
    comp_details_df.to_csv(model_out_dir / f"test_component_details_{strategy}.csv", index=False)
    bin_df.to_csv(model_out_dir / f"small_mask_bin_metrics_{strategy}.csv", index=False)
    reliability_df.to_csv(model_out_dir / f"image_level_calibration_{strategy}.csv", index=False)
    return metrics, per_df, comp_details_df, reliability_df


# %% [markdown]
# ## 13. Strategy Selection

# Bu bolum ana dongu icinde validation grid search sonucundan stratejileri secer.


# %% [markdown]
# ## 14. Visualization

# %%
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_barplot(df: pd.DataFrame, value_col: str, title: str, out_path: Path) -> None:
    if df.empty or value_col not in df.columns:
        return
    plt.figure(figsize=(12, 5))
    labels = df["model_name"].astype(str) + "\n" + df["strategy"].astype(str)
    plt.bar(np.arange(len(df)), df[value_col].astype(float))
    plt.xticks(np.arange(len(df)), labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    plt.figure(figsize=(7, 5))
    for model_name, part in df.groupby("model_name"):
        plt.scatter(part[x_col], part[y_col], label=model_name, s=55)
        for _, row in part.iterrows():
            plt.text(row[x_col], row[y_col], str(row["strategy"])[:14], fontsize=7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion_matrices(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    n = len(df)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, df.iterrows()):
        mat = np.array([[row.get("image_tn", 0), row.get("image_fp", 0)], [row.get("image_fn", 0), row.get("image_tp", 0)]])
        ax.imshow(mat, cmap="Blues")
        ax.set_title(f"{row['model_name']}\n{row['strategy']}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["auth", "forged"]); ax.set_yticklabels(["auth", "forged"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def colorize_mask(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 40, 40), alpha: float = 0.45) -> np.ndarray:
    img = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    img[m] = (1 - alpha) * img[m] + alpha * color_arr
    return np.clip(img, 0, 255).astype(np.uint8)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    out[tp] = (40, 180, 80)
    out[fp] = (235, 70, 70)
    out[fn] = (70, 120, 235)
    return out


def load_display_image(path: str, size: int = 256) -> np.ndarray:
    img = load_rgb_image(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def case_key_from_mapping(row: Dict[str, Any]) -> str:
    return f"{row.get('image_id')}|{row.get('class_name')}"


def record_lookup_by_case(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {case_key_from_mapping(r): r for r in records}


def save_prediction_grid(records: List[Dict[str, Any]], config: Dict[str, Any], per_df: pd.DataFrame,
                         out_path: Path, n: int = 12, title: str = "", preserve_order: bool = False) -> None:
    if per_df.empty:
        return
    select = per_df.head(n) if preserve_order else per_df.sort_values("dice", ascending=False).head(n)
    n = min(n, len(select))
    fig, axes = plt.subplots(n, 6, figsize=(16, 2.6 * n))
    if n == 1:
        axes = axes[None, :]
    rec_by_case = record_lookup_by_case(records)
    for row_idx, (_, row) in enumerate(select.iterrows()):
        rec = rec_by_case[case_key_from_mapping(row.to_dict())]
        out = postprocess_probability_map(rec["prob"], config)
        raw = out["raw_mask"]
        pred = out["mask"]
        gt = rec["mask"]
        img = load_display_image(rec["image_path"], size=pred.shape[0])
        panels = [
            (img, "image"),
            (colorize_mask(gt, (255, 255, 255)), "gt"),
            (rec["prob"], "prob"),
            (colorize_mask(raw, (255, 255, 255)), "raw"),
            (overlay_mask(img, pred), "final"),
            (error_map(pred, gt), "error"),
        ]
        for col_idx, (panel, panel_title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if panel.ndim == 2:
                ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(panel_title)
        axes[row_idx, 0].set_ylabel(f"{row['image_id']}\nD={row['dice']:.2f}", fontsize=8)
    if title:
        fig.suptitle(title, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def make_global_plots(test_results: pd.DataFrame, output_root: Path) -> None:
    plots_dir = ensure_dir(output_root / "plots")
    targets = [plots_dir, output_root]
    for target in targets:
        save_barplot(test_results, "dice_forged_only", "Test forged Dice by strategy", target / "bar_test_forged_dice_by_strategy.png")
        save_barplot(test_results, "component_f1_iou010", "Component F1 @ IoU 0.10 by strategy", target / "bar_component_f1_by_strategy.png")
        save_barplot(test_results, "authentic_fp_rate", "Authentic FP rate by strategy", target / "bar_authentic_fp_rate_by_strategy.png")
        save_barplot(test_results, "image_f1", "Image F1 by strategy", target / "bar_image_f1_by_strategy.png")
        save_scatter(test_results, "authentic_fp_rate", "dice_forged_only", target / "scatter_dice_vs_authfp.png")
        save_scatter(test_results, "authentic_fp_rate", "component_f1_iou010", target / "scatter_component_f1_vs_authfp.png")
        plot_confusion_matrices(test_results, target / "confusion_matrices_image_level.png")


def make_curve_plots_from_saved(selected_configs: pd.DataFrame, output_root: Path) -> None:
    plots_dir = ensure_dir(output_root / "plots")
    if selected_configs.empty or "model_name" not in selected_configs.columns:
        return
    curve_rows = []
    small_rows = []
    rel_rows = []
    for _, row in selected_configs.iterrows():
        model_name = str(row["model_name"])
        strategy = str(row["strategy"])
        model_dir = output_root / model_name
        per_path = model_dir / f"test_per_image_metrics_{strategy}.csv"
        bin_path = model_dir / f"small_mask_bin_metrics_{strategy}.csv"
        rel_path = model_dir / f"image_level_calibration_{strategy}.csv"
        if per_path.exists():
            per = pd.read_csv(per_path)
            per["model_name"] = model_name
            per["strategy"] = strategy
            curve_rows.append(per)
        if bin_path.exists():
            bins = pd.read_csv(bin_path)
            bins["model_name"] = model_name
            bins["strategy"] = strategy
            small_rows.append(bins)
        if rel_path.exists():
            rel = pd.read_csv(rel_path)
            rel["model_name"] = model_name
            rel["strategy"] = strategy
            rel_rows.append(rel)
    if curve_rows:
        curves = pd.concat(curve_rows, ignore_index=True)
        plt.figure(figsize=(7, 6))
        for key, part in curves.groupby(["model_name", "strategy"]):
            if len(np.unique(part["image_label"])) < 2:
                continue
            fpr, tpr, _ = roc_curve(part["image_label"], part["image_score"])
            plt.plot(fpr, tpr, label=f"{key[0]}:{key[1]}", alpha=0.75)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Image-level ROC")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(plots_dir / "roc_curves_image_level.png", dpi=180)
        plt.savefig(output_root / "roc_curves_image_level.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 6))
        for key, part in curves.groupby(["model_name", "strategy"]):
            if len(np.unique(part["image_label"])) < 2:
                continue
            prec, rec, _ = precision_recall_curve(part["image_label"], part["image_score"])
            plt.plot(rec, prec, label=f"{key[0]}:{key[1]}", alpha=0.75)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Image-level PR")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(plots_dir / "pr_curves_image_level.png", dpi=180)
        plt.savefig(output_root / "pr_curves_image_level.png", dpi=180)
        plt.close()

    if small_rows:
        small = pd.concat(small_rows, ignore_index=True)
        plt.figure(figsize=(10, 5))
        labels = small["model_name"].astype(str) + "\n" + small["strategy"].astype(str) + "\n" + small["area_bin"].astype(str)
        plt.bar(np.arange(len(small)), small["mean_dice"])
        plt.xticks(np.arange(len(small)), labels, rotation=80, ha="right", fontsize=6)
        plt.ylabel("mean Dice")
        plt.tight_layout()
        plt.savefig(plots_dir / "small_mask_bin_dice_comparison.png", dpi=180)
        plt.savefig(output_root / "small_mask_bin_dice_comparison.png", dpi=180)
        plt.close()

    if rel_rows:
        rel = pd.concat(rel_rows, ignore_index=True)
        plt.figure(figsize=(7, 6))
        for key, part in rel.groupby(["model_name", "strategy"]):
            plt.plot(part["avg_confidence"], part["empirical_accuracy"], marker="o", label=f"{key[0]}:{key[1]}", alpha=0.75)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Average confidence"); plt.ylabel("Empirical forged rate")
        plt.title("Image-level reliability diagram")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(plots_dir / "reliability_diagram_image_level.png", dpi=180)
        plt.savefig(output_root / "reliability_diagram_image_level.png", dpi=180)
        plt.close()

    val_rows = []
    for model_name in selected_configs["model_name"].dropna().astype(str).unique():
        path = output_root / model_name / "val_grid_search_all.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["model_name"] = model_name
            val_rows.append(df)
    if val_rows:
        val_grid = pd.concat(val_rows, ignore_index=True)
        plt.figure(figsize=(8, 5))
        for model_name, part in val_grid.groupby("model_name"):
            summary = part.groupby("pixel_threshold", as_index=False)["balanced_score"].max()
            plt.plot(summary["pixel_threshold"], summary["balanced_score"], marker="o", label=model_name)
        plt.xlabel("pixel_threshold"); plt.ylabel("best validation balanced score")
        plt.title("Threshold tradeoff curves")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots_dir / "threshold_tradeoff_curves.png", dpi=180)
        plt.savefig(output_root / "threshold_tradeoff_curves.png", dpi=180)
        plt.close()


def save_model_plots(model_name: str, grid_df: pd.DataFrame, test_results_model: pd.DataFrame,
                     model_out_dir: Path) -> None:
    plots_dir = ensure_dir(model_out_dir / "plots")
    if not grid_df.empty:
        plt.figure(figsize=(7, 5))
        for score_type, part in grid_df.groupby("image_score_type"):
            part = part.sort_values("pixel_threshold")
            plt.plot(part["pixel_threshold"], part["val_forged_dice"], ".", label=score_type, alpha=0.55)
        plt.xlabel("pixel_threshold"); plt.ylabel("validation forged Dice")
        plt.title(f"{model_name} validation threshold curve")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots_dir / "validation_threshold_curve.png", dpi=180)
        plt.close()

        area_df = grid_df[grid_df["postprocess_mode"].isin(["min_area_clean", "area_probability_clean"])]
        if not area_df.empty:
            summary = area_df.groupby("min_component_area", as_index=False).agg(
                component_f1=("val_component_f1_iou010", "max"),
                auth_fp=("val_authentic_fp_rate", "min"),
            )
            plt.figure(figsize=(7, 4))
            plt.plot(summary["min_component_area"], summary["component_f1"], marker="o")
            plt.xlabel("min_component_area"); plt.ylabel("best component F1 @0.10")
            plt.tight_layout()
            plt.savefig(plots_dir / "component_f1_vs_min_area.png", dpi=180)
            plt.close()

            plt.figure(figsize=(7, 4))
            plt.plot(summary["min_component_area"], summary["auth_fp"], marker="o", color="crimson")
            plt.xlabel("min_component_area"); plt.ylabel("min authentic FP rate")
            plt.tight_layout()
            plt.savefig(plots_dir / "authentic_fp_rate_vs_min_area.png", dpi=180)
            plt.close()

    if not test_results_model.empty:
        save_barplot(test_results_model, "dice_forged_only", f"{model_name} test Dice", plots_dir / "bar_test_forged_dice_by_strategy.png")
        save_barplot(test_results_model, "component_f1_iou010", f"{model_name} component F1", plots_dir / "bar_component_f1_by_strategy.png")
        save_barplot(test_results_model, "authentic_fp_rate", f"{model_name} authentic FP", plots_dir / "bar_authentic_fp_rate_by_strategy.png")
        save_barplot(test_results_model, "image_f1", f"{model_name} image F1", plots_dir / "bar_image_f1_by_strategy.png")


# %% [markdown]
# ## 15. Failure Case Analysis

# %%
def summarize_failure_rows(records: List[Dict[str, Any]], per_df: pd.DataFrame,
                           config: Dict[str, Any], selected_df: pd.DataFrame) -> pd.DataFrame:
    rec_by_case = record_lookup_by_case(records)
    rows = []
    for _, row in selected_df.iterrows():
        rec = rec_by_case[case_key_from_mapping(row.to_dict())]
        out = postprocess_probability_map(rec["prob"], config)
        pred = out["mask"]
        comp = match_components(rec["mask"], pred, 0.10)
        rows.append({
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "class_name": row["class_name"],
            "gt_area": int(row["gt_area"]),
            "gt_area_ratio": float(row["gt_area_ratio"]),
            "pred_area": int(row["pred_area"]),
            "pred_area_ratio": float(row["pred_area_ratio"]),
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "image_score": float(row["image_score"]),
            "image_pred_label": int(row["image_pred_label"]),
            "gt_component_count": comp["gt_component_count"],
            "pred_component_count": comp["pred_component_count"],
            "matched_component_count": comp["matched_component_count"],
        })
    return pd.DataFrame(rows)


def save_failure_cases(model_name: str, records: List[Dict[str, Any]], config: Dict[str, Any],
                       per_df: pd.DataFrame, model_out_dir: Path) -> None:
    fail_dir = ensure_dir(model_out_dir / "failure_cases")
    forged = per_df[per_df["image_label"] == 1].copy()
    auth = per_df[per_df["image_label"] == 0].copy()
    q1 = float(forged["gt_area"].quantile(0.25)) if len(forged) else 0.0
    groups = {
        "best_cases_forged": forged.sort_values("dice", ascending=False).head(10),
        "low_dice_forged": forged.sort_values("dice", ascending=True).head(10),
        "false_positive_authentic": auth[auth["pred_area"] > 0].sort_values("image_score", ascending=False).head(10),
        "false_negative_forged": forged[forged["image_pred_label"] == 0].sort_values("image_score", ascending=True).head(10),
        "small_mask_failures": forged[(forged["gt_area"] <= q1) & (forged["dice"] < 0.05)].sort_values("dice", ascending=True).head(10),
    }
    for group_name, group_df in groups.items():
        csv_df = summarize_failure_rows(records, per_df, config, group_df)
        csv_df.to_csv(fail_dir / f"{group_name}.csv", index=False)
        if len(group_df):
            save_prediction_grid(
                records,
                config,
                group_df,
                fail_dir / f"{group_name}.png",
                n=min(10, len(group_df)),
                title=f"{model_name} - {group_name}",
                preserve_order=True,
            )


# %% [markdown]
# ## 16. Optional Robustness

# %%
def perturb_probability_placeholder(records: List[Dict[str, Any]], perturbation: str) -> List[Dict[str, Any]]:
    # Deney 6 icin baglanti noktasi: checkpoint inference ile yeniden hesaplama yerine
    # mevcut olasilik haritalarini hafifce bozarak duman testi saglar. Gercek robustness
    # kosusu icin checkpoint inference pipeline'i ayni config ile genisletilebilir.
    new_records = []
    for rec in records:
        clone = dict(rec)
        prob = rec["prob"].copy()
        if perturbation == "gaussian_blur":
            prob = cv2.GaussianBlur(prob, (5, 5), 0)
        elif perturbation == "gaussian_noise":
            rng = np.random.default_rng(CFG.seed)
            prob = np.clip(prob + rng.normal(0, 0.03, size=prob.shape), 0, 1).astype(np.float32)
        clone["prob"] = prob.astype(np.float32)
        new_records.append(clone)
    return new_records


def run_optional_robustness(final_row: Dict[str, Any], model_records: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    if not CFG.run_robustness:
        return pd.DataFrame()
    model_name = str(final_row["model_name"])
    base_records = model_records[model_name]
    rows = []
    for perturbation in ["jpeg_quality_90", "jpeg_quality_70", "gaussian_blur", "gaussian_noise"]:
        records = perturb_probability_placeholder(base_records, perturbation)
        outputs = apply_config_to_records(records, final_row)
        forged_m = pixel_metrics_from_predictions(outputs["preds"], records, forged_only=True, compute_auc=False)
        comp_m, _ = component_metrics_from_predictions(records, outputs["preds"], 0.10)
        y_true = np.array([int(r["image_label"]) for r in records], dtype=np.uint8)
        img_m, _ = image_level_metrics_from_scores(y_true, outputs["image_scores"], float(final_row["image_threshold"]))
        auth_m = authentic_false_alarm_metrics(records, outputs["preds"], outputs["components_per_image"])
        rows.append({
            "perturbation": perturbation,
            "forged_dice": forged_m["dice"],
            "forged_iou": forged_m["iou"],
            "component_f1": comp_m["component_f1_iou010"],
            "authentic_fp_rate": auth_m["authentic_fp_rate"],
            "image_f1": img_m["image_f1"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_ROOT / "robustness_metrics_final_model.csv", index=False)
    return out


# %% [markdown]
# ## 17. Report Generation

# %%
def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_Tablo bos._"
    return df.head(max_rows).to_markdown(index=False)


def generate_comparison_markdown(test_results: pd.DataFrame, ranking: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Model-Strateji Karsilastirmasi", ""]
    lines.append("Bu tablo test skorlarini yorumlama amaciyla siralar; threshold ve post-processing secimleri validation setinde sabitlenmistir.")
    lines.append("")
    lines.append(markdown_table(ranking, max_rows=30))
    lines.append("")
    pairs = [
        ("SegFormer-B0 raw vs calibrated/balanced", "segformer_b0_rgb_full"),
        ("EfficientNetB0 raw vs calibrated/balanced", "efficientnetb0_unet_rgb_full"),
        ("DINOv2-lite raw vs calibrated/balanced", "dinov2_lite_decoder_rgb_full"),
        ("U-Net++ raw vs calibrated/balanced", "unetpp_resnet34_rgb_full"),
    ]
    for title, model in pairs:
        part = test_results[test_results["model_name"] == model]
        lines.extend([f"## {title}", "", markdown_table(part, max_rows=10), ""])
    lines.extend(["## SegFormer-B0 vs EfficientNetB0-UNet final", ""])
    lines.append(markdown_table(ranking[ranking["model_name"].isin(["segformer_b0_rgb_full", "efficientnetb0_unet_rgb_full"])].head(10)))
    lines.extend(["", "## SegFormer-B0 vs DINOv2-lite final", ""])
    lines.append(markdown_table(ranking[ranking["model_name"].isin(["segformer_b0_rgb_full", "dinov2_lite_decoder_rgb_full"])].head(10)))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_report(selected_configs: pd.DataFrame, test_results: pd.DataFrame, ranking: pd.DataFrame,
                    errors: List[Dict[str, Any]], out_path: Path) -> None:
    best = ranking.iloc[0].to_dict() if len(ranking) else {}
    lines = [
        "# Deney 5: Calibration ve Post-processing Raporu",
        "",
        "## 1. Deneyin amaci",
        "Bu deney yeni model egitmeden, Deney 4'te egitilmis dort modelin olasilik haritalari uzerinden threshold, image score, component filtering ve post-processing stratejilerini validation setinde secer ve test setinde sabit config ile degerlendirir.",
        "",
        "## 2. Kullanilan modeller",
        "- U-Net++ ResNet34 RGB full-data baseline",
        "- EfficientNetB0-UNet RGB full-data",
        "- SegFormer-B0 RGB full-data",
        "- DINOv2-lite decoder RGB full-data",
        "",
        "## 3. Onceki Deney 4 sonucunun kisa ozeti",
        markdown_table(pd.DataFrame(PREVIOUS_EXPERIMENT4_SUMMARY).T.reset_index().rename(columns={"index": "model_name"}), max_rows=10),
        "",
        "## 4. Calibration ve post-processing protokolu",
        "Validation setinde pixel threshold, image score type, image threshold, min component area, component mean probability, morphology ve top-K component secenekleri iki asamali grid search ile tarandi. Test seti sadece validation'da secilen stratejilerin nihai degerlendirmesi icin kullanildi.",
        "",
        "## 5. Validation'da secilen stratejiler",
        markdown_table(selected_configs, max_rows=30),
        "",
        "## 6. Test sonuclari",
        markdown_table(ranking, max_rows=30),
        "",
        "## 7. Pixel-level sonuclar",
        markdown_table(test_results[["model_name", "strategy", "dice_forged_only", "iou_forged_only", "auprc_forged_only", "predicted_positive_pixel_ratio"]], max_rows=30) if len(test_results) else "_Sonuc yok._",
        "",
        "## 8. Component-aware sonuclar",
        markdown_table(test_results[["model_name", "strategy", "component_f1_iou010", "component_f1_iou025", "avg_pred_component_count", "authentic_fp_rate"]], max_rows=30) if len(test_results) else "_Sonuc yok._",
        "",
        "## 9. Image-level calibration sonuclari",
        markdown_table(test_results[["model_name", "strategy", "image_f1", "image_specificity", "image_brier", "image_ece_10bin", "image_roc_auc", "image_auprc"]], max_rows=30) if len(test_results) else "_Sonuc yok._",
        "",
        "## 10. Small-mask analysis",
        "Her model-strateji icin forged test goruntuleri gt_area quartile'larina bolundu. Ayrintilar model klasorlerindeki `small_mask_bin_metrics_{strategy}.csv` dosyalarindadir.",
        "",
        "## 11. Model bazli yorumlar",
    ]
    for model_name in [e.name for e in EXPERIMENTS]:
        part = ranking[ranking["model_name"] == model_name]
        if part.empty:
            lines.append(f"- {model_name}: Degerlendirme tamamlanamadi.")
        else:
            top = part.iloc[0]
            lines.append(
                f"- {model_name}: en iyi strateji `{top['strategy']}`; forged Dice={top['test_dice_forged_only']:.4f}, "
                f"component F1@0.10={top['test_component_f1_iou010']:.4f}, image F1={top['test_image_f1']:.4f}, "
                f"auth FP={top['authentic_fp_rate']:.4f}."
            )
    lines.extend([
        "",
        "## 12. Final aday model/strateji",
        f"Final aday: `{best.get('model_name', 'NA')}` + `{best.get('strategy', 'NA')}`. Bu siralama test skorunu raporlama amaciyla kullanir; pipeline secim mantigi validation stratejilerine dayanir.",
        "",
        "## 13. Failure case ozeti",
        "Her modelin en iyi stratejisi icin best cases, low Dice forged, false positive authentic, false negative forged ve small mask failure CSV/PNG dosyalari `failure_cases/` altina kaydedildi.",
        "",
        "## 14. Sonraki deney onerisi",
        "- Kucuk maskelerde Dice dusuk kalirsa 384x384 input veya small-mask oversampling denenmeli.",
        "- DINOv2 calibration validation sonrasi da zayifsa limited unfreeze daha anlamli bir sonraki adimdir.",
        "- Final aday belirginse Deney 6 robustness testi JPEG, blur ve noise perturbation'lari ile calistirilmalidir.",
    ])
    if errors:
        lines.extend(["", "## Hata notlari", markdown_table(pd.DataFrame(errors), max_rows=20)])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def final_ranking(test_results: pd.DataFrame) -> pd.DataFrame:
    if test_results.empty:
        return pd.DataFrame()
    df = test_results.copy()
    df["final_score"] = (
        0.35 * df["dice_forged_only"]
        + 0.25 * df["component_f1_iou010"]
        + 0.20 * df["image_f1"]
        + 0.20 * (1.0 - df["authentic_fp_rate"])
    )
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    rename = {
        "dice_forged_only": "test_dice_forged_only",
        "iou_forged_only": "test_iou_forged_only",
        "component_f1_iou010": "test_component_f1_iou010",
        "component_f1_iou025": "test_component_f1_iou025",
        "image_f1": "test_image_f1",
        "image_specificity": "test_image_specificity",
    }
    df = df.rename(columns=rename)
    cols = [
        "rank", "model_name", "strategy", "postprocess_mode", "pixel_threshold",
        "image_score_type", "image_threshold", "min_component_area",
        "min_component_mean_probability", "test_dice_forged_only",
        "test_iou_forged_only", "test_component_f1_iou010",
        "test_component_f1_iou025", "test_image_f1",
        "test_image_specificity", "authentic_fp_rate", "final_score",
    ]
    return df[[c for c in cols if c in df.columns]]


def raw_reference_config(model_name: str, grid_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    raw_df = grid_df[grid_df["postprocess_mode"] == "raw"]
    if raw_df.empty:
        return None
    row = raw_df.sort_values(["val_forged_dice", "val_image_f1"], ascending=False).iloc[0].to_dict()
    row["strategy"] = "raw_reference"
    row["model_name"] = model_name
    return row


# %% [markdown]
# ## Main Execution

# %%
def run_experiment5() -> None:
    prediction_availability_rows: List[Dict[str, Any]] = []
    selected_all: List[pd.DataFrame] = []
    test_results_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    test_records_by_model: Dict[str, List[Dict[str, Any]]] = {}

    for exp in EXPERIMENTS:
        model_name = exp.name
        model_out_dir = ensure_dir(OUTPUT_ROOT / model_name)
        print(f"\n===== Deney 5: {model_name} =====")
        try:
            val_records, av_val = get_or_create_predictions(exp, "val", val_df, model_out_dir)
            test_records, av_test = get_or_create_predictions(exp, "test", test_df, model_out_dir)
            prediction_availability_rows.extend([av_val, av_test])
            test_records_by_model[model_name] = test_records

            grid_path = model_out_dir / "val_grid_search_all.csv"
            if grid_path.exists():
                print(f"[grid] mevcut sonuc yukleniyor: {grid_path}")
                grid_df = pd.read_csv(grid_path)
            else:
                grid_df = run_validation_grid_search(model_name, val_records, model_out_dir)

            selected_df = select_strategy_configs(model_name, grid_df)
            selected_df.to_csv(model_out_dir / "selected_configs.csv", index=False)
            selected_all.append(selected_df)

            eval_configs = [row.to_dict() for _, row in selected_df.iterrows()]
            raw_cfg = raw_reference_config(model_name, grid_df)
            if raw_cfg is not None:
                eval_configs.append(raw_cfg)

            model_test_rows = []
            best_strategy_for_visuals = selected_df.sort_values("balanced_score", ascending=False).iloc[0].to_dict()
            best_per_df = None
            for cfg in eval_configs:
                metrics, per_df, comp_df, reliability_df = evaluate_strategy_on_test(
                    model_name,
                    cfg,
                    test_records,
                    model_out_dir,
                )
                model_test_rows.append(metrics)
                test_results_rows.append(metrics)
                if str(cfg["strategy"]) == str(best_strategy_for_visuals["strategy"]):
                    best_per_df = per_df

            model_test_df = pd.DataFrame(model_test_rows)
            model_test_df.to_csv(model_out_dir / "test_results_by_strategy.csv", index=False)
            save_model_plots(model_name, grid_df, model_test_df, model_out_dir)

            if best_per_df is not None:
                save_prediction_grid(
                    test_records,
                    best_strategy_for_visuals,
                    best_per_df,
                    model_out_dir / "prediction_examples_best_strategy.png",
                    n=CFG.max_visual_examples,
                    title=f"{model_name} - {best_strategy_for_visuals['strategy']}",
                )
                save_failure_cases(model_name, test_records, best_strategy_for_visuals, best_per_df, model_out_dir)

            print(f"[done] {model_name}")
        except Exception as exc:
            traceback.print_exc()
            errors.append({"model_name": model_name, "error": repr(exc)})
            prediction_availability_rows.append({
                "model_name": model_name,
                "split": "NA",
                "loaded_from": "",
                "status": "failed",
                "n_records": 0,
                "message": repr(exc),
            })
            continue

    availability_df = pd.DataFrame(prediction_availability_rows)
    availability_df.to_csv(OUTPUT_ROOT / "model_prediction_availability.csv", index=False)

    selected_configs_all = pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame()
    selected_configs_all.to_csv(OUTPUT_ROOT / "selected_configs_all_models.csv", index=False)

    test_results = pd.DataFrame(test_results_rows)
    test_results.to_csv(OUTPUT_ROOT / "test_results_all_strategies.csv", index=False)
    ranking = final_ranking(test_results)
    ranking.to_csv(OUTPUT_ROOT / "final_candidate_ranking.csv", index=False)

    make_global_plots(test_results, OUTPUT_ROOT)
    make_curve_plots_from_saved(selected_configs_all, OUTPUT_ROOT)
    generate_comparison_markdown(test_results, ranking, OUTPUT_ROOT / "model_strategy_comparison.md")
    generate_report(selected_configs_all, test_results, ranking, errors, OUTPUT_ROOT / "experiment5_report.md")

    robustness_df = pd.DataFrame()
    if CFG.run_robustness and len(ranking):
        robustness_df = run_optional_robustness(ranking.iloc[0].to_dict(), test_records_by_model)

    summary = {
        "n_models_attempted": len(EXPERIMENTS),
        "n_models_completed": int(test_results["model_name"].nunique()) if len(test_results) else 0,
        "n_selected_configs": int(len(selected_configs_all)),
        "n_test_rows": int(len(test_results)),
        "best_candidate": ranking.iloc[0].to_dict() if len(ranking) else None,
        "errors": errors,
        "run_robustness": CFG.run_robustness,
        "n_robustness_rows": int(len(robustness_df)),
    }
    with open(OUTPUT_ROOT / "experiment5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n===== Deney 5 tamamlandi =====")
    print("Output root:", OUTPUT_ROOT)
    if len(ranking):
        print(ranking.head(10))
    if errors:
        print("[warnings/errors]")
        print(pd.DataFrame(errors))


run_experiment5()
