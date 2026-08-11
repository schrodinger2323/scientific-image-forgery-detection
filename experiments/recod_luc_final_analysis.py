# %% [markdown]
# # Recod.ai/LUC - Final Analiz Notebook'u
#
# Bu notebook yeni model egitmez. Deney 6 sonrasi belirlenen iki final aday modeli
# clean test seti, robustness bozulmalari, istatistiksel testler, failure case
# gruplari ve tezde kullanilacak tablo/grafikler uzerinden degerlendirir.
#
# Ana kural: Test setinde threshold veya post-processing parametresi secilmez.
# Deney 6 validation setinde secilmis strateji ayarlari clean, JPEG, blur ve noise
# kosullarinda sabit tutulur.

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
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*The secret `HF_TOKEN` does not exist.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")


def ensure_package(import_name: str, pip_name: Optional[str] = None) -> None:
    """Colab/Kaggle ortaminda eksik paketleri kurar."""
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"[install] {pip_name} kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure_package("cv2", "opencv-python-headless")
ensure_package("sklearn", "scikit-learn")
ensure_package("matplotlib")
ensure_package("seaborn")
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
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import albumentations as A
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation

try:
    from torch.amp import autocast
    TORCH_AMP_NEW_API = True
except Exception:
    from torch.cuda.amp import autocast
    TORCH_AMP_NEW_API = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EPS = 1e-7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(style="whitegrid", context="paper")


# %% [markdown]
# ## 2. Global Config

# %%
@dataclass
class ModelSpec:
    name: str
    model_type: str
    role: str
    preferred_strategy: str
    encoder_or_backbone: str
    image_size: int = 384
    classes: int = 1
    encoder_weights: Optional[str] = "imagenet"
    hf_model_name: Optional[str] = None
    pretrained: bool = True


@dataclass
class FinalAnalysisConfig:
    seed: int = 42
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    use_amp: bool = True
    create_submission: bool = False
    save_probability_maps: bool = True
    max_visual_examples: int = 12
    bootstrap_iterations: int = 5000
    robustness_enabled: bool = True
    reuse_existing_robustness_metrics: bool = True
    robustness_include_combined: bool = True
    component_iou_thresholds: Tuple[float, ...] = (0.10, 0.25, 0.50)
    degradation_order: Tuple[str, ...] = (
        "clean_png",
        "jpeg_q90",
        "jpeg_q70",
        "jpeg_q50",
        "gaussian_blur_light",
        "gaussian_blur_medium",
        "gaussian_noise_light",
        "gaussian_noise_medium",
        "combined_jpeg70_blur_light",
    )


CFG = FinalAnalysisConfig()

FINAL_MODELS = [
    ModelSpec(
        name="segformer_b0_rgb_384_smallmask",
        model_type="segformer",
        role="best_localization_model",
        preferred_strategy="balanced_final_score",
        encoder_or_backbone="nvidia/segformer-b0-finetuned-ade-512-512",
        hf_model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        encoder_weights=None,
    ),
    ModelSpec(
        name="efficientnetb0_unet_rgb_384_smallmask",
        model_type="unet",
        role="low_false_alarm_model",
        preferred_strategy="balanced_or_low_false_alarm",
        encoder_or_backbone="efficientnet-b0",
        encoder_weights="imagenet",
    ),
]

REFERENCE_MODELS = [
    {
        "model_name": "unetpp_resnet34_rgb_full",
        "display_name": "U-Net++ ResNet34 256",
        "role": "256_baseline",
        "image_size": 256,
        "preferred_strategy": "raw_reference",
    },
    {
        "model_name": "efficientnetb0_unet_rgb_full",
        "display_name": "EfficientNetB0-UNet 256",
        "role": "256_reference",
        "image_size": 256,
        "preferred_strategy": "balanced_final_score",
    },
    {
        "model_name": "segformer_b0_rgb_full",
        "display_name": "SegFormer-B0 256",
        "role": "256_reference",
        "image_size": 256,
        "preferred_strategy": "balanced_final_score",
    },
    {
        "model_name": "dinov2_lite_decoder_rgb_full",
        "display_name": "DINOv2-lite 256",
        "role": "256_reference",
        "image_size": 256,
        "preferred_strategy": "balanced_final_score",
    },
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
print("DEVICE:", DEVICE)


# %% [markdown]
# ## 3. Path Discovery

# %%
def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_paths() -> Dict[str, Optional[Path]]:
    dataset_candidates = [
        Path("/kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection"),
        Path("/content/drive/MyDrive/bitirmeProjesi/dataset"),
        Path("dataset"),
    ]
    experiments_full_candidates = [
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full"),
        Path("/kaggle/working/experiments_full"),
        Path("experiments_full"),
        Path("deney_6/experiments_full"),
        Path("deney_5/experiments_4_full"),
    ]
    split_candidates = [
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/_shared_splits_seed42"),
        Path("/kaggle/working/experiments_full/_shared_splits_seed42"),
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments/_shared_splits_seed42"),
        Path("/kaggle/working/experiments/_shared_splits_seed42"),
        Path("experiments_full/_shared_splits_seed42"),
        Path("experiments/_shared_splits_seed42"),
        Path("deney_4/_shared_splits_seed42"),
        Path("deney_5/_shared_splits_seed42"),
    ]
    exp6_candidates = [
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/experiment6_smallmask_384"),
        Path("/kaggle/working/experiments_full/experiment6_smallmask_384"),
        Path("experiments_full/experiment6_smallmask_384"),
        Path("deney_6/experiments_full/experiment6_smallmask_384"),
    ]
    exp5_candidates = [
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/experiment5_calibration_postprocessing"),
        Path("/kaggle/working/experiments_full/experiment5_calibration_postprocessing"),
        Path("experiments_full/experiment5_calibration_postprocessing"),
        Path("deney_5/experiments_4_full/experiment5_calibration_postprocessing"),
    ]
    exp4_candidates = [
        Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full"),
        Path("/kaggle/working/experiments_full"),
        Path("experiments_full"),
        Path("deney_5/experiments_4_full"),
    ]

    dataset_root = first_existing(dataset_candidates)
    split_dir = first_existing(split_candidates)
    exp6_root = first_existing(exp6_candidates)
    exp5_root = first_existing(exp5_candidates)
    exp4_root = first_existing(exp4_candidates)

    if dataset_root is None:
        raise FileNotFoundError("Dataset root bulunamadi. Kaggle input veya Google Drive dataset klasorunu yukleyin.")
    if split_dir is None:
        raise FileNotFoundError("Shared split klasoru bulunamadi. Yeni split olusturulmayacak.")
    for filename in ["full.csv", "train.csv", "val.csv", "test.csv"]:
        if not (split_dir / filename).exists():
            raise FileNotFoundError(f"Split dosyasi eksik: {split_dir / filename}")

    base_exp_root = first_existing(experiments_full_candidates)
    if base_exp_root is None:
        base_exp_root = Path("/kaggle/working/experiments_full") if Path("/kaggle/working").exists() else Path("experiments_full")
    final_root = base_exp_root / "final_analysis"
    final_root.mkdir(parents=True, exist_ok=True)
    for folder in ["plots", "failure_cases", "robustness", "statistics", "tables"]:
        (final_root / folder).mkdir(parents=True, exist_ok=True)

    return {
        "dataset_root": dataset_root,
        "split_dir": split_dir,
        "experiments_root": base_exp_root,
        "exp6_root": exp6_root,
        "exp5_root": exp5_root,
        "exp4_root": exp4_root,
        "final_root": final_root,
    }


PATHS = discover_paths()
DATASET_ROOT = PATHS["dataset_root"]
SPLIT_DIR = PATHS["split_dir"]
EXP6_ROOT = PATHS["exp6_root"]
EXP5_ROOT = PATHS["exp5_root"]
EXP4_ROOT = PATHS["exp4_root"]
FINAL_ROOT = PATHS["final_root"]
PLOTS_DIR = FINAL_ROOT / "plots"

print("DATASET_ROOT:", DATASET_ROOT)
print("SPLIT_DIR:", SPLIT_DIR)
print("EXP6_ROOT:", EXP6_ROOT)
print("EXP5_ROOT:", EXP5_ROOT)
print("EXP4_ROOT:", EXP4_ROOT)
print("FINAL_ROOT:", FINAL_ROOT)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


save_json(
    {
        "config": asdict(CFG),
        "final_models": [asdict(m) for m in FINAL_MODELS],
        "reference_models": REFERENCE_MODELS,
        "paths": {k: str(v) if v is not None else None for k, v in PATHS.items()},
    },
    FINAL_ROOT / "final_analysis_config.json",
)

save_json(
    {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    },
    FINAL_ROOT / "environment_info.json",
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


def overlap_count(a: pd.DataFrame, b: pd.DataFrame) -> int:
    return len(set(a["image_id"].astype(str)).intersection(set(b["image_id"].astype(str))))


def split_counts(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    labels = df["image_label"].astype(int)
    return {
        "split": split_name,
        "count": int(len(df)),
        "authentic": int((labels == 0).sum()),
        "forged": int((labels == 1).sum()),
    }


split_summary = pd.DataFrame(
    [
        split_counts(train_df, "train"),
        split_counts(val_df, "val"),
        split_counts(test_df, "test"),
        {"split": "leak_train_val", "count": overlap_count(train_df, val_df), "authentic": 0, "forged": 0},
        {"split": "leak_train_test", "count": overlap_count(train_df, test_df), "authentic": 0, "forged": 0},
        {"split": "leak_val_test", "count": overlap_count(val_df, test_df), "authentic": 0, "forged": 0},
    ]
)
split_summary.to_csv(FINAL_ROOT / "split_summary.csv", index=False)
print(split_summary)

if any(split_summary[split_summary["split"].str.startswith("leak_")]["count"].astype(int) != 0):
    raise RuntimeError("Leakage kontrolu basarisiz: splitler arasinda image_id overlap var.")


# %% [markdown]
# ## 5. Dataset and Mask Utilities

# %%
def load_image_rgb(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
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
        gt_area = 0
        if int(row["image_label"]) == 1:
            gt_area = int(load_mask_array(row["mask_path"], (h, w)).sum())
        out = row.to_dict()
        out.update(
            {
                "height": int(h),
                "width": int(w),
                "image_area": int(h * w),
                "gt_area": int(gt_area),
                "gt_area_ratio": float(gt_area / max(h * w, 1)),
            }
        )
        rows.append(out)
    out_df = pd.DataFrame(rows)
    forged = out_df[out_df["image_label"].astype(int) == 1].copy()
    out_df["mask_quartile"] = ""
    if len(forged) >= 4:
        q1, q2, q3 = np.quantile(forged["gt_area"].astype(float), [0.25, 0.50, 0.75])
        out_df.loc[forged.index, "mask_quartile"] = pd.cut(
            forged["gt_area"].astype(float),
            bins=[-1, q1, q2, q3, np.inf],
            labels=["Q1", "Q2", "Q3", "Q4"],
            include_lowest=True,
        ).astype(str)
    out_df.to_csv(FINAL_ROOT / f"mask_area_summary_{split_name}.csv", index=False)
    return out_df


train_df = add_mask_area_info(train_df, "train")
val_df = add_mask_area_info(val_df, "val")
test_df = add_mask_area_info(test_df, "test")


def get_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )


class ForgeryEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, degradation: str = "clean_png"):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.degradation = degradation
        self.transforms = get_eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image = load_image_rgb(row["image_path"])
        image = apply_degradation(image, self.degradation)
        if int(row["image_label"]) == 1:
            mask = load_mask_array(row["mask_path"], image.shape[:2])
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        augmented = self.transforms(image=image, mask=mask)
        image_tensor = torch.from_numpy(augmented["image"].astype(np.float32).transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy((augmented["mask"] > 0.5).astype(np.float32)[None]).float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "idx": torch.tensor(idx, dtype=torch.long),
            "image_id": str(row["image_id"]),
            "image_label": torch.tensor(int(row["image_label"]), dtype=torch.long),
        }


def make_records_from_df(df: pd.DataFrame, image_size: int) -> List[Dict[str, Any]]:
    records = []
    for i, row in df.reset_index(drop=True).iterrows():
        mask = load_mask_array(row["mask_path"], (int(row["height"]), int(row["width"]))) if int(row["image_label"]) == 1 else np.zeros((int(row["height"]), int(row["width"])), np.float32)
        mask = cv2.resize(mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        rec = row.to_dict()
        rec.update({"mask": mask, "prob_index": int(i)})
        records.append(rec)
    return records


# %% [markdown]
# ## 6. Model Builders and Checkpoint Loading

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


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.model_type == "unet":
        return smp.Unet(
            encoder_name=spec.encoder_or_backbone,
            encoder_weights=spec.encoder_weights if spec.pretrained else None,
            in_channels=3,
            classes=spec.classes,
            activation=None,
        )
    if spec.model_type == "segformer":
        return SegFormerBinaryWrapper(spec.hf_model_name or spec.encoder_or_backbone)
    raise ValueError(f"Bilinmeyen model_type: {spec.model_type}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_params": int(trainable)}


def model_output_dir(model_name: str) -> Path:
    out = FINAL_ROOT / model_name
    (out / "failure_cases").mkdir(parents=True, exist_ok=True)
    (out / "robustness").mkdir(parents=True, exist_ok=True)
    return out


def source_model_dir(model_name: str) -> Optional[Path]:
    candidates = []
    if EXP6_ROOT is not None:
        candidates.append(EXP6_ROOT / model_name)
    candidates.extend(
        [
            Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/experiment6_smallmask_384") / model_name,
            Path("/kaggle/working/experiments_full/experiment6_smallmask_384") / model_name,
            Path("experiments_full/experiment6_smallmask_384") / model_name,
            Path("deney_6/experiments_full/experiment6_smallmask_384") / model_name,
        ]
    )
    return first_existing(candidates)


def load_checkpoint_model(spec: ModelSpec) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    src = source_model_dir(spec.name)
    info = {"model_name": spec.name, "checkpoint_found": False, "message": ""}
    if src is None:
        info["message"] = "Deney 6 model klasoru bulunamadi."
        return None, info
    ckpt_path = src / "best_model.pth"
    if not ckpt_path.exists():
        info["message"] = f"Checkpoint bulunamadi: {ckpt_path}"
        return None, info
    try:
        model = build_model(spec).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        model.eval()
        info.update({"checkpoint_found": True, "checkpoint_path": str(ckpt_path), **count_parameters(model)})
        return model, info
    except Exception as exc:
        info["message"] = f"Checkpoint yukleme hatasi: {exc}"
        return None, info


# %% [markdown]
# ## 7. Prediction / Inference Utilities

# %%
def amp_context():
    enabled = CFG.use_amp and DEVICE.type == "cuda"
    if TORCH_AMP_NEW_API:
        return autocast(device_type="cuda", enabled=enabled)
    return autocast(enabled=enabled)


def load_npz_records(model_name: str, split_name: str, df: pd.DataFrame, image_size: int) -> Optional[List[Dict[str, Any]]]:
    src = source_model_dir(model_name)
    if src is None:
        return None
    npz_candidates = [
        src / f"{split_name}_prob_maps.npz",
        src / f"{split_name}_predictions_probs.npz",
        src / f"{split_name}_prediction_probs.npz",
        src / f"{split_name}_pred_probs.npz",
    ]
    npz_path = first_existing(npz_candidates)
    if npz_path is None:
        return None
    data = np.load(npz_path, allow_pickle=True)
    if "probs" in data:
        probs = data["probs"].astype(np.float32)
    elif "arr_0" in data:
        probs = data["arr_0"].astype(np.float32)
    else:
        raise ValueError(f"NPZ icinde probs/arr_0 bulunamadi: {npz_path}")
    df_use = df.copy()
    meta_path = src / f"{split_name}_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "image_id" in meta.columns and len(meta) == len(probs):
            meta_order = meta.copy()
            meta_order["image_id"] = meta_order["image_id"].astype(str)
            df_tmp = df_use.copy()
            df_tmp["image_id"] = df_tmp["image_id"].astype(str)

            join_cols = ["image_id"]
            if "class_name" in meta_order.columns and "class_name" in df_tmp.columns:
                meta_order["class_name"] = meta_order["class_name"].astype(str)
                df_tmp["class_name"] = df_tmp["class_name"].astype(str)
                join_cols = ["image_id", "class_name"]

            meta_order["_prob_order"] = np.arange(len(meta_order))
            aligned = meta_order[join_cols + ["_prob_order"]].merge(df_tmp, on=join_cols, how="left", validate="one_to_one")
            if len(aligned) == len(probs) and not aligned["image_path"].isna().any():
                df_use = aligned.sort_values("_prob_order").drop(columns=["_prob_order"]).reset_index(drop=True)
            else:
                print(f"[warn] {model_name} {split_name}: metadata split ile birebir eslesmedi, sirali eslestirme kullaniliyor.")
                df_use = df.iloc[: len(probs)].copy()
        else:
            df_use = df.iloc[: len(probs)].copy()
    else:
        if len(probs) != len(df):
            print(f"[warn] {model_name} {split_name}: probs sayisi split ile eslesmiyor ({len(probs)} vs {len(df)}). Sirali eslestirme yapiliyor.")
        df_use = df.iloc[: len(probs)].copy()
    if len(df_use) != len(probs):
        print(f"[warn] {model_name} {split_name}: hizalanmis kayit sayisi {len(df_use)}, prob sayisi {len(probs)}. Ortak uzunluga kirpiliyor.")
        df_use = df_use.iloc[: len(probs)].copy()
    records = make_records_from_df(df_use, image_size)
    for i, rec in enumerate(records):
        rec["prob"] = probs[i]
    out_dir = model_output_dir(model_name)
    if CFG.save_probability_maps:
        np.savez_compressed(out_dir / f"{split_name}_prob_maps.npz", probs=probs.astype(np.float16))
        pd.DataFrame([{k: v for k, v in r.items() if k not in ("mask", "prob")} for r in records]).to_csv(out_dir / f"{split_name}_metadata.csv", index=False)
    return records


@torch.no_grad()
def run_inference_records(model: nn.Module, spec: ModelSpec, df: pd.DataFrame, degradation: str, save_prefix: Optional[Path]) -> Tuple[List[Dict[str, Any]], float]:
    ds = ForgeryEvalDataset(df, spec.image_size, degradation=degradation)
    batch_size = CFG.batch_size if spec.model_type == "segformer" else max(CFG.batch_size, 4)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    model.eval()
    records = make_records_from_df(df, spec.image_size)
    probs_all = []
    start = time.time()
    for batch in tqdm(loader, desc=f"{spec.name} {degradation}"):
        images = batch["image"].to(DEVICE, non_blocking=True)
        with amp_context():
            logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0].astype(np.float32)
        probs_all.append(probs)
    elapsed = time.time() - start
    probs_arr = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, spec.image_size, spec.image_size), dtype=np.float32)
    for i, rec in enumerate(records):
        rec["prob"] = probs_arr[i]
    if save_prefix is not None and CFG.save_probability_maps:
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_prefix.with_suffix(".npz"), probs=probs_arr.astype(np.float16))
    inference_time = elapsed / max(len(records), 1)
    return records, float(inference_time)


def get_clean_records(model: Optional[nn.Module], spec: ModelSpec) -> Tuple[Optional[List[Dict[str, Any]]], float, str]:
    cached = load_npz_records(spec.name, "test", test_df, spec.image_size)
    if cached is not None:
        src = source_model_dir(spec.name)
        time_val = np.nan
        if src is not None and (src / "test_results_by_strategy.csv").exists():
            prev = pd.read_csv(src / "test_results_by_strategy.csv")
            col = "inference_time_per_image_sec"
            if col in prev.columns and len(prev):
                time_val = float(pd.to_numeric(prev[col], errors="coerce").dropna().iloc[0])
        return cached, time_val, "loaded_cached_probability_maps"
    if model is None:
        return None, np.nan, "clean probability map ve checkpoint bulunamadi"
    records, inference_time = run_inference_records(model, spec, test_df, "clean_png", model_output_dir(spec.name) / "test_prob_maps")
    return records, inference_time, "inferred_from_checkpoint"


def load_cached_degradation_records(spec: ModelSpec, degradation: str) -> Optional[List[Dict[str, Any]]]:
    cache_path = model_output_dir(spec.name) / "robustness" / f"{degradation}_prob_maps.npz"
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        probs = data["probs"].astype(np.float32) if "probs" in data else data["arr_0"].astype(np.float32)
    except Exception as exc:
        print(f"[warn] Cache okunamadi, yeniden inference yapilacak: {cache_path} ({exc})")
        return None
    if len(probs) != len(test_df):
        print(f"[warn] Cache uzunlugu split ile eslesmedi, yeniden inference yapilacak: {cache_path}")
        return None
    records = make_records_from_df(test_df, spec.image_size)
    for i, rec in enumerate(records):
        rec["prob"] = probs[i]
    return records


# %% [markdown]
# ## 8. Degradation Functions: JPEG, Blur, Noise

# %%
def jpeg_roundtrip_rgb(image: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode basarisiz.")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def add_gaussian_noise(image: np.ndarray, std: float, seed_offset: int = 0) -> np.ndarray:
    rng = np.random.default_rng(CFG.seed + seed_offset)
    noise = rng.normal(0.0, std * 255.0, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_degradation(image: np.ndarray, degradation: str) -> np.ndarray:
    if degradation == "clean_png":
        return image
    if degradation == "jpeg_q90":
        return jpeg_roundtrip_rgb(image, 90)
    if degradation == "jpeg_q70":
        return jpeg_roundtrip_rgb(image, 70)
    if degradation == "jpeg_q50":
        return jpeg_roundtrip_rgb(image, 50)
    if degradation == "gaussian_blur_light":
        return cv2.GaussianBlur(image, (3, 3), sigmaX=0.8)
    if degradation == "gaussian_blur_medium":
        return cv2.GaussianBlur(image, (5, 5), sigmaX=1.2)
    if degradation == "gaussian_noise_light":
        return add_gaussian_noise(image, 0.02, seed_offset=1)
    if degradation == "gaussian_noise_medium":
        return add_gaussian_noise(image, 0.05, seed_offset=2)
    if degradation == "combined_jpeg70_blur_light":
        return cv2.GaussianBlur(jpeg_roundtrip_rgb(image, 70), (3, 3), sigmaX=0.8)
    raise ValueError(f"Bilinmeyen degradation: {degradation}")


# %% [markdown]
# ## 9. Post-processing Functions

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
                "area": int(area),
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
    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    out = mask.astype(np.uint8)
    if morphology in ("open", "open_close"):
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if morphology in ("close", "open_close"):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out.astype(np.uint8)


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config)
    numeric_defaults = {
        "pixel_threshold": 0.5,
        "image_threshold": 0.5,
        "min_component_area": 0,
        "min_component_mean_probability": 0.0,
        "morph_kernel_size": 3,
    }
    for key, default in numeric_defaults.items():
        value = cfg.get(key, default)
        if pd.isna(value) or value == "":
            value = default
        cfg[key] = type(default)(float(value)) if isinstance(default, float) else type(default)(int(float(value)))
    for key, default in {
        "postprocess_mode": "raw",
        "morphology": "none",
        "image_score_type": "max_probability",
        "top_k_sort_by": "area",
    }.items():
        value = cfg.get(key, default)
        cfg[key] = default if pd.isna(value) or value == "" else str(value)
    top_k = cfg.get("top_k_components", None)
    cfg["top_k_components"] = None if top_k is None or pd.isna(top_k) or top_k == "" else int(float(top_k))
    return cfg


def postprocess_probability_map(prob: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = sanitize_config(config)
    threshold = float(cfg.get("pixel_threshold", 0.5))
    raw = (prob >= threshold).astype(np.uint8)
    work = raw.copy()
    mode = str(cfg.get("postprocess_mode", "raw"))
    min_area = int(cfg.get("min_component_area", 0) or 0)
    min_mean_prob = float(cfg.get("min_component_mean_probability", 0.0) or 0.0)
    morphology = str(cfg.get("morphology", "none"))
    kernel_size = int(cfg.get("morph_kernel_size", 3) or 3)
    top_k = cfg.get("top_k_components", None)
    sort_by = str(cfg.get("top_k_sort_by", "area"))

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

    if mode == "keep_topk_components" and top_k is not None:
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


def apply_config(records: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = sanitize_config(config)
    preds, raw_preds, scores, comps_per_image = [], [], [], []
    for rec in records:
        out = postprocess_probability_map(rec["prob"], cfg)
        pred = out["mask"].astype(np.uint8)
        raw = out["raw_mask"].astype(np.uint8)
        comps = out["components"]
        preds.append(pred)
        raw_preds.append(raw)
        comps_per_image.append(comps)
        scores.append(image_score_from_outputs(rec["prob"], raw, pred, comps, cfg["image_score_type"]))
    return {"preds": preds, "raw_preds": raw_preds, "scores": np.asarray(scores, dtype=np.float32), "components_per_image": comps_per_image}


# %% [markdown]
# ## 10. Pixel Metrics

# %%
def safe_auc(y_true: np.ndarray, y_score: np.ndarray, kind: str) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        if kind == "roc":
            return float(roc_auc_score(y_true, y_score))
        if kind == "pr":
            return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")
    return float("nan")


def binary_dice_np(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    return float((2 * inter + EPS) / (denom + EPS))


def binary_iou_np(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float((inter + EPS) / (union + EPS))


def pixel_metrics_from_records(records: List[Dict[str, Any]], preds: List[np.ndarray], probs: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
    gt_all = np.concatenate([rec["mask"].reshape(-1).astype(np.uint8) for rec in records])
    pred_all = np.concatenate([pred.reshape(-1).astype(np.uint8) for pred in preds])
    tp = int(((gt_all == 1) & (pred_all == 1)).sum())
    fp = int(((gt_all == 0) & (pred_all == 1)).sum())
    fn = int(((gt_all == 1) & (pred_all == 0)).sum())
    tn = int(((gt_all == 0) & (pred_all == 0)).sum())
    out = {
        "dice_all": float((2 * tp + EPS) / (2 * tp + fp + fn + EPS)),
        "iou_all": float((tp + EPS) / (tp + fp + fn + EPS)),
        "precision": float((tp + EPS) / (tp + fp + EPS)),
        "recall": float((tp + EPS) / (tp + fn + EPS)),
        "specificity": float((tn + EPS) / (tn + fp + EPS)),
        "predicted_positive_pixel_ratio": float(pred_all.mean()),
        "gt_positive_pixel_ratio": float(gt_all.mean()),
    }
    forged_idx = [i for i, r in enumerate(records) if int(r["image_label"]) == 1]
    if forged_idx:
        gt_f = np.concatenate([records[i]["mask"].reshape(-1).astype(np.uint8) for i in forged_idx])
        pred_f = np.concatenate([preds[i].reshape(-1).astype(np.uint8) for i in forged_idx])
        tp_f = int(((gt_f == 1) & (pred_f == 1)).sum())
        fp_f = int(((gt_f == 0) & (pred_f == 1)).sum())
        fn_f = int(((gt_f == 1) & (pred_f == 0)).sum())
        out["dice_forged_only"] = float((2 * tp_f + EPS) / (2 * tp_f + fp_f + fn_f + EPS))
        out["iou_forged_only"] = float((tp_f + EPS) / (tp_f + fp_f + fn_f + EPS))
    else:
        out["dice_forged_only"] = 0.0
        out["iou_forged_only"] = 0.0
    if probs is not None:
        prob_all = np.concatenate([p.reshape(-1).astype(np.float32) for p in probs])
        out["roc_auc_all"] = safe_auc(gt_all, prob_all, "roc")
        out["auprc_all"] = safe_auc(gt_all, prob_all, "pr")
        if forged_idx:
            prob_f = np.concatenate([probs[i].reshape(-1).astype(np.float32) for i in forged_idx])
            out["auprc_forged_only"] = safe_auc(gt_f, prob_f, "pr")
    return out


def per_image_metrics(records: List[Dict[str, Any]], preds: List[np.ndarray], scores: Sequence[float], image_threshold: float) -> pd.DataFrame:
    rows = []
    for rec, pred, score in zip(records, preds, scores):
        gt = rec["mask"].astype(np.uint8)
        pred = pred.astype(np.uint8)
        rows.append(
            {
                "sample_id": str(rec.get("sample_id", f"{rec['class_name']}__{rec['image_id']}")),
                "image_id": str(rec["image_id"]),
                "image_path": str(rec["image_path"]),
                "class_name": str(rec["class_name"]),
                "image_label": int(rec["image_label"]),
                "gt_area": int(gt.sum()),
                "gt_area_ratio": float(gt.mean()),
                "mask_quartile": str(rec.get("mask_quartile", "")),
                "pred_area": int(pred.sum()),
                "pred_area_ratio": float(pred.mean()),
                "dice": binary_dice_np(gt, pred),
                "iou": binary_iou_np(gt, pred),
                "image_score": float(score),
                "image_pred_label": int(score >= image_threshold),
            }
        )
    return pd.DataFrame(rows)


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
            }
        )
    return pd.DataFrame(rows)


def summarize_per_image(per_df: pd.DataFrame) -> Dict[str, Any]:
    forged = per_df[per_df["image_label"].astype(int) == 1]
    out = {
        "mean_dice_forged": float(forged["dice"].mean()) if len(forged) else 0.0,
        "median_dice_forged": float(forged["dice"].median()) if len(forged) else 0.0,
        "mean_iou_forged": float(forged["iou"].mean()) if len(forged) else 0.0,
        "median_iou_forged": float(forged["iou"].median()) if len(forged) else 0.0,
        "dice_lt_005_count": int((forged["dice"] < 0.05).sum()) if len(forged) else 0,
    }
    small_df = small_mask_metrics(per_df)
    for row in small_df.itertuples():
        q = str(row.mask_quartile).lower()
        out[f"{q}_dice"] = float(row.mean_dice)
        out[f"{q}_iou"] = float(row.mean_iou)
        out[f"{q}_dice_lt_005_count"] = int(row.dice_lt_005_count)
    return out


# %% [markdown]
# ## 11. Component-aware Metrics

# %%
def suffix_for_iou(iou_threshold: float) -> str:
    return f"{int(round(float(iou_threshold) * 100)):03d}"


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


def component_metrics(records: List[Dict[str, Any]], preds: List[np.ndarray], iou_threshold: float) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows, tp, fp, fn = [], 0, 0, 0
    for rec, pred in zip(records, preds):
        comp = match_components(rec["mask"].astype(np.uint8), pred.astype(np.uint8), iou_threshold)
        row = {
            "image_id": str(rec["image_id"]),
            "image_label": int(rec["image_label"]),
            "mask_quartile": str(rec.get("mask_quartile", "")),
            **comp,
        }
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
    counts, areas, alarm = [], [], 0
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


# %% [markdown]
# ## 12. Image-level Metrics and Calibration

# %%
def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.clip(np.asarray(y_score).astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_score >= lo) & (y_score < hi if i < n_bins - 1 else y_score <= hi)
        count = int(mask.sum())
        if count:
            avg_conf = float(y_score[mask].mean())
            avg_acc = float(y_true[mask].mean())
            gap = abs(avg_acc - avg_conf)
            ece += (count / max(n, 1)) * gap
        else:
            avg_conf, avg_acc, gap = np.nan, np.nan, np.nan
        rows.append({"bin": i, "lower": lo, "upper": hi, "count": count, "avg_confidence": avg_conf, "accuracy": avg_acc, "gap": gap})
    return float(ece), pd.DataFrame(rows)


def image_level_metrics(y_true: np.ndarray, y_score: np.ndarray, image_threshold: float) -> Tuple[Dict[str, Any], pd.DataFrame]:
    y_pred = (np.asarray(y_score) >= float(image_threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ece, reliability = expected_calibration_error(y_true, y_score, n_bins=10)
    metrics = {
        "image_accuracy": float(accuracy_score(y_true, y_pred)),
        "image_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "image_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "image_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "image_specificity": float(tn / max(tn + fp, 1)),
        "image_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "image_roc_auc": safe_auc(y_true, y_score, "roc"),
        "image_auprc": safe_auc(y_true, y_score, "pr"),
        "image_brier": float(brier_score_loss(y_true, np.clip(y_score, 0, 1))) if len(np.unique(y_true)) > 1 else float("nan"),
        "image_ece_10bin": ece,
        "image_tp": int(tp),
        "image_fn": int(fn),
        "image_tn": int(tn),
        "image_fp": int(fp),
    }
    return metrics, reliability


def evaluate_records(records: List[Dict[str, Any]], config: Dict[str, Any], inference_time_per_image: float = np.nan) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cfg = sanitize_config(config)
    outputs = apply_config(records, cfg)
    y_true = np.array([int(r["image_label"]) for r in records], dtype=int)
    image_threshold = float(cfg.get("image_threshold", 0.5))
    per_df = per_image_metrics(records, outputs["preds"], outputs["scores"], image_threshold)
    small_df = small_mask_metrics(per_df)
    pix = pixel_metrics_from_records(records, outputs["preds"], probs=[r["prob"] for r in records])
    per_summary = summarize_per_image(per_df)
    img_m, reliability_df = image_level_metrics(y_true, outputs["scores"], image_threshold)
    comp_all = {}
    comp_details = pd.DataFrame()
    for thr in CFG.component_iou_thresholds:
        comp_m, comp_df = component_metrics(records, outputs["preds"], float(thr))
        comp_all.update(comp_m)
        if abs(float(thr) - 0.10) < 1e-6:
            comp_details = comp_df
    auth_m = authentic_false_alarm_metrics(records, outputs["preds"], outputs["components_per_image"])
    metrics = {**pix, **per_summary, **img_m, **comp_all, **auth_m}
    metrics["inference_time_per_image"] = float(inference_time_per_image) if pd.notna(inference_time_per_image) else np.nan
    return metrics, per_df, small_df, comp_details, reliability_df, outputs


# %% [markdown]
# ## 13. Clean Test Evaluation

# %%
def load_selected_strategy(model_name: str, preferred_strategy: str) -> Optional[Dict[str, Any]]:
    src = source_model_dir(model_name)
    if src is None:
        return None
    path = src / "selected_configs.csv"
    if not path.exists():
        print(f"[warn] selected_configs.csv bulunamadi: {path}")
        return None
    df = pd.read_csv(path)
    if preferred_strategy == "balanced_or_low_false_alarm":
        choices = ["low_false_alarm", "balanced_final_score", "best_component_f1"]
    else:
        choices = [preferred_strategy, "balanced_final_score", "best_forged_dice"]
    for choice in choices:
        part = df[df["strategy"].astype(str) == choice]
        if len(part):
            return part.iloc[0].to_dict()
    return df.iloc[0].to_dict() if len(df) else None


def write_clean_outputs(model_name: str, strategy: str, metrics: Dict[str, Any], per_df: pd.DataFrame, small_df: pd.DataFrame, comp_df: pd.DataFrame, reliability_df: pd.DataFrame) -> None:
    out_dir = model_output_dir(model_name)
    pd.DataFrame([{**metrics, "model_name": model_name, "strategy": strategy}]).to_csv(out_dir / "clean_test_metrics.csv", index=False)
    per_df.to_csv(out_dir / f"test_per_image_metrics_{strategy}.csv", index=False)
    small_df.to_csv(out_dir / f"small_mask_bin_metrics_{strategy}.csv", index=False)
    comp_df.to_csv(out_dir / f"test_component_details_{strategy}.csv", index=False)
    reliability_df.to_csv(out_dir / f"image_level_calibration_{strategy}.csv", index=False)


model_runtime_info = []
clean_results_rows = []
clean_records_by_model: Dict[str, List[Dict[str, Any]]] = {}
clean_outputs_by_model: Dict[str, Dict[str, Any]] = {}
selected_configs_by_model: Dict[str, Dict[str, Any]] = {}
loaded_models: Dict[str, Optional[nn.Module]] = {}

for spec in FINAL_MODELS:
    print(f"\n[clean] {spec.name}")
    out_dir = model_output_dir(spec.name)
    selected = load_selected_strategy(spec.name, spec.preferred_strategy)
    if selected is None:
        print(f"[warn] {spec.name}: selected config yok, model atlandi.")
        continue
    selected_configs_by_model[spec.name] = selected
    model, info = load_checkpoint_model(spec)
    loaded_models[spec.name] = model
    model_runtime_info.append(info)
    records, inference_time, status = get_clean_records(model, spec)
    if records is None:
        print(f"[warn] {spec.name}: clean degerlendirme yapilamadi: {status}")
        continue
    metrics, per_df, small_df, comp_df, reliability_df, outputs = evaluate_records(records, selected, inference_time)
    strategy = str(selected.get("strategy", spec.preferred_strategy))
    row = {"model_name": spec.name, "role": spec.role, "image_size": spec.image_size, "strategy": strategy, "clean_status": status, **sanitize_config(selected), **metrics}
    clean_results_rows.append(row)
    clean_records_by_model[spec.name] = records
    clean_outputs_by_model[spec.name] = {"metrics": metrics, "per_df": per_df, "small_df": small_df, "comp_df": comp_df, "reliability_df": reliability_df, "outputs": outputs}
    write_clean_outputs(spec.name, strategy, metrics, per_df, small_df, comp_df, reliability_df)
    print(pd.DataFrame([row])[["model_name", "strategy", "dice_forged_only", "q1_dice", "component_f1_iou010", "authentic_fp_rate", "image_f1"]])

pd.DataFrame(model_runtime_info).to_csv(FINAL_ROOT / "model_runtime_info.csv", index=False)
clean_results_df = pd.DataFrame(clean_results_rows)
clean_results_df.to_csv(FINAL_ROOT / "clean_final_candidate_results.csv", index=False)


# %% [markdown]
# ## 14. Robustness Evaluation

# %%
def robustness_metric_row(model_name: str, degradation: str, metrics: Dict[str, Any], inference_time: float) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "degradation": degradation,
        "forged_dice": metrics.get("dice_forged_only", np.nan),
        "forged_iou": metrics.get("iou_forged_only", np.nan),
        "q1_dice": metrics.get("q1_dice", np.nan),
        "q2_dice": metrics.get("q2_dice", np.nan),
        "q3_dice": metrics.get("q3_dice", np.nan),
        "q4_dice": metrics.get("q4_dice", np.nan),
        "dice_lt_005_count": metrics.get("dice_lt_005_count", np.nan),
        "component_f1_iou010": metrics.get("component_f1_iou010", np.nan),
        "component_f1_iou025": metrics.get("component_f1_iou025", np.nan),
        "component_f1_iou050": metrics.get("component_f1_iou050", np.nan),
        "authentic_fp_rate": metrics.get("authentic_fp_rate", np.nan),
        "image_f1": metrics.get("image_f1", np.nan),
        "image_roc_auc": metrics.get("image_roc_auc", np.nan),
        "image_auprc": metrics.get("image_auprc", np.nan),
        "image_specificity": metrics.get("image_specificity", np.nan),
        "image_recall": metrics.get("image_recall", np.nan),
        "inference_time_per_image": inference_time,
    }


robustness_rows = []
robustness_errors = []
existing_robustness_path = FINAL_ROOT / "robustness_metrics_all.csv"
existing_robustness_delta_path = FINAL_ROOT / "robustness_delta_from_clean.csv"

if CFG.reuse_existing_robustness_metrics and existing_robustness_path.exists():
    robustness_df = pd.read_csv(existing_robustness_path)
    print(f"[cache] Existing robustness metrics kullanildi: {existing_robustness_path}")
    for spec in FINAL_MODELS:
        model_metrics = robustness_df[robustness_df["model_name"].astype(str) == spec.name]
        if not model_metrics.empty:
            model_metrics.to_csv(model_output_dir(spec.name) / "robustness_metrics.csv", index=False)
        if spec.name in clean_outputs_by_model:
            clean_outputs_by_model[spec.name]["per_df"].to_csv(model_output_dir(spec.name) / "robustness_per_image_clean_png.csv", index=False)
else:
    for spec in FINAL_MODELS:
        selected = selected_configs_by_model.get(spec.name)
        model = loaded_models.get(spec.name)
        if selected is None:
            continue
        model_robust_rows = []
        if spec.name in clean_outputs_by_model:
            clean_metrics = clean_outputs_by_model[spec.name]["metrics"]
            clean_time = clean_metrics.get("inference_time_per_image", np.nan)
            clean_outputs_by_model[spec.name]["per_df"].to_csv(model_output_dir(spec.name) / "robustness_per_image_clean_png.csv", index=False)
            row = robustness_metric_row(spec.name, "clean_png", clean_metrics, clean_time)
            robustness_rows.append(row)
            model_robust_rows.append(row)
        if model is None:
            robustness_errors.append({"model_name": spec.name, "message": "checkpoint bulunamadigi icin robustness calistirilamadi"})
            print(f"[warn] {spec.name}: checkpoint yok, robustness atlandi.")
            pd.DataFrame(model_robust_rows).to_csv(model_output_dir(spec.name) / "robustness_metrics.csv", index=False)
            continue
        degradations = [d for d in CFG.degradation_order if d != "clean_png"]
        if not CFG.robustness_include_combined:
            degradations = [d for d in degradations if not d.startswith("combined_")]
        if not CFG.robustness_enabled:
            degradations = []
        for degradation in degradations:
            try:
                cached_records = load_cached_degradation_records(spec, degradation)
                if cached_records is not None:
                    records, inference_time = cached_records, np.nan
                    print(f"[cache] {spec.name} {degradation}: cached probability map kullanildi.")
                else:
                    records, inference_time = run_inference_records(model, spec, test_df, degradation, model_output_dir(spec.name) / "robustness" / f"{degradation}_prob_maps")
                metrics, per_df, small_df, comp_df, reliability_df, _ = evaluate_records(records, selected, inference_time)
                per_df.to_csv(model_output_dir(spec.name) / f"robustness_per_image_{degradation}.csv", index=False)
                small_df.to_csv(model_output_dir(spec.name) / "robustness" / f"small_mask_bins_{degradation}.csv", index=False)
                comp_df.to_csv(model_output_dir(spec.name) / "robustness" / f"component_details_{degradation}.csv", index=False)
                reliability_df.to_csv(model_output_dir(spec.name) / "robustness" / f"calibration_{degradation}.csv", index=False)
                row = robustness_metric_row(spec.name, degradation, metrics, inference_time)
                robustness_rows.append(row)
                model_robust_rows.append(row)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                robustness_errors.append({"model_name": spec.name, "degradation": degradation, "message": msg})
                print(f"[warn] robustness hata: {spec.name} {degradation}: {msg}")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
        pd.DataFrame(model_robust_rows).to_csv(model_output_dir(spec.name) / "robustness_metrics.csv", index=False)

    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df.to_csv(FINAL_ROOT / "robustness_metrics_all.csv", index=False)
    pd.DataFrame(robustness_errors).to_csv(FINAL_ROOT / "robustness_errors.csv", index=False)

delta_rows = []
if CFG.reuse_existing_robustness_metrics and existing_robustness_delta_path.exists():
    robustness_delta_df = pd.read_csv(existing_robustness_delta_path)
    print(f"[cache] Existing robustness delta kullanildi: {existing_robustness_delta_path}")
elif not robustness_df.empty:
    for model_name, part in robustness_df.groupby("model_name"):
        clean = part[part["degradation"] == "clean_png"]
        if clean.empty:
            continue
        base = clean.iloc[0]
        for _, row in part.iterrows():
            out = {"model_name": model_name, "degradation": row["degradation"]}
            for metric in ["forged_dice", "q1_dice", "component_f1_iou010", "authentic_fp_rate", "image_f1"]:
                out[f"delta_{metric}"] = float(row.get(metric, np.nan)) - float(base.get(metric, np.nan))
            delta_rows.append(out)
    robustness_delta_df = pd.DataFrame(delta_rows)
    robustness_delta_df.to_csv(FINAL_ROOT / "robustness_delta_from_clean.csv", index=False)
else:
    robustness_delta_df = pd.DataFrame(delta_rows)
    robustness_delta_df.to_csv(FINAL_ROOT / "robustness_delta_from_clean.csv", index=False)


# %% [markdown]
# ## 15. Final Model Comparison Table

# %%
def normalize_result_row(row: Dict[str, Any], display_name: str, role: str, image_size: int, interpretation: str) -> Dict[str, Any]:
    aliases = {
        "forged_dice": ["dice_forged_only", "forged_dice"],
        "forged_iou": ["iou_forged_only", "forged_iou"],
        "brier_score": ["image_brier", "brier_score"],
        "ece_10_bins": ["image_ece_10bin", "ece_10_bins"],
        "inference_time_per_image": ["inference_time_per_image", "inference_time_per_image_sec"],
        "trainable_params": ["trainable_params", "trainable_parameters"],
    }
    def get_value(name: str, default=np.nan):
        for key in aliases.get(name, [name]):
            if key in row and pd.notna(row[key]):
                return row[key]
        return row.get(name, default)
    out = {
        "model_name": display_name,
        "source_model_name": row.get("model_name", display_name),
        "role": role,
        "image_size": image_size,
        "strategy": row.get("strategy", ""),
        "forged_dice": get_value("forged_dice"),
        "forged_iou": get_value("forged_iou"),
        "q1_dice": row.get("q1_dice", np.nan),
        "q2_dice": row.get("q2_dice", np.nan),
        "q3_dice": row.get("q3_dice", np.nan),
        "q4_dice": row.get("q4_dice", np.nan),
        "dice_lt_005_count": row.get("dice_lt_005_count", np.nan),
        "component_f1_iou010": row.get("component_f1_iou010", np.nan),
        "component_f1_iou025": row.get("component_f1_iou025", np.nan),
        "component_f1_iou050": row.get("component_f1_iou050", np.nan),
        "authentic_fp_rate": row.get("authentic_fp_rate", np.nan),
        "image_f1": row.get("image_f1", np.nan),
        "image_roc_auc": row.get("image_roc_auc", np.nan),
        "image_auprc": row.get("image_auprc", np.nan),
        "brier_score": get_value("brier_score"),
        "ece_10_bins": get_value("ece_10_bins"),
        "inference_time_per_image": get_value("inference_time_per_image"),
        "trainable_params": get_value("trainable_params"),
        "final_interpretation": interpretation,
    }
    return out


def load_exp5_all_results() -> pd.DataFrame:
    candidates = []
    if EXP5_ROOT is not None:
        candidates.extend([EXP5_ROOT / "test_results_all_strategies.csv", EXP5_ROOT / "experiment5_all_results.csv"])
    candidates.extend(
        [
            Path("deney_5/experiments_4_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv"),
            Path("/content/drive/MyDrive/bitirmeProjesi/experiments_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv"),
            Path("/kaggle/working/experiments_full/experiment5_calibration_postprocessing/test_results_all_strategies.csv"),
        ]
    )
    frames = [pd.read_csv(p) for p in candidates if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def choose_reference_row(results: pd.DataFrame, model_name: str, preferred_strategy: str) -> Optional[pd.Series]:
    if results.empty or "model_name" not in results.columns:
        return None
    part = results[results["model_name"].astype(str) == model_name].copy()
    if part.empty:
        return None
    strategies = [preferred_strategy, "balanced_final_score", "raw_reference", "best_forged_dice", "best_component_f1", "low_false_alarm"]
    for strategy in strategies:
        sub = part[part["strategy"].astype(str) == strategy] if "strategy" in part.columns else pd.DataFrame()
        if len(sub):
            return sub.iloc[0]
    if "final_score" in part.columns:
        return part.sort_values("final_score", ascending=False).iloc[0]
    return part.iloc[0]


def find_model_artifact_csv(model_name: str, filename: str) -> Optional[Path]:
    candidates = []
    for root in [EXP5_ROOT, EXP6_ROOT, EXP4_ROOT]:
        if root is not None:
            candidates.append(root / model_name / filename)
    candidates.extend(
        [
            Path("deney_5/experiments_4_full/experiment5_calibration_postprocessing") / model_name / filename,
            Path("deney_5/experiments_4_full") / model_name / filename,
            Path("deney_6/experiments_full/experiment6_smallmask_384") / model_name / filename,
            Path("experiments_full/experiment5_calibration_postprocessing") / model_name / filename,
            Path("experiments_full/experiment6_smallmask_384") / model_name / filename,
        ]
    )
    return first_existing(candidates)


def enrich_row_with_small_bins(row: Dict[str, Any], model_name: str, strategy: str) -> Dict[str, Any]:
    out = dict(row)
    path = find_model_artifact_csv(model_name, f"small_mask_bin_metrics_{strategy}.csv")
    if path is None:
        return out
    try:
        bins = pd.read_csv(path)
    except Exception:
        return out
    if "mask_quartile" not in bins.columns:
        return out
    for _, b in bins.iterrows():
        q = str(b.get("mask_quartile", "")).lower()
        if q in ["q1", "q2", "q3", "q4"]:
            if "mean_dice" in bins.columns:
                out[f"{q}_dice"] = b.get("mean_dice", out.get(f"{q}_dice", np.nan))
            if "mean_iou" in bins.columns:
                out[f"{q}_iou"] = b.get("mean_iou", out.get(f"{q}_iou", np.nan))
            if "dice_lt_005_count" in bins.columns:
                out[f"{q}_dice_lt_005_count"] = b.get("dice_lt_005_count", out.get(f"{q}_dice_lt_005_count", np.nan))
    return out


exp5_results_df = load_exp5_all_results()
comparison_rows = []

for ref in REFERENCE_MODELS:
    row = choose_reference_row(exp5_results_df, ref["model_name"], ref["preferred_strategy"])
    if row is None:
        print(f"[warn] Referans sonuc bulunamadi: {ref['model_name']}")
        continue
    row_dict = enrich_row_with_small_bins(row.to_dict(), ref["model_name"], str(row.get("strategy", ref["preferred_strategy"])))
    interpretation = "256x256 referans/baseline; yeniden egitilmedi, Deney 5/4 CSV sonucundan okundu."
    comparison_rows.append(normalize_result_row(row_dict, ref["display_name"], ref["role"], int(ref["image_size"]), interpretation))

for spec in FINAL_MODELS:
    part = clean_results_df[clean_results_df["model_name"].astype(str) == spec.name] if not clean_results_df.empty else pd.DataFrame()
    if part.empty:
        continue
    display = "SegFormer-B0 384" if spec.name.startswith("segformer") else "EfficientNetB0-UNet 384"
    interpretation = (
        "Localization-oriented final model; pixel/component lokalizasyonu onceliklidir."
        if spec.name.startswith("segformer")
        else "Conservative low-false-alarm final model; pratik kullanimda yanlis alarm maliyeti dusunulerek raporlanir."
    )
    comparison_rows.append(normalize_result_row(part.iloc[0].to_dict(), display, spec.role, spec.image_size, interpretation))

final_comparison_df = pd.DataFrame(comparison_rows)
if not final_comparison_df.empty:
    sort_cols = ["forged_dice", "component_f1_iou010", "q1_dice", "authentic_fp_rate", "image_f1"]
    for col in sort_cols:
        final_comparison_df[col] = pd.to_numeric(final_comparison_df[col], errors="coerce")
    final_comparison_df = final_comparison_df.sort_values(
        by=["forged_dice", "component_f1_iou010", "q1_dice", "authentic_fp_rate", "image_f1"],
        ascending=[False, False, False, True, False],
        na_position="last",
    )
final_comparison_df.to_csv(FINAL_ROOT / "final_model_comparison.csv", index=False)
final_comparison_df.to_csv(FINAL_ROOT / "tables" / "final_model_comparison.csv", index=False)
with open(FINAL_ROOT / "final_model_comparison.md", "w", encoding="utf-8") as f:
    f.write("# Final Model Karsilastirma Tablosu\n\n")
    f.write("Threshold ve post-processing ayarlari validation setinde secilmis, testte sabit tutulmustur.\n\n")
    f.write(final_comparison_df.to_markdown(index=False) if not final_comparison_df.empty else "_Sonuc bulunamadi._")


# %% [markdown]
# ## 16. Statistical Tests

# %%
def find_per_image_csv(model_name: str, strategy: str) -> Optional[Path]:
    candidates = [
        model_output_dir(model_name) / f"test_per_image_metrics_{strategy}.csv",
    ]
    if EXP5_ROOT is not None:
        candidates.append(EXP5_ROOT / model_name / f"test_per_image_metrics_{strategy}.csv")
    if EXP6_ROOT is not None:
        candidates.append(EXP6_ROOT / model_name / f"test_per_image_metrics_{strategy}.csv")
    candidates.extend(
        [
            Path("deney_5/experiments_4_full/experiment5_calibration_postprocessing") / model_name / f"test_per_image_metrics_{strategy}.csv",
            Path("deney_6/experiments_full/experiment6_smallmask_384") / model_name / f"test_per_image_metrics_{strategy}.csv",
        ]
    )
    return first_existing(candidates)


def load_per_image_for_model(model_name: str, strategy: str) -> Optional[pd.DataFrame]:
    path = find_per_image_csv(model_name, strategy)
    if path is None:
        return None
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str)
    df = enrich_per_image_with_test_metadata(df, model_name, strategy)
    return df


def enrich_per_image_with_test_metadata(df: pd.DataFrame, model_name: str, strategy: str) -> pd.DataFrame:
    """Eski Deney 5 per-image CSV'lerinde eksik olan mask_quartile/class metadata'sini ekler."""
    out = df.copy()
    required = ["mask_quartile", "class_name", "gt_area", "gt_area_ratio"]
    if all(col in out.columns for col in required):
        return out

    meta_cols = ["image_id", "class_name", "image_label", "gt_area", "gt_area_ratio", "mask_quartile", "image_path"]
    meta = test_df[[c for c in meta_cols if c in test_df.columns]].copy()
    meta["image_id"] = meta["image_id"].astype(str)

    join_cols = ["image_id"]
    if "class_name" in out.columns and "class_name" in meta.columns:
        out["class_name"] = out["class_name"].astype(str)
        meta["class_name"] = meta["class_name"].astype(str)
        join_cols = ["image_id", "class_name"]

    fill_cols = [c for c in meta.columns if c not in join_cols and c not in out.columns]
    if not fill_cols:
        return out

    meta_small = meta[join_cols + fill_cols].drop_duplicates(subset=join_cols)
    enriched = out.merge(meta_small, on=join_cols, how="left", validate="many_to_one")
    if "mask_quartile" not in enriched.columns:
        print(f"[warn] {model_name}/{strategy}: mask_quartile eklenemedi; Q1/Q2 istatistikleri atlanabilir.")
    return enriched


def paired_arrays(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, subset: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    a = df_a.copy()
    b = df_b.copy()
    if subset in ["Q1", "Q2", "Q3", "Q4"]:
        if "mask_quartile" not in a.columns or "mask_quartile" not in b.columns:
            return np.array([], dtype=float), np.array([], dtype=float), pd.DataFrame()
        a = a[(a["image_label"].astype(int) == 1) & (a["mask_quartile"].astype(str) == subset)]
        b = b[(b["image_label"].astype(int) == 1) & (b["mask_quartile"].astype(str) == subset)]
    elif subset == "forged":
        a = a[a["image_label"].astype(int) == 1]
        b = b[b["image_label"].astype(int) == 1]
    merged = a[["image_id", metric]].merge(b[["image_id", metric]], on="image_id", suffixes=("_a", "_b"))
    return merged[f"{metric}_a"].astype(float).to_numpy(), merged[f"{metric}_b"].astype(float).to_numpy(), merged


def bootstrap_mean_diff_ci(diff: np.ndarray, iterations: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(iterations)]
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_biserial_from_wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    n = len(diff)
    if n == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    return float((pos - neg) / (n * (n + 1) / 2))


def run_paired_tests(name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame, metric: str, subset: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    a, b, merged = paired_arrays(df_a, df_b, metric, subset=subset)
    diff = a - b
    diff = diff[np.isfinite(diff)]
    label = subset or "all"
    if len(diff) < 2:
        return (
            {"comparison": f"{name_a} vs {name_b}", "metric": metric, "subset": label, "n": len(diff), "mean_diff": np.nan, "paired_t_p": np.nan, "wilcoxon_p": np.nan, "cohens_d": np.nan, "rank_biserial": np.nan},
            {"comparison": f"{name_a} vs {name_b}", "metric": metric, "subset": label, "n": len(diff), "mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan},
        )
    t_stat, t_p = stats.ttest_rel(a[: len(diff)], b[: len(diff)], nan_policy="omit")
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except Exception:
        w_stat, w_p = np.nan, np.nan
    mean_diff, ci_low, ci_high = bootstrap_mean_diff_ci(diff, CFG.bootstrap_iterations, CFG.seed)
    cohens_d = float(np.mean(diff) / (np.std(diff, ddof=1) + EPS))
    rb = rank_biserial_from_wilcoxon(diff)
    return (
        {
            "comparison": f"{name_a} vs {name_b}",
            "metric": metric,
            "subset": label,
            "n": int(len(diff)),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(mean_diff),
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
            "wilcoxon_stat": float(w_stat) if pd.notna(w_stat) else np.nan,
            "wilcoxon_p": float(w_p) if pd.notna(w_p) else np.nan,
            "cohens_d": cohens_d,
            "rank_biserial": rb,
        },
        {"comparison": f"{name_a} vs {name_b}", "metric": metric, "subset": label, "n": int(len(diff)), "mean_diff": mean_diff, "ci_low": ci_low, "ci_high": ci_high},
    )


def mcnemar_test(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> Dict[str, Any]:
    cols = ["image_id", "image_label", "image_pred_label"]
    if not all(c in df_a.columns for c in cols) or not all(c in df_b.columns for c in cols):
        return {"comparison": f"{name_a} vs {name_b}", "metric": "image_correctness", "n": 0, "mcnemar_p": np.nan}
    merged = df_a[cols].merge(df_b[cols], on="image_id", suffixes=("_a", "_b"))
    correct_a = merged["image_pred_label_a"].astype(int).to_numpy() == merged["image_label_a"].astype(int).to_numpy()
    correct_b = merged["image_pred_label_b"].astype(int).to_numpy() == merged["image_label_b"].astype(int).to_numpy()
    b01 = int(((correct_a == True) & (correct_b == False)).sum())
    b10 = int(((correct_a == False) & (correct_b == True)).sum())
    n = b01 + b10
    p = float(stats.binomtest(min(b01, b10), n=n, p=0.5).pvalue) if n > 0 else np.nan
    return {"comparison": f"{name_a} vs {name_b}", "metric": "image_correctness", "n": int(len(merged)), "a_correct_b_wrong": b01, "a_wrong_b_correct": b10, "mcnemar_p": min(p, 1.0) if pd.notna(p) else np.nan}


per_image_registry: Dict[str, pd.DataFrame] = {}
strategy_registry: Dict[str, str] = {}
for spec in FINAL_MODELS:
    strategy_registry[spec.name] = str(selected_configs_by_model.get(spec.name, {}).get("strategy", spec.preferred_strategy))
    df = load_per_image_for_model(spec.name, strategy_registry[spec.name])
    if df is not None:
        per_image_registry[spec.name] = df
for ref in REFERENCE_MODELS:
    row = choose_reference_row(exp5_results_df, ref["model_name"], ref["preferred_strategy"])
    if row is not None:
        strategy = str(row.get("strategy", ref["preferred_strategy"]))
        strategy_registry[ref["model_name"]] = strategy
        df = load_per_image_for_model(ref["model_name"], strategy)
        if df is not None:
            per_image_registry[ref["model_name"]] = df

stat_comparisons = [
    ("segformer_b0_rgb_384_smallmask", "efficientnetb0_unet_rgb_384_smallmask"),
    ("segformer_b0_rgb_384_smallmask", "segformer_b0_rgb_full"),
    ("efficientnetb0_unet_rgb_384_smallmask", "efficientnetb0_unet_rgb_full"),
    ("segformer_b0_rgb_384_smallmask", "unetpp_resnet34_rgb_full"),
    ("efficientnetb0_unet_rgb_384_smallmask", "unetpp_resnet34_rgb_full"),
]

stat_rows, ci_rows, mcnemar_rows = [], [], []
for a_name, b_name in stat_comparisons:
    if a_name not in per_image_registry or b_name not in per_image_registry:
        print(f"[warn] Istatistik icin per-image eksik: {a_name} vs {b_name}")
        continue
    for metric, subset in [("dice", "forged"), ("iou", "forged"), ("dice", "Q1"), ("dice", "Q2")]:
        row, ci = run_paired_tests(a_name, per_image_registry[a_name], b_name, per_image_registry[b_name], metric, subset=subset)
        stat_rows.append(row)
        ci_rows.append(ci)
    mcnemar_rows.append(mcnemar_test(per_image_registry[a_name], per_image_registry[b_name], a_name, b_name))

statistical_tests_df = pd.DataFrame(stat_rows)
bootstrap_ci_df = pd.DataFrame(ci_rows)
mcnemar_df = pd.DataFrame(mcnemar_rows)
statistical_tests_df.to_csv(FINAL_ROOT / "statistical_tests.csv", index=False)
bootstrap_ci_df.to_csv(FINAL_ROOT / "bootstrap_confidence_intervals.csv", index=False)
mcnemar_df.to_csv(FINAL_ROOT / "statistics" / "mcnemar_tests.csv", index=False)
statistical_tests_df.to_csv(FINAL_ROOT / "statistics" / "statistical_tests.csv", index=False)
bootstrap_ci_df.to_csv(FINAL_ROOT / "statistics" / "bootstrap_confidence_intervals.csv", index=False)


# %% [markdown]
# ## 17. Failure Case Analysis

# %%
def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    err = np.zeros((*gt.shape, 3), dtype=np.uint8)
    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)
    err[tp] = (60, 180, 75)
    err[fp] = (230, 70, 70)
    err[fn] = (65, 105, 225)
    return err


def resize_for_grid(arr: np.ndarray, size: int = 160, is_mask: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if arr.ndim == 2:
        return cv2.resize(arr, (size, size), interpolation=interp)
    return cv2.resize(arr, (size, size), interpolation=interp)


def save_failure_grid(records: List[Dict[str, Any]], outputs: Dict[str, Any], selected_indices: List[int], title: str, out_path: Path) -> None:
    if not selected_indices:
        return
    n = len(selected_indices)
    cols = 7
    fig, axes = plt.subplots(n, cols, figsize=(cols * 2.0, n * 1.9))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    headers = ["original", "gt mask", "prob map", "raw mask", "post mask", "overlay", "TP/FP/FN"]
    for r, idx in enumerate(selected_indices):
        rec = records[idx]
        image = load_image_rgb(rec["image_path"])
        image = cv2.resize(image, (rec["prob"].shape[1], rec["prob"].shape[0]), interpolation=cv2.INTER_AREA)
        is_authentic = int(rec.get("image_label", 0)) == 0 or str(rec.get("class_name", "")).lower() == "authentic"
        gt = np.zeros_like(rec["mask"], dtype=np.uint8) if is_authentic else rec["mask"].astype(np.uint8)
        prob = rec["prob"].astype(np.float32)
        raw = outputs["raw_preds"][idx].astype(np.uint8)
        pred = outputs["preds"][idx].astype(np.uint8)
        overlay = overlay_mask(image, pred, (255, 120, 0) if is_authentic else (255, 0, 0), 0.45)
        err = make_error_map(gt, pred)
        panels = [
            resize_for_grid(image),
            resize_for_grid(gt * 255, is_mask=True),
            resize_for_grid((prob * 255).clip(0, 255).astype(np.uint8)),
            resize_for_grid(raw * 255, is_mask=True),
            resize_for_grid(pred * 255, is_mask=True),
            resize_for_grid(overlay),
            resize_for_grid(err),
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            if panel.ndim == 2:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panel)
            ax.axis("off")
            if r == 0:
                ax.set_title(headers[c], fontsize=8)
        case_key = f"{rec['image_id']}|{rec.get('class_name', '')}"
        axes[r, 0].set_ylabel(f"{rec['image_id']}\n{rec.get('class_name', '')}\nD={per_df_lookup.get(case_key, np.nan):.3f}", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_failure_case_rows(model_name: str, records: List[Dict[str, Any]], outputs: Dict[str, Any], per_df: pd.DataFrame, comp_df: pd.DataFrame, indices: List[int], group_name: str) -> pd.DataFrame:
    comp_lookup = {}
    if not comp_df.empty:
        # image_id tek basina her zaman tekil olmayabilir; component satirlari records ile
        # ayni sirada uretildigi icin image_id+class_name anahtariyla guvenli hizaliyoruz.
        for rec_i, comp_row in zip(records, comp_df.reset_index(drop=True).to_dict("records")):
            key = f"{rec_i['image_id']}|{rec_i.get('class_name', '')}"
            comp_lookup[key] = comp_row
    rows = []
    per_rows = per_df.reset_index(drop=True).to_dict("records")
    for idx in indices:
        rec = records[idx]
        image_id = str(rec["image_id"])
        per = per_rows[idx] if idx < len(per_rows) else {}
        comp = comp_lookup.get(f"{image_id}|{rec.get('class_name', '')}", {})
        rows.append(
            {
                "image_id": image_id,
                "image_path": rec["image_path"],
                "class_name": rec["class_name"],
                "gt_area": per.get("gt_area", np.nan),
                "gt_area_ratio": per.get("gt_area_ratio", np.nan),
                "mask_quartile": per.get("mask_quartile", ""),
                "pred_area": per.get("pred_area", np.nan),
                "pred_area_ratio": per.get("pred_area_ratio", np.nan),
                "dice": per.get("dice", np.nan),
                "iou": per.get("iou", np.nan),
                "image_score": per.get("image_score", np.nan),
                "image_pred_label": per.get("image_pred_label", np.nan),
                "gt_component_count": comp.get("gt_component_count", np.nan),
                "pred_component_count": comp.get("pred_component_count", np.nan),
                "matched_component_count": comp.get("matched_component_count", np.nan),
                "failure_group": group_name,
                "model_name": model_name,
            }
        )
    return pd.DataFrame(rows)


failure_summary_rows = []
all_failure_case_rows = []

for spec in FINAL_MODELS:
    if spec.name not in clean_records_by_model or spec.name not in clean_outputs_by_model:
        continue
    records = clean_records_by_model[spec.name]
    pack = clean_outputs_by_model[spec.name]
    per_df = pack["per_df"].copy()
    comp_df = pack["comp_df"].copy()
    outputs = pack["outputs"]
    per_df["case_key"] = per_df["image_id"].astype(str) + "|" + per_df["class_name"].astype(str)
    case_key_to_idx = {f"{r['image_id']}|{r.get('class_name', '')}": i for i, r in enumerate(records)}
    per_df_lookup = per_df.set_index("case_key")["dice"].to_dict()
    if not comp_df.empty and len(comp_df) == len(per_df):
        per_df["pred_component_count_iou010"] = comp_df.reset_index(drop=True)["pred_component_count"].to_numpy()
    elif "pred_component_count_iou010" not in per_df.columns:
        per_df["pred_component_count_iou010"] = np.nan

    forged = per_df[per_df["image_label"].astype(int) == 1].copy()
    authentic = per_df[per_df["image_label"].astype(int) == 0].copy()
    groups: Dict[str, List[int]] = {}
    def indices_from_cases(frame: pd.DataFrame) -> List[int]:
        return [case_key_to_idx[str(key)] for key in frame["case_key"].astype(str).tolist() if str(key) in case_key_to_idx]

    groups["best_forged_cases"] = indices_from_cases(forged.sort_values("dice", ascending=False).head(CFG.max_visual_examples))
    groups["worst_forged_cases"] = indices_from_cases(forged.sort_values("dice", ascending=True).head(CFG.max_visual_examples))
    small_fail = forged[(forged["mask_quartile"].astype(str).isin(["Q1", "Q2"])) & (forged["dice"] < 0.05)].sort_values(["dice", "gt_area"])
    groups["small_mask_failures"] = indices_from_cases(small_fail.head(CFG.max_visual_examples))
    auth_case = authentic[
        (authentic["pred_component_count_iou010"].fillna(0) > 0) | (authentic["pred_area"].fillna(0) > 0)
    ].sort_values("image_score", ascending=False)
    groups["false_positive_authentic"] = indices_from_cases(auth_case.head(CFG.max_visual_examples))
    fn_case = forged[forged["image_pred_label"].astype(int) == 0].sort_values(["image_score", "dice"], ascending=[True, True])
    groups["false_negative_forged"] = indices_from_cases(fn_case.head(CFG.max_visual_examples))
    q4 = forged[forged["mask_quartile"].astype(str) == "Q4"]
    groups["large_mask_success"] = indices_from_cases(q4.sort_values("dice", ascending=False).head(CFG.max_visual_examples))
    groups["large_mask_failure"] = indices_from_cases(q4.sort_values("dice", ascending=True).head(CFG.max_visual_examples))

    out_dir = model_output_dir(spec.name) / "failure_cases"
    for group_name, indices in groups.items():
        df_group = build_failure_case_rows(spec.name, records, outputs, per_df, comp_df, indices, group_name)
        df_group.to_csv(out_dir / f"{group_name}.csv", index=False)
        all_failure_case_rows.append(df_group)
        save_failure_grid(records, outputs, indices, f"{spec.name} - {group_name}", out_dir / f"{group_name}.png")
        failure_summary_rows.append({"model_name": spec.name, "failure_group": group_name, "n": int(len(indices)), "csv": str(out_dir / f"{group_name}.csv"), "png": str(out_dir / f"{group_name}.png")})

final_failure_case_summary = pd.DataFrame(failure_summary_rows)
final_failure_case_summary.to_csv(FINAL_ROOT / "final_failure_case_summary.csv", index=False)
if all_failure_case_rows:
    pd.concat(all_failure_case_rows, ignore_index=True).to_csv(FINAL_ROOT / "failure_cases" / "all_failure_cases.csv", index=False)


def save_model_disagreement_cases() -> None:
    seg_name = "segformer_b0_rgb_384_smallmask"
    eff_name = "efficientnetb0_unet_rgb_384_smallmask"
    if seg_name not in clean_outputs_by_model or eff_name not in clean_outputs_by_model:
        return
    seg_per = clean_outputs_by_model[seg_name]["per_df"].copy()
    eff_per = clean_outputs_by_model[eff_name]["per_df"].copy()
    seg_per["case_key"] = seg_per["image_id"].astype(str) + "|" + seg_per["class_name"].astype(str)
    eff_per["case_key"] = eff_per["image_id"].astype(str) + "|" + eff_per["class_name"].astype(str)
    merged = seg_per.merge(eff_per, on="case_key", suffixes=("_segformer", "_efficientnet"), validate="one_to_one")
    merged = merged[merged["image_label_segformer"].astype(int) == 1].copy()
    merged["dice_diff_segformer_minus_efficientnet"] = merged["dice_segformer"] - merged["dice_efficientnet"]
    seg_wins = merged.sort_values("dice_diff_segformer_minus_efficientnet", ascending=False).head(CFG.max_visual_examples)
    eff_wins = merged.sort_values("dice_diff_segformer_minus_efficientnet", ascending=True).head(CFG.max_visual_examples)
    rows = []
    for group_name, part in [("segformer_success_efficientnet_failure", seg_wins), ("efficientnet_success_segformer_failure", eff_wins)]:
        temp = part.copy()
        temp["failure_group"] = group_name
        rows.append(temp)
    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_df.to_csv(FINAL_ROOT / "model_disagreement_cases.csv", index=False)
    out_df.to_csv(FINAL_ROOT / "failure_cases" / "model_disagreement_cases.csv", index=False)

    # Disagreement gorselinde iki modelin post maskeleri yan yana verilir.
    # Her iki yondeki ayrismayi gostermek icin satirlari dengeli seciyoruz:
    # SegFormer iyi/EfficientNet zayif ve EfficientNet iyi/SegFormer zayif.
    n_total = min(CFG.max_visual_examples, len(out_df))
    n_seg = min(len(seg_wins), (n_total + 1) // 2)
    n_eff = min(len(eff_wins), n_total - n_seg)
    if n_eff < n_total - n_seg and len(seg_wins) > n_seg:
        n_seg = min(len(seg_wins), n_total - n_eff)
    selected = pd.concat([seg_wins.head(n_seg), eff_wins.head(n_eff)], ignore_index=True) if n_total else pd.DataFrame()
    if not selected.empty:
        selected_groups = (
            ["segformer_success_efficientnet_failure"] * n_seg
            + ["efficientnet_success_segformer_failure"] * n_eff
        )
        selected["failure_group"] = selected_groups[: len(selected)]
    if selected.empty:
        return
    fig, axes = plt.subplots(len(selected), 6, figsize=(12, max(2, len(selected) * 1.9)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)
    seg_records = clean_records_by_model[seg_name]
    eff_records = clean_records_by_model[eff_name]
    seg_idx = {f"{r['image_id']}|{r.get('class_name', '')}": i for i, r in enumerate(seg_records)}
    eff_idx = {f"{r['image_id']}|{r.get('class_name', '')}": i for i, r in enumerate(eff_records)}
    seg_out = clean_outputs_by_model[seg_name]["outputs"]
    eff_out = clean_outputs_by_model[eff_name]["outputs"]
    headers = ["original", "gt", "Seg prob", "Seg pred", "Eff prob", "Eff pred"]
    for r, row in enumerate(selected.itertuples()):
        image_id = str(row.image_id_segformer)
        case_key = str(row.case_key)
        si, ei = seg_idx[case_key], eff_idx[case_key]
        rec = seg_records[si]
        image = load_image_rgb(rec["image_path"])
        image = cv2.resize(image, (rec["prob"].shape[1], rec["prob"].shape[0]), interpolation=cv2.INTER_AREA)
        panels = [
            resize_for_grid(image),
            resize_for_grid(rec["mask"] * 255, is_mask=True),
            resize_for_grid((rec["prob"] * 255).astype(np.uint8)),
            resize_for_grid(seg_out["preds"][si] * 255, is_mask=True),
            resize_for_grid((eff_records[ei]["prob"] * 255).astype(np.uint8)),
            resize_for_grid(eff_out["preds"][ei] * 255, is_mask=True),
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(headers[c], fontsize=8)
        axes[r, 0].set_ylabel(f"{image_id}\nΔ={row.dice_diff_segformer_minus_efficientnet:.3f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(FINAL_ROOT / "model_disagreement_cases.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "model_disagreement_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


save_model_disagreement_cases()


# %% [markdown]
# ## 18. Visualizations

# %%
def save_bar_metrics(df: pd.DataFrame, metrics: List[str], labels: List[str], title: str, out_path: Path) -> None:
    if df.empty:
        return
    plot_df = df[["model_name"] + metrics].melt(id_vars="model_name", var_name="metric", value_name="value")
    label_map = dict(zip(metrics, labels))
    plot_df["metric"] = plot_df["metric"].map(label_map)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model_name")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Skor / oran")
    plt.xticks(rotation=20, ha="right")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_final_plots() -> None:
    if not final_comparison_df.empty:
        save_bar_metrics(
            final_comparison_df,
            ["forged_dice", "q1_dice", "component_f1_iou010", "authentic_fp_rate", "image_f1"],
            ["Forged Dice", "Q1 Dice", "Component F1", "Authentic FP", "Image F1"],
            "Final model karsilastirmasi",
            PLOTS_DIR / "final_model_comparison_barplots.png",
        )
        res = final_comparison_df[final_comparison_df["source_model_name"].isin(["efficientnetb0_unet_rgb_full", "efficientnetb0_unet_rgb_384_smallmask", "segformer_b0_rgb_full", "segformer_b0_rgb_384_smallmask"])]
        if not res.empty:
            save_bar_metrics(
                res,
                ["forged_dice", "q1_dice", "q2_dice", "component_f1_iou010", "authentic_fp_rate"],
                ["Forged Dice", "Q1 Dice", "Q2 Dice", "Component F1", "Authentic FP"],
                "256x256 ve 384x384 cozumurluk etkisi",
                PLOTS_DIR / "resolution_gain_256_vs_384.png",
            )
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=final_comparison_df, x="q1_dice", y="authentic_fp_rate", hue="model_name", s=90)
        plt.xlabel("Q1 Dice")
        plt.ylabel("Authentic FP rate")
        plt.title("Kucuk maske basarimi - yanlis alarm dengesi")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "scatter_q1_dice_vs_authfp_final.png", dpi=300)
        plt.close()
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=final_comparison_df, x="forged_dice", y="image_f1", hue="model_name", s=90)
        plt.xlabel("Forged Dice")
        plt.ylabel("Image F1")
        plt.title("Lokalizasyon ve goruntu duzeyi basarim")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "scatter_forged_dice_vs_image_f1_final.png", dpi=300)
        plt.close()

    if not robustness_df.empty:
        for metric, filename, title, ylabel in [
            ("forged_dice", "robustness_forged_dice.png", "Robustness - Forged Dice", "Forged Dice"),
            ("q1_dice", "robustness_q1_dice.png", "Robustness - Q1 Dice", "Q1 Dice"),
            ("authentic_fp_rate", "robustness_auth_fp_rate.png", "Robustness - Authentic FP rate", "Authentic FP rate"),
            ("component_f1_iou010", "robustness_component_f1.png", "Robustness - Component F1@0.10", "Component F1"),
        ]:
            plt.figure(figsize=(10, 4.5))
            sns.lineplot(data=robustness_df, x="degradation", y=metric, hue="model_name", marker="o")
            plt.xticks(rotation=25, ha="right")
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel("Bozulma kosulu")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / filename, dpi=300)
            plt.close()

    small_rows = []
    for model_name, pack in clean_outputs_by_model.items():
        df = pack["small_df"].copy()
        df["model_name"] = model_name
        small_rows.append(df)
    if small_rows:
        small_all = pd.concat(small_rows, ignore_index=True)
        plt.figure(figsize=(7, 4.5))
        sns.barplot(data=small_all, x="mask_quartile", y="mean_dice", hue="model_name")
        plt.title("Final modellerde maske boyutu quartile Dice")
        plt.xlabel("Maske quartile")
        plt.ylabel("Per-image Dice")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "small_mask_quartile_final.png", dpi=300)
        plt.close()

    dist_rows = []
    for model_name, pack in clean_outputs_by_model.items():
        per = pack["per_df"].copy()
        per = per[per["image_label"].astype(int) == 1]
        per["model_name"] = model_name
        dist_rows.append(per)
    if dist_rows:
        dist_all = pd.concat(dist_rows, ignore_index=True)
        plt.figure(figsize=(7, 4.5))
        sns.violinplot(data=dist_all, x="model_name", y="dice", inner="box", cut=0)
        plt.xticks(rotation=15, ha="right")
        plt.title("Per-image Dice dagilimi")
        plt.xlabel("")
        plt.ylabel("Dice")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "per_image_dice_distribution.png", dpi=300)
        plt.close()

    if clean_outputs_by_model:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for model_name, pack in clean_outputs_by_model.items():
            per = pack["per_df"]
            y_true = per["image_label"].astype(int).to_numpy()
            scores = per["image_score"].astype(float).to_numpy()
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, scores)
                prec, rec, _ = precision_recall_curve(y_true, scores)
                axes[0].plot(fpr, tpr, label=f"{model_name} AUC={roc_auc_score(y_true, scores):.3f}")
                axes[1].plot(rec, prec, label=f"{model_name} AP={average_precision_score(y_true, scores):.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[0].set_title("ROC egrisi")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[1].set_title("Precision-Recall egrisi")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        for ax in axes:
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "roc_pr_curves_final.png", dpi=300)
        plt.close(fig)

        plt.figure(figsize=(6, 5))
        for model_name, pack in clean_outputs_by_model.items():
            rel = pack["reliability_df"]
            rel = rel[rel["count"] > 0]
            plt.plot(rel["avg_confidence"], rel["accuracy"], marker="o", label=model_name)
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        plt.xlabel("Ortalama guven")
        plt.ylabel("Gercek pozitif orani")
        plt.title("Reliability diagram")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "reliability_diagram_final.png", dpi=300)
        plt.close()

        fig, axes = plt.subplots(1, len(clean_outputs_by_model), figsize=(5 * len(clean_outputs_by_model), 4))
        if len(clean_outputs_by_model) == 1:
            axes = [axes]
        for ax, (model_name, pack) in zip(axes, clean_outputs_by_model.items()):
            per = pack["per_df"]
            cm = confusion_matrix(per["image_label"].astype(int), per["image_pred_label"].astype(int), labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(model_name)
            ax.set_xlabel("Pred")
            ax.set_ylabel("GT")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "confusion_matrices_final.png", dpi=300)
        plt.close(fig)


make_final_plots()


# %% [markdown]
# ## 19. Optional Kaggle Submission

# %%
def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def create_optional_submission() -> None:
    if not CFG.create_submission:
        return
    spec = FINAL_MODELS[0]
    model = loaded_models.get(spec.name)
    selected = selected_configs_by_model.get(spec.name)
    if model is None or selected is None:
        print("[warn] Submission icin checkpoint/config yok.")
        return
    sample_path = DATASET_ROOT / "sample_submission.csv"
    test_img_dir = DATASET_ROOT / "test_images"
    if not sample_path.exists() or not test_img_dir.exists():
        print("[warn] sample_submission.csv veya test_images bulunamadi; submission atlandi.")
        return
    sample = pd.read_csv(sample_path)
    rows = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="submission"):
        image_id = str(row.get("image_id", row.iloc[0]))
        img_path = test_img_dir / f"{image_id}.png"
        if not img_path.exists():
            rows.append({**row.to_dict(), "rle": ""})
            continue
        image = load_image_rgb(str(img_path))
        h, w = image.shape[:2]
        aug = get_eval_transforms(spec.image_size)(image=image, mask=np.zeros((h, w), np.float32))
        tensor = torch.from_numpy(aug["image"].astype(np.float32).transpose(2, 0, 1))[None].float().to(DEVICE)
        with torch.no_grad(), amp_context():
            prob = torch.sigmoid(model(tensor)).detach().cpu().numpy()[0, 0]
        out = postprocess_probability_map(prob, selected)
        pred = cv2.resize(out["mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        encoded = rle_encode(pred)
        rec = row.to_dict()
        if "rle" in rec:
            rec["rle"] = encoded
        elif "encoded_pixels" in rec:
            rec["encoded_pixels"] = encoded
        else:
            print("[warn] Submission format belirsiz; `rle` kolonu eklendi.")
            rec["rle"] = encoded
        rows.append(rec)
    pd.DataFrame(rows).to_csv(FINAL_ROOT / "submission.csv", index=False)


create_optional_submission()


# %% [markdown]
# ## 20. Report Generation

# %%
def fmt_metric(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def generate_final_report() -> None:
    lines = []
    lines.append("# Final Analiz Raporu\n")
    lines.append("## 1. Final analizin amaci\n")
    lines.append("Bu analiz Deney 6 sonrasi iki final aday modeli yeni egitim yapmadan clean test, robustness, failure case ve istatistiksel karsilastirma protokolleriyle degerlendirir.\n")
    lines.append("## 2. Onceki deneylerin kisa ozeti\n")
    lines.append("Deney 5 calibration ve post-processing ile authentic false alarm oranini dusurmus, fakat kucuk sahtecilik bolgelerinde Q1 performansi sinirli kalmistir. Deney 6, EfficientNetB0-UNet ve SegFormer-B0 modellerini 384x384 cozumurlukte egiterek Q1/Q2 performansini belirgin artirmistir.\n")
    lines.append("## 3. Final aday modeller\n")
    lines.append("- `SegFormer-B0 384 balanced`: Localization-oriented final model.\n")
    lines.append("- `EfficientNetB0-UNet 384 balanced / low false alarm`: Conservative low-false-alarm final model.\n")
    lines.append("## 4. PNG goruntulerde JPEG robustness testinin gerekcesi\n")
    lines.append("Bilimsel gorseller makale, PDF, sunum ve web sureclerinde yeniden kaydedilebilir. Bu nedenle PNG test goruntuleri kalici olarak degistirilmeden in-memory JPEG encode/decode ile bozulur; maskeler degistirilmez ve threshold ayarlari clean validation secimiyle sabit tutulur.\n")
    lines.append("## 5. Ana final karsilastirma tablosu\n")
    lines.append(final_comparison_df.to_markdown(index=False) if not final_comparison_df.empty else "_Final karsilastirma tablosu uretilemedi._")
    lines.append("\n## 6. 256x256 vs 384x384 cozumurluk etkisi\n")
    if not statistical_tests_df.empty:
        res_rows = statistical_tests_df[statistical_tests_df["comparison"].str.contains("384_smallmask vs", regex=False, na=False)]
        lines.append(res_rows.to_markdown(index=False) if not res_rows.empty else "Cozumurluk etkisi icin per-image referans CSV'leri eksik olabilir.")
    else:
        lines.append("Istatistiksel testler icin gerekli per-image CSV'ler eksik oldugundan bu bolum sinirli uretildi.")
    lines.append("\n## 7. SegFormer-B0 384 analizi\n")
    seg = clean_results_df[clean_results_df["model_name"].astype(str) == "segformer_b0_rgb_384_smallmask"] if not clean_results_df.empty else pd.DataFrame()
    if not seg.empty:
        r = seg.iloc[0]
        lines.append(f"SegFormer-B0 384 forged Dice={fmt_metric(r.get('dice_forged_only'))}, Q1 Dice={fmt_metric(r.get('q1_dice'))}, Component F1@0.10={fmt_metric(r.get('component_f1_iou010'))}, authentic FP={fmt_metric(r.get('authentic_fp_rate'))}.")
    else:
        lines.append("SegFormer clean sonuc dosyasi uretilemedi.")
    lines.append("\n## 8. EfficientNetB0-UNet 384 analizi\n")
    eff = clean_results_df[clean_results_df["model_name"].astype(str) == "efficientnetb0_unet_rgb_384_smallmask"] if not clean_results_df.empty else pd.DataFrame()
    if not eff.empty:
        r = eff.iloc[0]
        lines.append(f"EfficientNetB0-UNet 384 forged Dice={fmt_metric(r.get('dice_forged_only'))}, Q1 Dice={fmt_metric(r.get('q1_dice'))}, Component F1@0.10={fmt_metric(r.get('component_f1_iou010'))}, authentic FP={fmt_metric(r.get('authentic_fp_rate'))}.")
    else:
        lines.append("EfficientNet clean sonuc dosyasi uretilemedi.")
    lines.append("\n## 9. Localization vs false alarm trade-off\n")
    lines.append("SegFormer-B0 384 genel lokalizasyon odakli model olarak; EfficientNetB0-UNet 384 ise daha muhafazakar yanlis alarm profili icin birlikte raporlanmalidir.\n")
    lines.append("## 10. Robustness sonuclari\n")
    lines.append(robustness_df.to_markdown(index=False) if not robustness_df.empty else "_Checkpoint bulunamadigi veya robustness kapali oldugu icin robustness sonucu yok._")
    lines.append("\n## 11. JPEG sikistirma etkisi\n")
    lines.append("JPEG90 hafif sikistirma, JPEG70 daha gercekci dagilim kaymasi, JPEG50 ise stres testi olarak yorumlanmalidir. Threshold yeniden secilmedigi icin bu bolum modelin dagilim kaymasi altindaki dogrudan davranisini gosterir.\n")
    lines.append("## 12. Blur ve noise etkisi\n")
    lines.append("Blur Q1 Dice'i dusuruyorsa kucuk sahtecilik bolgelerinin sinir/kenar sinyaline bagimliligi; noise authentic FP oranini artiriyorsa gurultu ile sahtecilik izinin karistigi raporlanmalidir.\n")
    lines.append("## 13. Kucuk maske analizi\n")
    small_tables = []
    for model_name, pack in clean_outputs_by_model.items():
        t = pack["small_df"].copy()
        t["model_name"] = model_name
        small_tables.append(t)
    lines.append(pd.concat(small_tables, ignore_index=True).to_markdown(index=False) if small_tables else "_Kucuk maske tablosu yok._")
    lines.append("\n## 14. Failure case analizi\n")
    lines.append(final_failure_case_summary.to_markdown(index=False) if not final_failure_case_summary.empty else "_Failure case dosyalari uretilemedi._")
    lines.append("\n## 15. Istatistiksel test sonuclari\n")
    lines.append(statistical_tests_df.to_markdown(index=False) if not statistical_tests_df.empty else "_Istatistiksel test sonucu yok._")
    lines.append("\n## 16. Final pipeline onerisi\n")
    lines.append("Aday 1: `SegFormer-B0 384 balanced` - `Localization-oriented final model`. Forged bolgenin en iyi lokalizasyonu istendiginde kullanilmalidir.\n")
    lines.append("Aday 2: `EfficientNetB0-UNet 384 balanced / low false alarm` - `Conservative low-false-alarm final model`. Gercek goruntulerde yanlis alarm maliyeti yuksek oldugunda kullanilmalidir.\n")
    lines.append("Eger ana hedef piksel/bilesen lokalizasyon basarisiysa final model SegFormer-B0 384'tur. Eger pratik kullanimda dusuk yanlis alarm ve goruntu duzeyi guvenilirlik oncelikliyse final model EfficientNetB0-UNet 384'tur. Tezde iki model birlikte raporlanmalidir; cunku farkli operasyonel onceliklere hizmet etmektedirler.\n")
    lines.append("## 17. Sinirliliklar\n")
    lines.append("Robustness bolumu checkpoint gerektirir. Checkpoint eksikse clean analiz cached probability map ile tamamlanir, ancak degrade goruntu forward pass'i yapilamaz. Sonuclar validation secimli threshold'lara baglidir ve test setinde yeniden ayar yapilmaz.\n")
    lines.append("## 18. Gelecek calisma onerileri\n")
    lines.append("DINOv2-lite limited unfreeze, daha guclu domain augmentation, JPEG/blur/noise ile validation-time threshold adaptation ve uncertainty-aware post-processing gelecek calisma olarak degerlendirilebilir.\n")
    if robustness_errors:
        lines.append("\n## Robustness uyarilari\n")
        lines.append(pd.DataFrame(robustness_errors).to_markdown(index=False))
    with open(FINAL_ROOT / "final_analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


decision = {
    "primary_localization_model": "segformer_b0_rgb_384_smallmask",
    "primary_localization_label": "Localization-oriented final model",
    "conservative_model": "efficientnetb0_unet_rgb_384_smallmask",
    "conservative_label": "Conservative low-false-alarm final model",
    "decision_sentence": "Eger ana hedef piksel/bilesen lokalizasyon basarisiysa final model SegFormer-B0 384'tur; eger pratik kullanimda dusuk yanlis alarm ve goruntu duzeyi guvenilirlik oncelikliyse final model EfficientNetB0-UNet 384'tur. Tezde iki model birlikte raporlanmalidir.",
}
save_json(decision, FINAL_ROOT / "final_decision_summary.json")
generate_final_report()

print("\nFinal analiz tamamlandi.")
print("Cikti klasoru:", FINAL_ROOT)
print("Kaggle/Colab icin gerekli inputlar:")
print("- Dataset root: /kaggle/input/competitions/recodai-luc-scientific-image-forgery-detection veya /content/drive/MyDrive/bitirmeProjesi/dataset")
print("- Deney root: /content/drive/MyDrive/bitirmeProjesi/experiments_full veya /kaggle/working/experiments_full")
print("- Gerekli klasorler: experiment6_smallmask_384/{efficientnetb0_unet_rgb_384_smallmask,segformer_b0_rgb_384_smallmask}, experiment5_calibration_postprocessing, _shared_splits_seed42")
