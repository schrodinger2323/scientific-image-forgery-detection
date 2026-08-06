# %% [markdown]
# # Recod.ai/LUC - Scientific Image Forgery Detection
#
# Bu notebook, scientific image forgery detection problemi için pixel-level binary segmentation
# deneylerini Colab veya Kaggle üzerinde tekrarlanabilir şekilde çalıştırmak üzere hazırlanmıştır.
#
# Ana özellikler:
# - Authentic görüntüler için otomatik sıfır maske üretimi.
# - Forged görüntüler için `.npy` maske okuma ve multi-channel maske birleştirme.
# - `image_id` group olarak kullanılarak leakage engelleyen stratified group split.
# - RGB baseline U-Net++.
# - RGB + SRM edge-aware multi-task U-Net++.
# - Threshold search, pixel-level, image-level ve boundary metrikleri.
# - Robustness testleri, görselleştirme ve failure-case analizi.

# %% [markdown]
# ## 1) Imports and Config

# %%
import os
import sys
import json
import math
import time
import random
import warnings
import subprocess
import shutil
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Colab/Jupyter ortamında bazı kütüphaneler eğitimle ilgisi olmayan çok gürültülü
# uyarılar basabiliyor. Gerçek hataları saklamadan bilinen sürüm/geçiş uyarılarını
# susturuyoruz.
warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*Unclosed socket.*")
warnings.filterwarnings("ignore", message=".*can only test a child process.*")
warnings.filterwarnings("ignore", message=".*multi-threaded, use of fork.*")
warnings.filterwarnings("ignore", message=".*The secret `HF_TOKEN` does not exist.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow\\(\\) is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyparsing.*")
warnings.filterwarnings("ignore", message=".*parseString.*deprecated.*")
warnings.filterwarnings("ignore", message=".*resetCache.*deprecated.*")
warnings.filterwarnings("ignore", message=".*oneOf.*deprecated.*")
warnings.filterwarnings("ignore", message=".*setParseAction.*deprecated.*")


KAGGLE_DATASET_CANDIDATES = [
    "/kaggle/input/datasets/koushikkumardinda/scientific-image-forgery-detection/recodai-luc-scientific-image-forgery-detection",
    "/kaggle/input/scientific-image-forgery-detection/recodai-luc-scientific-image-forgery-detection",
    "/kaggle/input/recodai-luc-scientific-image-forgery-detection",
]
COLAB_DATASET_ROOT = "/content/drive/MyDrive/bitirmeProjesi/dataset"
COLAB_EXPERIMENTS_ROOT = "/content/drive/MyDrive/bitirmeProjesi/experiments"
KAGGLE_EXPERIMENTS_ROOT = "/kaggle/working/experiments"


def is_kaggle_runtime() -> bool:
    return Path("/kaggle/input").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def default_dataset_root() -> str:
    if is_kaggle_runtime():
        for candidate in KAGGLE_DATASET_CANDIDATES:
            if Path(candidate).exists():
                return candidate
        # Klasör henüz attach edilmediyse bile verilen resmi yolu default göster.
        return KAGGLE_DATASET_CANDIDATES[0]
    return COLAB_DATASET_ROOT


def default_experiments_root() -> str:
    if is_kaggle_runtime():
        return KAGGLE_EXPERIMENTS_ROOT
    return COLAB_EXPERIMENTS_ROOT


def ensure_package(import_name: str, pip_name: Optional[str] = None) -> None:
    """Colab ortamında eksik paketleri güvenli şekilde kurar."""
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"[install] {pip_name} kuruluyor...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as exc:
            runtime = "Kaggle" if is_kaggle_runtime() else "Colab/local"
            raise RuntimeError(
                f"{pip_name} kurulamadı. Ortam: {runtime}. "
                "Kaggle kullanıyorsan notebook ayarlarından Internet'i aç veya paketi önceden eklenmiş bir environment kullan."
            ) from exc


ensure_package("cv2", "opencv-python-headless")
ensure_package("sklearn", "scikit-learn")
ensure_package("matplotlib")
ensure_package("tqdm")
ensure_package("segmentation_models_pytorch", "segmentation-models-pytorch")
try:
    ensure_package("albumentations")
    ALBUMENTATIONS_AVAILABLE = True
except Exception as exc:
    print(f"[warning] Albumentations kurulamadı veya import edilemedi: {exc}")
    ALBUMENTATIONS_AVAILABLE = False

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from torch.amp import GradScaler, autocast
    TORCH_AMP_NEW_API = True
except Exception:
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AMP_NEW_API = False

import segmentation_models_pytorch as smp

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except Exception as exc:
    print(f"[warning] Albumentations fallback aktif: {exc}")
    ALBUMENTATIONS_AVAILABLE = False


@dataclass
class GlobalConfig:
    dataset_root: str = default_dataset_root()
    experiments_root: str = default_experiments_root()
    seed: int = 42
    image_size: int = 256
    batch_size: int = 8
    # Colab/Jupyter içinde multiprocessing DataLoader worker'ları bazen
    # "can only test a child process" ve ZMQ socket uyarıları üretir.
    # 0 daha temiz ve daha tekrarlanabilir; hız istersen terminal/script ortamında 2-4 yapılabilir.
    num_workers: int = 0
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    threshold_start: float = 0.10
    threshold_end: float = 0.91
    threshold_step: float = 0.05
    edge_loss_weight: float = 0.4
    edge_enhance_weight: float = 0.2
    image_level_area_threshold: float = 0.001
    pos_weight_max: float = 50.0
    edge_pos_weight_max: float = 80.0
    pos_weight_scan_samples: int = 700
    use_compression_aug: bool = False  # Veri PNG olduğu için ilk koşuda JPEG augmentasyonu kapalı.
    run_robustness: bool = True
    use_amp: bool = True


@dataclass
class ExperimentConfig:
    name: str
    model_name: str = "UnetPlusPlus"
    encoder_name: str = "resnet34"
    input_mode: str = "rgb"
    output_channels: int = 1
    encoder_weights: str = "imagenet"
    run: bool = True
    primary_inference_mode: str = "surface"


CFG = GlobalConfig()

EXPERIMENTS = [
    ExperimentConfig(
        name="unetpp_resnet34_rgb_baseline",
        encoder_name="resnet34",
        input_mode="rgb",
        output_channels=1,
        primary_inference_mode="surface",
    ),
    ExperimentConfig(
        name="unetpp_resnet34_rgb_srm_edge_multitask",
        encoder_name="resnet34",
        input_mode="rgb_srm",
        output_channels=2,
        primary_inference_mode="edge_enhanced",
    ),
    # GPU belleği yeterliyse run=True yaparak üçüncü deneyi açabilirsiniz.
    ExperimentConfig(
        name="unetpp_resnet50_rgb_srm_edge_multitask",
        encoder_name="resnet50",
        input_mode="rgb_srm",
        output_channels=2,
        primary_inference_mode="edge_enhanced",
        run=False,
    ),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(CFG.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## 2) Path Discovery

# %%
def runtime_name() -> str:
    if is_kaggle_runtime():
        return "kaggle"
    if "google.colab" in sys.modules:
        return "colab"
    return "local"


def mount_drive_if_colab() -> None:
    """Colab içinde çalışırken Google Drive'ı bağlar; Kaggle'da hiçbir şey yapmaz."""
    if "google.colab" in sys.modules:
        from google.colab import drive

        drive.mount("/content/drive")


mount_drive_if_colab()

DATASET_ROOT = Path(CFG.dataset_root)
EXPERIMENTS_ROOT = Path(CFG.experiments_root)
EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

AUTH_DIR = DATASET_ROOT / "train_images" / "authentic"
FORGED_DIR = DATASET_ROOT / "train_images" / "forged"
MASK_DIR = DATASET_ROOT / "train_masks"
TEST_IMAGE_DIR = DATASET_ROOT / "test_images"
SAMPLE_SUBMISSION = DATASET_ROOT / "sample_submission.csv"

print(f"Runtime: {runtime_name()}")
print(f"Dataset root: {DATASET_ROOT}")
print(f"Experiments root: {EXPERIMENTS_ROOT}")
for path in [DATASET_ROOT, AUTH_DIR, FORGED_DIR, MASK_DIR, TEST_IMAGE_DIR]:
    print(f"{path}: exists={path.exists()}")

# %% [markdown]
# ## 3) DataFrame Creation

# %%
def image_id_from_path(path: Path) -> str:
    return path.stem


def build_dataframe(dataset_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    authentic_paths = sorted((dataset_root / "train_images" / "authentic").glob("*.png"))
    forged_paths = sorted((dataset_root / "train_images" / "forged").glob("*.png"))

    for image_path in authentic_paths:
        image_id = image_id_from_path(image_path)
        rows.append(
            {
                "sample_id": f"authentic__{image_id}",
                "image_id": image_id,
                "class_name": "authentic",
                "label": 0,
                "image_path": str(image_path),
                "mask_path": "",
            }
        )

    skipped_missing_masks = 0
    for image_path in forged_paths:
        image_id = image_id_from_path(image_path)
        mask_path = dataset_root / "train_masks" / f"{image_id}.npy"
        if not mask_path.exists():
            warnings.warn(f"Maske bulunamadı, örnek skip ediliyor: {mask_path}")
            skipped_missing_masks += 1
            continue
        rows.append(
            {
                "sample_id": f"forged__{image_id}",
                "image_id": image_id,
                "class_name": "forged",
                "label": 1,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Dataset boş görünüyor. DATASET_ROOT doğru mu? {dataset_root}")

    print(f"Toplam örnek: {len(df)}")
    print(df["class_name"].value_counts())
    if skipped_missing_masks:
        print(f"[warning] Eksik maske nedeniyle skip edilen forged örnek: {skipped_missing_masks}")
    duplicated_ids = df["image_id"].duplicated(keep=False).sum()
    print(f"Aynı image_id paylaşan örnek sayısı: {duplicated_ids}")
    return df


full_df = build_dataframe(DATASET_ROOT)
full_df.head()

# %% [markdown]
# ## 4) Group-Aware Train/Validation/Test Split

# %%
def group_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    group_df = (
        df.groupby("image_id", as_index=False)
        .agg(group_label=("label", "max"), n_samples=("sample_id", "count"))
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    return group_df


def safe_stratified_group_split(
    df: pd.DataFrame,
    seed: int,
    test_fraction: float = 0.20,
    val_fraction: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Önce %20 test, sonra kalan içinden toplamın %10'u validation seçer."""
    group_df = group_level_frame(df)
    groups = group_df["image_id"].values
    y = group_df["group_label"].values

    try:
        sgkf_test = StratifiedGroupKFold(n_splits=int(round(1 / test_fraction)), shuffle=True, random_state=seed)
        trainval_group_idx, test_group_idx = next(sgkf_test.split(group_df, y, groups=groups))
    except Exception as exc:
        print(f"[warning] StratifiedGroupKFold test split başarısız, GroupShuffleSplit kullanılıyor: {exc}")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        trainval_group_idx, test_group_idx = next(splitter.split(group_df, y, groups=groups))

    trainval_groups = group_df.iloc[trainval_group_idx].reset_index(drop=True)
    test_groups = group_df.iloc[test_group_idx]["image_id"].tolist()

    relative_val_fraction = val_fraction / (1.0 - test_fraction)
    n_splits_val = max(2, int(round(1.0 / relative_val_fraction)))
    y_trainval = trainval_groups["group_label"].values
    groups_trainval = trainval_groups["image_id"].values

    try:
        sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed + 1)
        train_group_idx, val_group_idx = next(
            sgkf_val.split(trainval_groups, y_trainval, groups=groups_trainval)
        )
    except Exception as exc:
        print(f"[warning] StratifiedGroupKFold validation split başarısız, GroupShuffleSplit kullanılıyor: {exc}")
        splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val_fraction, random_state=seed + 1)
        train_group_idx, val_group_idx = next(splitter.split(trainval_groups, y_trainval, groups=groups_trainval))

    train_groups = trainval_groups.iloc[train_group_idx]["image_id"].tolist()
    val_groups = trainval_groups.iloc[val_group_idx]["image_id"].tolist()

    train_df = df[df["image_id"].isin(train_groups)].reset_index(drop=True)
    val_df = df[df["image_id"].isin(val_groups)].reset_index(drop=True)
    test_df = df[df["image_id"].isin(test_groups)].reset_index(drop=True)

    overlap = (
        set(train_df["image_id"]) & set(val_df["image_id"])
        or set(train_df["image_id"]) & set(test_df["image_id"])
        or set(val_df["image_id"]) & set(test_df["image_id"])
    )
    if overlap:
        raise RuntimeError(f"Group leakage tespit edildi: {list(overlap)[:5]}")

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = part["label"].value_counts(normalize=False).to_dict()
        print(f"{name}: n={len(part)} ratio={len(part)/len(df):.3f} labels={counts}")

    return train_df, val_df, test_df


train_df, val_df, test_df = safe_stratified_group_split(full_df, CFG.seed)

split_dir = EXPERIMENTS_ROOT / "_shared_splits_seed42"
split_dir.mkdir(parents=True, exist_ok=True)
full_df.to_csv(split_dir / "full.csv", index=False)
train_df.to_csv(split_dir / "train.csv", index=False)
val_df.to_csv(split_dir / "val.csv", index=False)
test_df.to_csv(split_dir / "test.csv", index=False)

# %% [markdown]
# ## 5) Mask Loading

# %%
def load_rgb_image(path: str) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def binarize_loaded_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        binary = mask > 0
    elif mask.ndim == 3:
        # (C, H, W) veya (H, W, C) ayrımı: kanal boyutu genelde küçük olur.
        if mask.shape[0] <= 16 and mask.shape[1] > 16 and mask.shape[2] > 16:
            binary = np.any(mask > 0, axis=0)
        else:
            binary = np.any(mask > 0, axis=-1)
    else:
        raise ValueError(f"Desteklenmeyen maske shape: {mask.shape}")
    return binary.astype(np.uint8)


def load_binary_mask(mask_path: str, label: int, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if int(label) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    if not mask_path or not Path(mask_path).exists():
        warnings.warn(f"Forged örnekte maske yok, sıfır maske üretildi: {mask_path}")
        return np.zeros((h, w), dtype=np.uint8)

    raw_mask = np.load(mask_path, allow_pickle=False)
    mask = binarize_loaded_mask(raw_mask)
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def make_edge_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    edge = (dilated - eroded) > 0
    return edge.astype(np.uint8)

# %% [markdown]
# ## 6) SRM Feature Extraction

# %%
SRM_KERNELS = [
    np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
]


def extract_srm_features(rgb_image: np.ndarray) -> np.ndarray:
    """RGB görüntüden 5 kanallı normalize SRM residual üretir."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    features = []
    for kernel in SRM_KERNELS:
        residual = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        mean = float(residual.mean())
        std = float(residual.std())
        normalized = (residual - mean) / (std + 1e-6)
        normalized = np.clip(normalized, -5.0, 5.0) / 5.0
        features.append(normalized.astype(np.float32))
    return np.stack(features, axis=-1)


def preprocess_input(rgb_image: np.ndarray, input_mode: str) -> np.ndarray:
    rgb = rgb_image.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

    if input_mode == "rgb":
        return np.transpose(rgb, (2, 0, 1)).astype(np.float32)

    if input_mode == "rgb_srm":
        srm = extract_srm_features(rgb_image)
        stacked = np.concatenate([rgb, srm], axis=-1)
        return np.transpose(stacked, (2, 0, 1)).astype(np.float32)

    if input_mode == "rgb_srm_ela":
        # Altyapı hazır bırakıldı. İlk deneylerde kullanılmıyor.
        srm = extract_srm_features(rgb_image)
        ela_placeholder = np.zeros((*rgb_image.shape[:2], 1), dtype=np.float32)
        stacked = np.concatenate([rgb, srm, ela_placeholder], axis=-1)
        return np.transpose(stacked, (2, 0, 1)).astype(np.float32)

    raise ValueError(f"Bilinmeyen input_mode: {input_mode}")


def input_channels(input_mode: str) -> int:
    if input_mode == "rgb":
        return 3
    if input_mode == "rgb_srm":
        return 8
    if input_mode == "rgb_srm_ela":
        return 9
    raise ValueError(input_mode)

# %% [markdown]
# ## 7) Dataset Class

# %%
def apply_perturbation(rgb: np.ndarray, perturbation: Optional[str]) -> np.ndarray:
    if perturbation is None or perturbation == "none":
        return rgb

    if perturbation == "jpeg90" or perturbation == "jpeg70":
        quality = 90 if perturbation == "jpeg90" else 70
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return rgb
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if perturbation == "gaussian_blur":
        return cv2.GaussianBlur(rgb, (5, 5), sigmaX=1.2)

    if perturbation == "gaussian_noise":
        noise = np.random.normal(0.0, 8.0, size=rgb.shape).astype(np.float32)
        return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    raise ValueError(f"Bilinmeyen perturbation: {perturbation}")


class ForgeryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        input_mode: str,
        transforms=None,
        perturbation: Optional[str] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.input_mode = input_mode
        self.transforms = transforms
        self.perturbation = perturbation

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        rgb = load_rgb_image(row["image_path"])
        mask = load_binary_mask(row["mask_path"], int(row["label"]), rgb.shape[:2])

        rgb = apply_perturbation(rgb, self.perturbation)

        if self.transforms is not None:
            transformed = self.transforms(image=rgb, mask=mask)
            rgb = transformed["image"]
            mask = transformed["mask"]
        else:
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.float32)
        edge = make_edge_mask(mask).astype(np.float32)
        image_tensor = torch.from_numpy(np.ascontiguousarray(preprocess_input(rgb, self.input_mode)))
        mask_tensor = torch.from_numpy(mask[None, ...].astype(np.float32))
        edge_tensor = torch.from_numpy(edge[None, ...].astype(np.float32))

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "edge": edge_tensor,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "mask_path": str(row["mask_path"]),
        }


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "edge": torch.stack([item["edge"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
    }

# %% [markdown]
# ## 8) Augmentations

# %%
class FallbackTransforms:
    """Albumentations yoksa temel OpenCV augmentasyonları."""

    def __init__(self, image_size: int, train: bool = True) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if not self.train:
            return {"image": image, "mask": mask}

        if random.random() < 0.5:
            image = np.ascontiguousarray(np.flip(image, axis=1))
            mask = np.ascontiguousarray(np.flip(mask, axis=1))
        if random.random() < 0.5:
            image = np.ascontiguousarray(np.flip(image, axis=0))
            mask = np.ascontiguousarray(np.flip(mask, axis=0))
        k = random.randint(0, 3)
        if k:
            image = np.ascontiguousarray(np.rot90(image, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-18, 18)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return {"image": image, "mask": mask}


def get_transforms(image_size: int, train: bool, use_compression_aug: bool = False):
    if not ALBUMENTATIONS_AVAILABLE:
        return FallbackTransforms(image_size, train=train)

    if train:
        try:
            noise_aug = A.GaussNoise(std_range=(0.02, 0.08), p=0.20)
        except Exception:
            noise_aug = A.GaussNoise(var_limit=(5.0, 30.0), p=0.20)

        affine_kwargs = {
            "translate_percent": (-0.05, 0.05),
            "scale": (0.90, 1.10),
            "rotate": (-20, 20),
            "p": 0.5,
        }
        affine_params = inspect.signature(A.Affine).parameters
        if "border_mode" in affine_params:
            affine_kwargs["border_mode"] = cv2.BORDER_REFLECT_101
        elif "mode" in affine_params:
            affine_kwargs["mode"] = cv2.BORDER_REFLECT_101
        geometric_aug = A.Affine(**affine_kwargs)

        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            geometric_aug,
            A.RandomBrightnessContrast(p=0.35),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            noise_aug,
        ]
        if use_compression_aug:
            compression_aug = None
            if hasattr(A, "ImageCompression"):
                try:
                    compression_aug = A.ImageCompression(quality_lower=70, quality_upper=95, p=0.20)
                except Exception:
                    compression_aug = A.ImageCompression(quality_range=(70, 95), p=0.20)
            elif hasattr(A, "JpegCompression"):
                compression_aug = A.JpegCompression(quality_lower=70, quality_upper=95, p=0.20)
            if compression_aug is not None:
                transforms.append(compression_aug)
        return A.Compose(transforms)

    return A.Compose([A.Resize(image_size, image_size)])


def make_loader(
    df: pd.DataFrame,
    exp: ExperimentConfig,
    train: bool,
    shuffle: bool,
    perturbation: Optional[str] = None,
) -> DataLoader:
    dataset = ForgeryDataset(
        df=df,
        image_size=CFG.image_size,
        input_mode=exp.input_mode,
        transforms=get_transforms(CFG.image_size, train=train, use_compression_aug=CFG.use_compression_aug),
        perturbation=perturbation,
    )
    generator = torch.Generator()
    generator.manual_seed(CFG.seed + (1 if train else 2))
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
        generator=generator,
    )

# %% [markdown]
# ## 9) Model Creation

# %%
def create_model(exp: ExperimentConfig) -> nn.Module:
    if exp.model_name.lower() not in {"unetplusplus", "unet++", "unet"}:
        raise ValueError(f"Desteklenmeyen model_name: {exp.model_name}")

    model_cls = smp.UnetPlusPlus if exp.model_name.lower() in {"unetplusplus", "unet++"} else smp.Unet
    model = model_cls(
        encoder_name=exp.encoder_name,
        encoder_weights=exp.encoder_weights,
        in_channels=input_channels(exp.input_mode),
        classes=exp.output_channels,
        activation=None,
    )
    return model


def split_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if logits.shape[1] == 1:
        return logits[:, :1], None
    return logits[:, :1], logits[:, 1:2]

# %% [markdown]
# ## 10) Losses and Metrics

# %%
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class MultiTaskForgeryLoss(nn.Module):
    def __init__(
        self,
        mask_pos_weight: float,
        edge_pos_weight: float,
        edge_loss_weight: float = 0.4,
        output_channels: int = 2,
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.edge_loss_weight = edge_loss_weight
        self.dice = DiceLoss()
        self.mask_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([mask_pos_weight], dtype=torch.float32))
        self.edge_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([edge_pos_weight], dtype=torch.float32))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mask_bce.to(*args, **kwargs)
        self.edge_bce.to(*args, **kwargs)
        return self

    def forward(self, logits: torch.Tensor, mask_targets: torch.Tensor, edge_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask_logits, edge_logits = split_logits(logits)
        mask_loss = self.mask_bce(mask_logits, mask_targets) + self.dice(mask_logits, mask_targets)
        if self.output_channels == 1 or edge_logits is None:
            return {"loss": mask_loss, "mask_loss": mask_loss.detach(), "edge_loss": torch.zeros_like(mask_loss).detach()}

        edge_loss = self.edge_bce(edge_logits, edge_targets) + self.dice(edge_logits, edge_targets)
        total = mask_loss + self.edge_loss_weight * edge_loss
        return {"loss": total, "mask_loss": mask_loss.detach(), "edge_loss": edge_loss.detach()}


def compute_pos_weights(df: pd.DataFrame, image_size: int, max_samples: int) -> Tuple[float, float]:
    scan_df = df.sample(n=min(len(df), max_samples), random_state=CFG.seed).reset_index(drop=True)
    pos_pixels = 0.0
    edge_pixels = 0.0
    total_pixels = 0.0
    for row in tqdm(scan_df.itertuples(index=False), total=len(scan_df), desc="pos_weight scan"):
        rgb = load_rgb_image(row.image_path)
        mask = load_binary_mask(row.mask_path, int(row.label), rgb.shape[:2])
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        edge = make_edge_mask(mask)
        pos_pixels += float(mask.sum())
        edge_pixels += float(edge.sum())
        total_pixels += float(mask.size)

    neg_pixels = max(total_pixels - pos_pixels, 1.0)
    non_edge_pixels = max(total_pixels - edge_pixels, 1.0)
    mask_weight = neg_pixels / max(pos_pixels, 1.0)
    edge_weight = non_edge_pixels / max(edge_pixels, 1.0)
    mask_weight = float(np.clip(mask_weight, 1.0, CFG.pos_weight_max))
    edge_weight = float(np.clip(edge_weight, 1.0, CFG.edge_pos_weight_max))
    print(f"mask_pos_weight={mask_weight:.3f} edge_pos_weight={edge_weight:.3f}")
    return mask_weight, edge_weight


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    return tp, fp, fn, tn


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = dice
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray, max_points: int = 2_000_000) -> float:
    y_true = y_true.reshape(-1).astype(np.uint8)
    y_prob = y_prob.reshape(-1).astype(np.float32)
    if y_true.max() == 0:
        return 0.0
    if y_true.size > max_points:
        rng = np.random.default_rng(CFG.seed)
        idx = rng.choice(y_true.size, size=max_points, replace=False)
        y_true = y_true[idx]
        y_prob = y_prob[idx]
    return float(average_precision_score(y_true, y_prob))


def per_image_scores(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, tn = binary_confusion(gt, pred)
    base = metrics_from_confusion(tp, fp, fn, tn)
    base["gt_area"] = int(gt.sum())
    base["predicted_area"] = int(pred.sum())
    return base

# %% [markdown]
# ## 11) Training Loop

# %%
def final_probability(surface_prob: np.ndarray, edge_prob: Optional[np.ndarray], inference_mode: str) -> np.ndarray:
    if inference_mode == "surface" or edge_prob is None:
        return surface_prob
    if inference_mode == "edge_enhanced":
        return np.clip(surface_prob * (1.0 + CFG.edge_enhance_weight * edge_prob), 0.0, 1.0)
    raise ValueError(f"Bilinmeyen inference_mode: {inference_mode}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "mask_loss": 0.0, "edge_loss": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        edges = batch["edge"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = CFG.use_amp and device.type == "cuda"
        if TORCH_AMP_NEW_API:
            amp_context = autocast("cuda", enabled=use_amp)
        else:
            amp_context = autocast(enabled=use_amp)

        with amp_context:
            logits = model(images)
            loss_dict = criterion(logits, masks, edges)
            loss = loss_dict["loss"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        for key in totals:
            totals[key] += float(loss_dict[key].detach().cpu())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_loss(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "mask_loss": 0.0, "edge_loss": 0.0}
    n_batches = 0
    for batch in tqdm(loader, desc="val_loss", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        edges = batch["edge"].to(device, non_blocking=True)
        logits = model(images)
        loss_dict = criterion(logits, masks, edges)
        for key in totals:
            totals[key] += float(loss_dict[key].detach().cpu())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_dice: float, exp: ExperimentConfig) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_dice": best_val_dice,
            "experiment": asdict(exp),
            "global_config": asdict(CFG),
        },
        path,
    )


def save_training_curves(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(metrics_df["epoch"], metrics_df["train_loss"], label="train")
    axes[0].plot(metrics_df["epoch"], metrics_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(metrics_df["epoch"], metrics_df["val_dice"], label="val dice")
    axes[1].plot(metrics_df["epoch"], metrics_df["val_iou"], label="val iou")
    axes[1].set_title("Validation overlap")
    axes[1].legend()
    axes[2].plot(metrics_df["epoch"], metrics_df["lr"], label="lr")
    axes[2].set_title("Learning rate")
    axes[2].set_yscale("log")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# %% [markdown]
# ## 12) Validation/Test Evaluation

# %%
@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
    inference_mode: str,
    threshold: float,
    keep_images: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame, List[Dict[str, object]]]:
    model.eval()
    total_tp = total_fp = total_fn = total_tn = 0
    edge_tp = edge_fp = edge_fn = edge_tn = 0
    y_true_pixels: List[np.ndarray] = []
    y_prob_pixels: List[np.ndarray] = []
    per_rows: List[Dict[str, object]] = []
    visual_items: List[Dict[str, object]] = []

    image_true = []
    image_pred = []

    for batch in tqdm(loader, desc=f"eval:{inference_mode}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        mask_logits, edge_logits = split_logits(logits)
        surface_probs = torch.sigmoid(mask_logits).detach().cpu().numpy()[:, 0]
        edge_probs = None if edge_logits is None else torch.sigmoid(edge_logits).detach().cpu().numpy()[:, 0]

        masks = batch["mask"].numpy()[:, 0].astype(np.uint8)
        edges = batch["edge"].numpy()[:, 0].astype(np.uint8)

        for i in range(surface_probs.shape[0]):
            surface_prob = surface_probs[i]
            edge_prob = None if edge_probs is None else edge_probs[i]
            prob = final_probability(surface_prob, edge_prob, inference_mode)
            pred = (prob >= threshold).astype(np.uint8)
            gt = masks[i].astype(np.uint8)
            gt_edge = edges[i].astype(np.uint8)
            pred_edge = (edge_prob >= threshold).astype(np.uint8) if edge_prob is not None else make_edge_mask(pred)

            tp, fp, fn, tn = binary_confusion(gt, pred)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            etp, efp, efn, etn = binary_confusion(gt_edge, pred_edge)
            edge_tp += etp
            edge_fp += efp
            edge_fn += efn
            edge_tn += etn

            y_true_pixels.append(gt.reshape(-1))
            y_prob_pixels.append(prob.reshape(-1))

            pred_area = int(pred.sum())
            gt_area = int(gt.sum())
            image_level_score = float(prob.max())
            image_level_pred = int((pred_area / pred.size) >= CFG.image_level_area_threshold or image_level_score >= threshold)
            image_true.append(int(batch["label"][i]))
            image_pred.append(image_level_pred)

            scores = per_image_scores(gt, pred)
            row = {
                "sample_id": batch["sample_id"][i],
                "image_id": batch["image_id"][i],
                "image_path": batch["image_path"][i],
                "mask_path": batch["mask_path"][i],
                "label": int(batch["label"][i]),
                "inference_mode": inference_mode,
                "threshold": threshold,
                "max_probability": image_level_score,
                **scores,
            }
            per_rows.append(row)

            if keep_images and len(visual_items) < 64:
                rgb = load_rgb_image(batch["image_path"][i])
                rgb = cv2.resize(rgb, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR)
                visual_items.append(
                    {
                        **row,
                        "image": rgb,
                        "gt": gt,
                        "surface_prob": surface_prob,
                        "edge_prob": np.zeros_like(surface_prob) if edge_prob is None else edge_prob,
                        "final_prob": prob,
                        "pred": pred,
                    }
                )

    pixel_metrics = metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)
    edge_metrics = metrics_from_confusion(edge_tp, edge_fp, edge_fn, edge_tn)
    y_true_flat = np.concatenate(y_true_pixels) if y_true_pixels else np.array([0], dtype=np.uint8)
    y_prob_flat = np.concatenate(y_prob_pixels) if y_prob_pixels else np.array([0.0], dtype=np.float32)
    pixel_metrics["auprc"] = safe_auprc(y_true_flat, y_prob_flat)
    pixel_metrics["boundary_f1"] = edge_metrics["f1"]

    if len(set(image_true)) > 1 or len(image_true) > 0:
        pixel_metrics["image_level_accuracy"] = float(accuracy_score(image_true, image_pred))
        pixel_metrics["image_level_precision"] = float(precision_score(image_true, image_pred, zero_division=0))
        pixel_metrics["image_level_recall"] = float(recall_score(image_true, image_pred, zero_division=0))
        pixel_metrics["image_level_f1"] = float(f1_score(image_true, image_pred, zero_division=0))
    else:
        pixel_metrics["image_level_accuracy"] = 0.0
        pixel_metrics["image_level_precision"] = 0.0
        pixel_metrics["image_level_recall"] = 0.0
        pixel_metrics["image_level_f1"] = 0.0

    pixel_metrics["threshold"] = float(threshold)
    pixel_metrics["inference_mode"] = inference_mode
    return pixel_metrics, pd.DataFrame(per_rows), visual_items

# %% [markdown]
# ## 13) Threshold Search

# %%
def threshold_values() -> np.ndarray:
    return np.round(np.arange(CFG.threshold_start, CFG.threshold_end, CFG.threshold_step), 2)


def threshold_search(model, loader, device, inference_modes: List[str]) -> pd.DataFrame:
    rows = []
    for mode in inference_modes:
        for thr in threshold_values():
            metrics, _, _ = collect_predictions(model, loader, device, inference_mode=mode, threshold=float(thr), keep_images=False)
            rows.append(metrics)
    search_df = pd.DataFrame(rows)
    return search_df.sort_values(["dice", "iou"], ascending=False).reset_index(drop=True)


def best_threshold_for_mode(search_df: pd.DataFrame, mode: str) -> float:
    part = search_df[search_df["inference_mode"] == mode].sort_values(["dice", "iou"], ascending=False)
    if part.empty:
        return 0.5
    return float(part.iloc[0]["threshold"])

# %% [markdown]
# ## 14) Visualization

# %%
def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    overlay = rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask > 0] = (1 - alpha) * overlay[mask > 0] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_prediction_visualization(items: List[Dict[str, object]], out_path: Path, n: int = 12) -> None:
    if not items:
        print("[warning] Görselleştirme için örnek yok.")
        return
    selected = items[: min(n, len(items))]
    fig, axes = plt.subplots(len(selected), 6, figsize=(18, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]

    col_titles = ["Original", "GT mask", "Surface prob", "Edge prob", "Binary mask", "Overlay"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title)

    for r, item in enumerate(selected):
        rgb = item["image"]
        gt = item["gt"]
        surface_prob = item["surface_prob"]
        edge_prob = item["edge_prob"]
        pred = item["pred"]
        overlay = overlay_mask(rgb, pred)
        images = [rgb, gt, surface_prob, edge_prob, pred, overlay]
        cmaps = [None, "gray", "magma", "magma", "gray", None]
        for c, img in enumerate(images):
            axes[r, c].imshow(img, cmap=cmaps[c], vmin=0 if c in [2, 3] else None, vmax=1 if c in [2, 3] else None)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(str(item["image_id"])[:18])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_failure_grid(items: List[Dict[str, object]], out_path: Path, title: str, n: int = 10) -> None:
    if not items:
        print(f"[warning] {title}: örnek yok.")
        return
    selected = items[: min(n, len(items))]
    fig, axes = plt.subplots(len(selected), 4, figsize=(13, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]
    for r, item in enumerate(selected):
        rgb = item["image"]
        gt = item["gt"]
        pred = item["pred"]
        overlay = overlay_mask(rgb, pred)
        for c, (img, cmap, subtitle) in enumerate(
            [(rgb, None, "image"), (gt, "gray", "gt"), (pred, "gray", "pred"), (overlay, None, "overlay")]
        ):
            axes[r, c].imshow(img, cmap=cmap)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(subtitle)
        axes[r, 0].set_ylabel(f"{item['image_id']}\ndice={item['dice']:.3f}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# %% [markdown]
# ## 15) Robustness Test

# %%
def run_robustness_tests(
    model,
    exp: ExperimentConfig,
    df: pd.DataFrame,
    device,
    threshold: float,
    inference_mode: str,
) -> pd.DataFrame:
    perturbations = ["jpeg90", "jpeg70", "gaussian_blur", "gaussian_noise"]
    rows = []
    for perturbation in perturbations:
        loader = make_loader(df, exp, train=False, shuffle=False, perturbation=perturbation)
        metrics, _, _ = collect_predictions(
            model,
            loader,
            device,
            inference_mode=inference_mode,
            threshold=threshold,
            keep_images=False,
        )
        metrics["perturbation"] = perturbation
        rows.append(metrics)
    return pd.DataFrame(rows)

# %% [markdown]
# ## 16) Failure Case Analysis

# %%
def enrich_failure_items(per_image_df: pd.DataFrame, visual_items: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    item_by_sample = {item["sample_id"]: item for item in visual_items}
    forged = per_image_df[per_image_df["label"] == 1].copy()
    authentic = per_image_df[per_image_df["label"] == 0].copy()

    low_dice_forged = forged.sort_values(["dice", "iou"], ascending=True).head(10)
    false_positive = authentic[authentic["predicted_area"] > 0].sort_values(
        ["predicted_area", "max_probability"], ascending=False
    ).head(10)
    false_negative = forged[forged["predicted_area"] <= np.maximum(3, forged["gt_area"] * 0.02)].sort_values(
        ["recall", "dice"], ascending=True
    ).head(10)

    def rows_to_items(rows: pd.DataFrame) -> List[Dict[str, object]]:
        items = []
        for sample_id in rows["sample_id"].tolist():
            if sample_id in item_by_sample:
                items.append(item_by_sample[sample_id])
        return items

    return {
        "low_dice_forged": rows_to_items(low_dice_forged),
        "false_positive": rows_to_items(false_positive),
        "false_negative": rows_to_items(false_negative),
    }


@torch.no_grad()
def build_visual_items_from_rows(
    model,
    exp: ExperimentConfig,
    rows: pd.DataFrame,
    threshold: float,
    inference_mode: str,
    device,
) -> List[Dict[str, object]]:
    model.eval()
    items = []
    for row in rows.itertuples(index=False):
        rgb = load_rgb_image(row.image_path)
        # Eski bir koşudan gelen per-image CSV'de mask_path olmayabilir.
        # Bu durumda forged örnek için image_id'den maskeyi tekrar buluyoruz,
        # authentic örnek için zaten sıfır maske üretilecek.
        mask_path = getattr(row, "mask_path", "")
        if not mask_path and int(row.label) == 1:
            mask_path = str(MASK_DIR / f"{row.image_id}.npy")
        mask = load_binary_mask(mask_path, int(row.label), rgb.shape[:2])
        rgb_resized = cv2.resize(rgb, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(preprocess_input(rgb_resized, exp.input_mode))[None].to(device)
        logits = model(x)
        mask_logits, edge_logits = split_logits(logits)
        surface_prob = torch.sigmoid(mask_logits)[0, 0].detach().cpu().numpy()
        edge_prob = None if edge_logits is None else torch.sigmoid(edge_logits)[0, 0].detach().cpu().numpy()
        prob = final_probability(surface_prob, edge_prob, inference_mode)
        pred = (prob >= threshold).astype(np.uint8)
        items.append(
            {
                "sample_id": row.sample_id,
                "image_id": row.image_id,
                "image": rgb_resized,
                "gt": mask_resized.astype(np.uint8),
                "surface_prob": surface_prob,
                "edge_prob": np.zeros_like(surface_prob) if edge_prob is None else edge_prob,
                "final_prob": prob,
                "pred": pred,
                "dice": float(row.dice),
                "iou": float(row.iou),
            }
        )
    return items


def save_failure_case_analysis(
    per_image_df: pd.DataFrame,
    exp_dir: Path,
    model,
    exp: ExperimentConfig,
    threshold: float,
    inference_mode: str,
    device,
) -> None:
    columns = ["image_id", "sample_id", "label", "dice", "iou", "predicted_area", "gt_area", "max_probability"]

    forged = per_image_df[per_image_df["label"] == 1].copy()
    authentic = per_image_df[per_image_df["label"] == 0].copy()

    low_dice_forged = forged.sort_values(["dice", "iou"], ascending=True).head(10)
    false_positive = authentic[authentic["predicted_area"] > 0].sort_values(
        ["predicted_area", "max_probability"], ascending=False
    ).head(10)
    false_negative = forged[forged["predicted_area"] <= np.maximum(3, forged["gt_area"] * 0.02)].sort_values(
        ["recall", "dice"], ascending=True
    ).head(10)

    low_dice_forged[columns].to_csv(exp_dir / "failure_cases_low_dice_forged.csv", index=False)
    false_positive[columns].to_csv(exp_dir / "failure_cases_false_positive.csv", index=False)
    false_negative[columns].to_csv(exp_dir / "failure_cases_false_negative.csv", index=False)

    fp_items = build_visual_items_from_rows(model, exp, false_positive, threshold, inference_mode, device)
    fn_items = build_visual_items_from_rows(model, exp, false_negative, threshold, inference_mode, device)
    save_failure_grid(
        fp_items,
        exp_dir / "failure_cases_false_positive.png",
        "Authentic olup yanlış forged bulunan örnekler",
    )
    save_failure_grid(
        fn_items,
        exp_dir / "failure_cases_false_negative.png",
        "Forged olup neredeyse hiç yakalanmayan örnekler",
    )

# %% [markdown]
# ## 17) Experiment Runner

# %%
def inference_modes_for_experiment(exp: ExperimentConfig) -> List[str]:
    return ["surface", "edge_enhanced"] if exp.output_channels == 2 else ["surface"]


def run_experiment(exp: ExperimentConfig) -> Dict[str, float]:
    seed_everything(CFG.seed)
    exp_dir = EXPERIMENTS_ROOT / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"global": asdict(CFG), "experiment": asdict(exp)}, f, indent=2, ensure_ascii=False)

    print(f"\n===== {exp.name} =====")
    print(asdict(exp))

    train_loader = make_loader(train_df, exp, train=True, shuffle=True)
    val_loader = make_loader(val_df, exp, train=False, shuffle=False)
    test_loader = make_loader(test_df, exp, train=False, shuffle=False)

    mask_pos_weight, edge_pos_weight = compute_pos_weights(train_df, CFG.image_size, CFG.pos_weight_scan_samples)
    model = create_model(exp).to(DEVICE)
    criterion = MultiTaskForgeryLoss(
        mask_pos_weight=mask_pos_weight,
        edge_pos_weight=edge_pos_weight,
        edge_loss_weight=CFG.edge_loss_weight,
        output_channels=exp.output_channels,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CFG.reduce_lr_factor,
        patience=CFG.reduce_lr_patience,
    )
    if TORCH_AMP_NEW_API:
        scaler = GradScaler("cuda", enabled=CFG.use_amp and DEVICE.type == "cuda")
    else:
        scaler = GradScaler(enabled=CFG.use_amp and DEVICE.type == "cuda")

    best_val_dice = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, CFG.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss_metrics = validate_loss(model, val_loader, criterion, DEVICE)

        quick_metrics, _, _ = collect_predictions(
            model,
            val_loader,
            DEVICE,
            inference_mode=exp.primary_inference_mode,
            threshold=0.5,
            keep_images=False,
        )
        val_dice = quick_metrics["dice"]
        val_iou = quick_metrics["iou"]
        scheduler.step(val_dice)

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mask_loss": train_metrics["mask_loss"],
            "train_edge_loss": train_metrics["edge_loss"],
            "val_loss": val_loss_metrics["loss"],
            "val_mask_loss": val_loss_metrics["mask_loss"],
            "val_edge_loss": val_loss_metrics["edge_loss"],
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_precision": quick_metrics["precision"],
            "val_recall": quick_metrics["recall"],
            "val_f1": quick_metrics["f1"],
            "val_auprc": quick_metrics["auprc"],
            "lr": lr,
            "seconds": time.time() - start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(exp_dir / "metrics.csv", index=False)
        save_checkpoint(exp_dir / "last_model.pth", model, optimizer, scheduler, epoch, best_val_dice, exp)

        print(
            f"epoch {epoch:03d}/{CFG.epochs} "
            f"loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} lr={lr:.2e}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(exp_dir / "best_model.pth", model, optimizer, scheduler, epoch, best_val_dice, exp)
        else:
            patience_counter += 1

        if patience_counter >= CFG.patience:
            print(f"Early stopping: best_epoch={best_epoch}, best_val_dice={best_val_dice:.4f}")
            break

    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(exp_dir / "metrics.csv", index=False)
    save_training_curves(metrics_df, exp_dir / "training_curves.png")

    best_ckpt = torch.load(exp_dir / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])

    modes = inference_modes_for_experiment(exp)
    search_df = threshold_search(model, val_loader, DEVICE, modes)
    search_df.to_csv(exp_dir / "threshold_search.csv", index=False)
    primary_threshold = best_threshold_for_mode(search_df, exp.primary_inference_mode)

    val_rows = []
    test_rows = []
    primary_test_per_image = None
    primary_visual_items = None
    for mode in modes:
        mode_threshold = best_threshold_for_mode(search_df, mode)
        val_metrics, _, _ = collect_predictions(
            model,
            val_loader,
            DEVICE,
            inference_mode=mode,
            threshold=mode_threshold,
            keep_images=False,
        )
        val_metrics["split"] = "validation"
        val_metrics["experiment_name"] = exp.name
        val_rows.append(val_metrics)

        test_metrics, per_image_df, visual_items = collect_predictions(
            model,
            test_loader,
            DEVICE,
            inference_mode=mode,
            threshold=mode_threshold,
            keep_images=True,
        )
        test_metrics["split"] = "test"
        test_metrics["experiment_name"] = exp.name
        test_rows.append(test_metrics)

        if mode == exp.primary_inference_mode:
            primary_test_per_image = per_image_df
            primary_visual_items = visual_items

    val_report = pd.DataFrame(val_rows)
    test_report = pd.DataFrame(test_rows)
    val_report.to_csv(exp_dir / "val_metrics.csv", index=False)
    test_report.to_csv(exp_dir / "test_metrics.csv", index=False)

    if primary_visual_items is not None:
        save_prediction_visualization(primary_visual_items, exp_dir / "predictions_visualization.png", n=12)

    if primary_test_per_image is not None and primary_visual_items is not None:
        primary_test_per_image.to_csv(exp_dir / "test_per_image_metrics.csv", index=False)
        save_failure_case_analysis(
            primary_test_per_image,
            exp_dir,
            model,
            exp,
            primary_threshold,
            exp.primary_inference_mode,
            DEVICE,
        )

    if CFG.run_robustness:
        robustness_df = run_robustness_tests(
            model,
            exp,
            test_df,
            DEVICE,
            threshold=primary_threshold,
            inference_mode=exp.primary_inference_mode,
        )
        robustness_df["experiment_name"] = exp.name
        robustness_df.to_csv(exp_dir / "robustness_metrics.csv", index=False)
    else:
        pd.DataFrame().to_csv(exp_dir / "robustness_metrics.csv", index=False)

    best_val_row = val_report[val_report["inference_mode"] == exp.primary_inference_mode].iloc[0].to_dict()
    best_test_row = test_report[test_report["inference_mode"] == exp.primary_inference_mode].iloc[0].to_dict()
    summary = {
        "experiment_name": exp.name,
        "input_mode": exp.input_mode,
        "encoder": exp.encoder_name,
        "primary_inference_mode": exp.primary_inference_mode,
        "best_epoch": best_epoch,
        "selected_threshold": primary_threshold,
        "val_dice": best_val_row["dice"],
        "val_iou": best_val_row["iou"],
        "test_dice": best_test_row["dice"],
        "test_iou": best_test_row["iou"],
        "test_precision": best_test_row["precision"],
        "test_recall": best_test_row["recall"],
        "test_auprc": best_test_row["auprc"],
        "image_level_f1": best_test_row["image_level_f1"],
        "boundary_f1": best_test_row["boundary_f1"],
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def zip_experiments_for_download() -> Optional[Path]:
    """Kaggle'da /kaggle/working altındaki deneyleri tek zip dosyasına toplar."""
    if not is_kaggle_runtime():
        print("[zip] Kaggle runtime değil; zip otomatik oluşturulmadı.")
        return None
    zip_base = Path("/kaggle/working/recod_luc_experiments")
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=str(EXPERIMENTS_ROOT)))
    print(f"[zip] Deney çıktıları indirilebilir zip olarak kaydedildi: {zip_path}")
    return zip_path

# %% [markdown]
# ## 18) Run Experiments

# %%
all_summaries = []
for exp in EXPERIMENTS:
    if not exp.run:
        print(f"[skip] {exp.name}")
        continue
    summary = run_experiment(exp)
    all_summaries.append(summary)

comparison_df = pd.DataFrame(all_summaries)
comparison_path = EXPERIMENTS_ROOT / "experiment_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
zip_experiments_for_download()
comparison_df

# %% [markdown]
# ## 19) Optional Kaggle Submission Inference
#
# Yarışma submission formatı farklı olabileceği için bu bölüm isteğe bağlıdır. Pixel-level RLE veya image-level
# kolonlarını `sample_submission.csv` yapısına göre doldurmaya çalışır.

# %%
def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.astype(np.uint8).T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


@torch.no_grad()
def create_submission_for_experiment(exp_name: str) -> Optional[pd.DataFrame]:
    exp_dir = EXPERIMENTS_ROOT / exp_name
    config_path = exp_dir / "config.json"
    best_model_path = exp_dir / "best_model.pth"
    threshold_path = exp_dir / "threshold_search.csv"
    if not (config_path.exists() and best_model_path.exists() and threshold_path.exists() and SAMPLE_SUBMISSION.exists()):
        print("[submission] Gerekli dosyalar yok, atlandı.")
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    exp = ExperimentConfig(**cfg_dict["experiment"])
    threshold_df = pd.read_csv(threshold_path)
    threshold = best_threshold_for_mode(threshold_df, exp.primary_inference_mode)

    model = create_model(exp).to(DEVICE)
    ckpt = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    test_paths = sorted(TEST_IMAGE_DIR.glob("*.png"))
    for image_path in tqdm(test_paths, desc="submission"):
        rgb = load_rgb_image(str(image_path))
        rgb_resized = cv2.resize(rgb, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(preprocess_input(rgb_resized, exp.input_mode))[None].to(DEVICE)
        logits = model(x)
        mask_logits, edge_logits = split_logits(logits)
        surface_prob = torch.sigmoid(mask_logits)[0, 0].detach().cpu().numpy()
        edge_prob = None if edge_logits is None else torch.sigmoid(edge_logits)[0, 0].detach().cpu().numpy()
        prob = final_probability(surface_prob, edge_prob, exp.primary_inference_mode)
        pred = (prob >= threshold).astype(np.uint8)
        rows.append(
            {
                "id": image_path.stem,
                "image_id": image_path.stem,
                "label": int(pred.sum() > 0),
                "score": float(prob.max()),
                "rle": rle_encode(pred),
            }
        )

    pred_df = pd.DataFrame(rows)
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    output = sample.copy()
    id_col = sample.columns[0]
    pred_df[id_col] = pred_df["id"].astype(str)
    output[id_col] = output[id_col].astype(str)
    merged = output[[id_col]].merge(pred_df, on=id_col, how="left")

    for col in output.columns:
        if col == id_col:
            continue
        lower = col.lower()
        if any(token in lower for token in ["rle", "encoded", "mask"]):
            output[col] = merged["rle"].fillna("")
        elif any(token in lower for token in ["score", "prob"]):
            output[col] = merged["score"].fillna(0.0)
        else:
            output[col] = merged["label"].fillna(0).astype(int)

    out_path = exp_dir / "submission.csv"
    output.to_csv(out_path, index=False)
    print(f"[submission] Kaydedildi: {out_path}")
    return output


# Örnek:
# create_submission_for_experiment("unetpp_resnet34_rgb_srm_edge_multitask")
