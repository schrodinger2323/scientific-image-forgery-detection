from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.utils import Sequence

from .config import ExperimentConfig
from .utils import ensure_dir


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _list_images(path: Path) -> List[Path]:
    files: List[Path] = []
    for suffix in IMAGE_EXTENSIONS:
        files.extend(path.glob(f"*{suffix}"))
    return sorted(files)


def build_dataset_index(config: ExperimentConfig) -> pd.DataFrame:
    root = config.dataset_root
    auth_dir = root / "train_images" / "authentic"
    forged_dir = root / "train_images" / "forged"
    mask_dir = root / "train_masks"

    rows: List[Dict[str, object]] = []
    for image_path in _list_images(auth_dir):
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "mask_path": "",
                "label": 0,
                "class_name": "authentic",
                "has_mask_file": False,
            }
        )
    for image_path in _list_images(forged_dir):
        mask_path = mask_dir / f"{image_path.stem}.npy"
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "label": 1,
                "class_name": "forged",
                "has_mask_file": mask_path.exists(),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No training images found under {root}")

    df = pd.DataFrame(rows).sort_values(["class_name", "image_id"]).reset_index(drop=True)
    df["group_id"] = df["image_id"].astype(str)
    df["sample_id"] = df["class_name"] + "/" + df["image_id"].astype(str)
    return df


def write_dataset_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    class_summary = df.groupby("class_name").size().rename("count").reset_index()
    class_summary.to_csv(output_dir / "class_summary.csv", index=False)

    missing_masks = df[(df["label"] == 1) & (~df["has_mask_file"].astype(bool))]
    missing_masks.to_csv(output_dir / "missing_forged_masks.csv", index=False)

    collisions = (
        df.groupby("image_id")
        .agg(
            n_samples=("sample_id", "count"),
            n_classes=("class_name", "nunique"),
            classes=("class_name", lambda x: ",".join(sorted(set(x)))),
        )
        .query("n_samples > 1 or n_classes > 1")
        .reset_index()
        .sort_values(["n_classes", "n_samples", "image_id"], ascending=[False, False, True])
    )
    collisions.to_csv(output_dir / "image_id_collision_report.csv", index=False)


def make_or_load_splits(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_root = ensure_dir(config.resolved_split_root())
    full_path = split_root / "full_dataset_index.csv"
    train_path = split_root / "train_split.csv"
    val_path = split_root / "val_tune_split.csv"
    test_path = split_root / "internal_test_split.csv"

    if all(p.exists() for p in [full_path, train_path, val_path, test_path]):
        return (
            pd.read_csv(full_path),
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    df = build_dataset_index(config)
    write_dataset_diagnostics(df, split_root)
    if ((df["label"] == 1) & (~df["has_mask_file"].astype(bool))).any():
        raise FileNotFoundError("At least one forged image is missing its .npy mask. See missing_forged_masks.csv")

    if full_path.exists():
        selected = pd.read_csv(full_path)
    else:
        selected = select_balanced_subset(df, config.samples_per_class, config.subset_seed or config.seed)
    train_df, val_df, test_df = stratified_group_split(selected, config.seed)
    assert_no_group_leakage(train_df, val_df, test_df)

    selected.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return selected, train_df, val_df, test_df


def copy_shared_splits_to_model_dir(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: Path,
) -> None:
    ensure_dir(model_dir)
    full_df.to_csv(model_dir / "full_dataset_index.csv", index=False)
    train_df.to_csv(model_dir / "train_split.csv", index=False)
    val_df.to_csv(model_dir / "val_tune_split.csv", index=False)
    test_df.to_csv(model_dir / "internal_test_split.csv", index=False)


def select_balanced_subset(df: pd.DataFrame, samples_per_class: int | None, seed: int) -> pd.DataFrame:
    if samples_per_class is None:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    parts = []
    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]
        n = min(samples_per_class, len(class_df))
        parts.append(class_df.sample(n=n, random_state=seed + int(label)))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_group_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = df["label"].values
    groups = df["group_id"].astype(str).values
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_test.split(df, labels, groups))

    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    sgkf_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    tv_labels = train_val["label"].values
    tv_groups = train_val["group_id"].astype(str).values
    train_idx, val_idx = next(sgkf_val.split(train_val, tv_labels, tv_groups))

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def assert_no_group_leakage(*splits: pd.DataFrame) -> None:
    names = ["train", "val_tune", "internal_test"]
    group_sets = [set(split["group_id"].astype(str)) for split in splits]
    for i, left in enumerate(group_sets):
        for j, right in enumerate(group_sets):
            if i >= j:
                continue
            overlap = left.intersection(right)
            if overlap:
                preview = ", ".join(sorted(list(overlap))[:10])
                raise RuntimeError(f"Data leakage between {names[i]} and {names[j]} groups: {preview}")


def load_rgb_image(path: str | Path, img_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def load_binary_mask(mask_path: str | Path, label: int, img_size: int) -> np.ndarray:
    if int(label) == 0:
        return np.zeros((img_size, img_size, 1), dtype=np.float32)

    mask = np.load(str(mask_path))
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=0).astype(np.uint8)
    elif mask.ndim == 2:
        mask = (mask > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape} for {mask_path}")
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return mask[..., None].astype(np.float32)


def plain_preprocess(image: np.ndarray) -> np.ndarray:
    return image / 255.0


def efficientnet_preprocess(image: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.efficientnet import preprocess_input

    return preprocess_input(image.copy())


def resnet50_preprocess(image: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input

    return preprocess_input(image.copy())


PREPROCESSORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "plain_unet": plain_preprocess,
    "unetplusplus": plain_preprocess,
    "efficientnetb0_unet": efficientnet_preprocess,
    "resnet50_unet": resnet50_preprocess,
}


class SegmentationSequence(Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int,
        batch_size: int,
        model_name: str,
        augment: bool,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.preprocess = PREPROCESSORS[model_name]
        self.rng = np.random.default_rng(seed)
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        images, masks = [], []
        for row in self.df.iloc[batch_idx].itertuples(index=False):
            image = load_rgb_image(row.image_path, self.img_size)
            mask = load_binary_mask(row.mask_path, row.label, self.img_size)
            if self.augment:
                image, mask = self.apply_augmentation(image, mask)
            images.append(self.preprocess(image))
            masks.append(mask)
        return np.stack(images).astype(np.float32), np.stack(masks).astype(np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        k = int(self.rng.integers(0, 4))
        if k:
            image = np.rot90(image, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
        if self.rng.random() < 0.5:
            alpha = float(self.rng.uniform(0.85, 1.15))
            beta = float(self.rng.uniform(-18.0, 18.0))
            image = np.clip(image * alpha + beta, 0, 255)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def make_sequence(
    df: pd.DataFrame,
    config: ExperimentConfig,
    model_name: str,
    augment: bool,
    shuffle: bool,
    seed_offset: int = 0,
) -> SegmentationSequence:
    return SegmentationSequence(
        df=df,
        img_size=config.img_size,
        batch_size=config.batch_size,
        model_name=model_name,
        augment=augment,
        seed=config.seed + seed_offset,
        shuffle=shuffle,
    )


def list_kaggle_test_images(dataset_root: Path) -> List[Path]:
    test_dir = dataset_root / "test_images"
    return _list_images(test_dir)
