from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data import load_binary_mask, load_rgb_image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TorchSegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, augment: bool, seed: int) -> None:
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        image = load_rgb_image(row.image_path, self.img_size)
        mask = load_binary_mask(row.mask_path, int(row.label), self.img_size)
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        image = image / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "sample_id": str(row.sample_id),
            "image_id": str(row.image_id),
            "image_path": str(row.image_path),
            "mask_path": "" if pd.isna(row.mask_path) else str(row.mask_path),
            "label": int(row.label),
        }

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


def make_torch_loader(
    df: pd.DataFrame,
    config: ExperimentConfig,
    augment: bool,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader:
    dataset = TorchSegmentationDataset(
        df=df,
        img_size=config.img_size,
        augment=augment,
        seed=config.seed + seed_offset,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        collate_fn=segmentation_collate_fn,
    )


def segmentation_collate_fn(batch):
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }
