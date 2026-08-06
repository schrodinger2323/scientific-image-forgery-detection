from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


KERAS_MODEL_NAMES = ("plain_unet", "unetplusplus", "efficientnetb0_unet", "resnet50_unet")
PYTORCH_MODEL_NAMES = ("segformer_b0", "deeplabv3plus", "dinov2_seg")
TRACE_MODEL_NAMES = ("mantranet", "mvssnet", "mvssnetpp")
COPYMOVE_MODEL_NAMES = ("busternet", "cmfdformer", "doagan")
FINAL_CMFD_MODEL_NAMES = ("qdl_cmfd", "siamese_cmfd", "selfcorr_cmfd")
FORENSIC_MODEL_NAMES = TRACE_MODEL_NAMES + COPYMOVE_MODEL_NAMES + FINAL_CMFD_MODEL_NAMES
MODEL_NAMES = KERAS_MODEL_NAMES + PYTORCH_MODEL_NAMES + FORENSIC_MODEL_NAMES


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path = Path("/content/drive/MyDrive/bitirmeProjesi/dataset")
    output_root: Path = Path("/content/drive/MyDrive/bitirmeProjesi")
    split_root: Path | None = None
    seed: int = 42
    img_size: int = 256
    batch_size: int = 8
    epochs: int = 40
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.3
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    samples_per_class: int | None = 300
    subset_seed: int | None = None
    image_decision_mode: str = "max_probability"
    threshold_values: Tuple[float, ...] = tuple(round(x / 100, 2) for x in range(10, 91, 5))
    models: Tuple[str, ...] = MODEL_NAMES
    prediction_examples: int = 24
    num_workers: int = 4
    notes: List[str] = field(
        default_factory=lambda: [
            "Initial experiments intentionally use a stratified subset of 300 authentic and 300 forged images.",
            "A single 60/20/20 stratified split is reused across all models.",
            "Kaggle test_images are never used for threshold tuning or internal testing.",
        ]
    )

    def resolved_split_root(self) -> Path:
        subset_tag = "full" if self.samples_per_class is None else f"subset{self.samples_per_class}"
        if self.subset_seed is not None:
            subset_tag = f"{subset_tag}_subsetseed{self.subset_seed}"
        return self.split_root or self.output_root / f"_shared_splits_seed{self.seed}_{subset_tag}"

    def model_dir(self, model_name: str) -> Path:
        return self.output_root / model_name

    def to_json_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["dataset_root"] = str(self.dataset_root)
        data["output_root"] = str(self.output_root)
        data["split_root"] = str(self.resolved_split_root())
        data["threshold_values"] = list(self.threshold_values)
        data["models"] = list(self.models)
        return data


def validate_model_name(model_name: str) -> str:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(MODEL_NAMES)}")
    return model_name
