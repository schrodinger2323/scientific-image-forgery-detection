from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luc_forgery_pipeline.compare import create_model_comparison
from luc_forgery_pipeline.config import ExperimentConfig, FINAL_CMFD_MODEL_NAMES, FORENSIC_MODEL_NAMES
from luc_forgery_pipeline.torch_train import train_one_torch_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QDL-CMFD/Siamese/Self-correlation CMFD baselines.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/drive/MyDrive/bitirmeProjesi/dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("/content/drive/MyDrive/bitirmeProjesi"))
    parser.add_argument("--models", nargs="+", default=list(FINAL_CMFD_MODEL_NAMES), choices=list(FORENSIC_MODEL_NAMES))
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-comparison", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        samples_per_class=None if args.full_data else args.samples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        models=tuple(args.models),
        num_workers=args.num_workers,
    )
    for model_name in args.models:
        print(f"\n===== Training {model_name} =====")
        train_one_torch_model(model_name, config)
    if not args.skip_comparison:
        comparison_dir = create_model_comparison(config, args.models)
        print(f"\nModel comparison written to: {comparison_dir}")


if __name__ == "__main__":
    main()
