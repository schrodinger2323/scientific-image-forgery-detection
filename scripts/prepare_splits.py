from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luc_forgery_pipeline.config import ExperimentConfig
from luc_forgery_pipeline.data import make_or_load_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or inspect shared leakage-safe splits.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/drive/MyDrive/bitirmeProjesi/dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("/content/drive/MyDrive/bitirmeProjesi"))
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        samples_per_class=None if args.full_data else args.samples_per_class,
    )
    full_df, train_df, val_df, test_df = make_or_load_splits(config)
    print("Full subset:")
    print(full_df.groupby("class_name").size())
    print("\nTrain:")
    print(train_df.groupby("class_name").size())
    print("\nVal-tune:")
    print(val_df.groupby("class_name").size())
    print("\nInternal-test:")
    print(test_df.groupby("class_name").size())
    print(f"\nShared split directory: {config.resolved_split_root()}")


if __name__ == "__main__":
    main()
