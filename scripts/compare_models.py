from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luc_forgery_pipeline.compare import create_model_comparison
from luc_forgery_pipeline.config import MODEL_NAMES, ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Create comparison tables/plots for completed model runs.")
    parser.add_argument("--output-root", type=Path, default=Path("/content/drive/MyDrive/bitirmeProjesi"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    args = parser.parse_args()
    config = ExperimentConfig(output_root=args.output_root, models=tuple(args.models))
    out_dir = create_model_comparison(config, args.models)
    print(f"Model comparison written to: {out_dir}")


if __name__ == "__main__":
    main()
