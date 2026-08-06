from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for name in ["numpy", "pandas", "sklearn", "tensorflow", "cv2", "matplotlib", "scipy"]:
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            info[name] = f"not available: {exc}"
    try:
        import tensorflow as tf

        info["tf_gpus"] = [gpu.name for gpu in tf.config.list_physical_devices("GPU")]
    except Exception:
        info["tf_gpus"] = []
    return info


def read_training_epochs(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    try:
        import pandas as pd

        log = pd.read_csv(log_path)
        return int(len(log))
    except Exception:
        return 0
