from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import ExperimentConfig, validate_model_name
from .data import copy_shared_splits_to_model_dir, make_or_load_splits, make_sequence
from .evaluation import (
    build_per_image_metrics,
    create_submission,
    evaluate_thresholds,
    metric_summary_with_ci,
    predict_dataframe,
    save_final_figures,
    save_prediction_examples,
    save_threshold_plot,
    save_training_curves,
)
from .losses_metrics import bce_dice_loss, dice_coefficient, dice_loss, iou_coefficient
from .models import build_model, compile_model
from .reports import write_model_report
from .utils import ensure_dir, environment_info, read_training_epochs, save_json, set_global_determinism


CUSTOM_OBJECTS = {
    "bce_dice_loss": bce_dice_loss,
    "dice_loss": dice_loss,
    "dice_coefficient": dice_coefficient,
    "iou_coefficient": iou_coefficient,
}


def model_paths(model_dir: Path) -> Dict[str, Path]:
    return {
        "best_keras": model_dir / "best_model.keras",
        "last_keras": model_dir / "last_model.keras",
        "best_export": model_dir / "best_model",
        "last_export": model_dir / "last_model",
        "backup": model_dir / "backup",
        "training_log": model_dir / "training_log.csv",
    }


def load_or_build_model(model_name: str, config: ExperimentConfig, model_dir: Path):
    paths = model_paths(model_dir)
    if paths["last_keras"].exists():
        return tf.keras.models.load_model(paths["last_keras"], custom_objects=CUSTOM_OBJECTS)
    model = build_model(model_name, config.img_size)
    return compile_model(model, config.learning_rate, config.optimizer)


def export_saved_model(model, export_dir: Path) -> None:
    try:
        if export_dir.exists():
            shutil.rmtree(export_dir)
        model.export(str(export_dir))
    except Exception:
        try:
            tf.saved_model.save(model, str(export_dir))
        except Exception as exc:
            marker = export_dir.with_suffix(".export_failed.txt")
            marker.write_text(str(exc), encoding="utf-8")


def write_model_summary(model, model_dir: Path) -> None:
    stream = io.StringIO()
    model.summary(print_fn=lambda line: stream.write(line + "\n"))
    (model_dir / "model_summary.txt").write_text(stream.getvalue(), encoding="utf-8")


def train_one_model(model_name: str, config: ExperimentConfig) -> Dict[str, float]:
    validate_model_name(model_name)
    set_global_determinism(config.seed)

    model_dir = ensure_dir(config.model_dir(model_name))
    save_json(config.to_json_dict(), model_dir / "experiment_config.json")
    save_json(environment_info(), model_dir / "environment_info.json")

    full_df, train_df, val_df, test_df = make_or_load_splits(config)
    copy_shared_splits_to_model_dir(full_df, train_df, val_df, test_df, model_dir)
    dataset_summary = (
        pd.concat(
            [
                train_df.assign(split="train"),
                val_df.assign(split="val_tune"),
                test_df.assign(split="internal_test"),
            ],
            ignore_index=True,
        )
        .groupby(["split", "class_name"])
        .size()
        .rename("count")
        .reset_index()
    )

    model = load_or_build_model(model_name, config, model_dir)
    write_model_summary(model, model_dir)

    train_seq = make_sequence(train_df, config, model_name, augment=True, shuffle=True, seed_offset=10)
    val_seq = make_sequence(val_df, config, model_name, augment=False, shuffle=False, seed_offset=20)
    paths = model_paths(model_dir)
    initial_epoch = min(read_training_epochs(paths["training_log"]), config.epochs)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths["best_keras"]),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths["last_keras"]),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            mode="max",
            patience=config.reduce_lr_patience,
            factor=config.reduce_lr_factor,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(paths["training_log"]), append=initial_epoch > 0),
        tf.keras.callbacks.BackupAndRestore(str(paths["backup"])),
    ]

    if initial_epoch < config.epochs:
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=config.epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1,
        )
        save_json({k: [float(x) for x in v] for k, v in history.history.items()}, model_dir / "history_last_fit_segment.json")
    if paths["training_log"].exists():
        log = pd.read_csv(paths["training_log"])
        save_json(log.to_dict(orient="list"), model_dir / "history.json")

    save_training_curves(paths["training_log"], model_dir)

    if paths["best_keras"].exists():
        best_model = tf.keras.models.load_model(paths["best_keras"], custom_objects=CUSTOM_OBJECTS)
    else:
        best_model = model
        best_model.save(str(paths["best_keras"]))
    if paths["last_keras"].exists():
        last_model = tf.keras.models.load_model(paths["last_keras"], custom_objects=CUSTOM_OBJECTS)
    else:
        last_model = best_model
        last_model.save(str(paths["last_keras"]))
    export_saved_model(best_model, paths["best_export"])
    export_saved_model(last_model, paths["last_export"])

    val_predictions = predict_dataframe(best_model, val_df, config, model_name)
    threshold_df = evaluate_thresholds(val_predictions, config.threshold_values, config.image_decision_mode)
    threshold_df.to_csv(model_dir / "threshold_analysis.csv", index=False)
    save_threshold_plot(threshold_df, model_dir / "threshold_analysis.png")
    best_threshold = float(threshold_df[threshold_df["selected"]].iloc[0]["threshold"])

    test_predictions = predict_dataframe(best_model, test_df, config, model_name)
    per_image, image_metrics, cm = build_per_image_metrics(
        test_predictions,
        best_threshold,
        config.image_decision_mode,
    )
    per_image.to_csv(model_dir / "test_per_image_metrics.csv", index=False)
    test_summary = metric_summary_with_ci(per_image, image_metrics)
    test_summary.to_csv(model_dir / "test_metrics_summary.csv", index=False)
    pd.DataFrame(cm, index=["true_authentic", "true_forged"], columns=["pred_authentic", "pred_forged"]).to_csv(
        model_dir / "test_confusion_matrix.csv"
    )
    save_final_figures(per_image, cm, model_dir)
    save_prediction_examples(
        test_predictions,
        best_threshold,
        model_dir / "prediction_examples",
        max_examples=config.prediction_examples,
    )
    create_submission(best_model, config, model_name, best_threshold, model_dir / "submission.csv")
    write_model_report(model_dir, model_name, config, dataset_summary, threshold_df, test_summary, image_metrics)

    summary = {
        "model_name": model_name,
        "best_threshold": best_threshold,
        **{row["metric"]: row["mean"] for row in test_summary.to_dict(orient="records")},
    }
    save_json({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summary.items()}, model_dir / "run_summary.json")
    return summary
