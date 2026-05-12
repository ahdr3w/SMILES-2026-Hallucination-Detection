from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


MLFLOW_AVAILABLE = True

try:
    import mlflow
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


EXPERIMENT_NAME = "smiles-2026-hallucination-detection"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}

    for key, value in d.items():
        new_key = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(_flatten_dict(value, new_key))
        else:
            out[new_key] = _to_jsonable(value)

    return out


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def setup_mlflow(tracking_uri: str = "file:./mlruns") -> None:
    if not MLFLOW_AVAILABLE:
        print("[MLflow] mlflow is not installed, logging disabled.")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_mlflow_run(
    run_name: str,
    config: dict[str, Any],
    tracking_uri: str = "file:./mlruns",
):
    if not MLFLOW_AVAILABLE:
        return None

    setup_mlflow(tracking_uri)

    run = mlflow.start_run(run_name=run_name)

    mlflow.set_tag("branch", _git_value(["rev-parse", "--abbrev-ref", "HEAD"]))
    mlflow.set_tag("commit", _git_value(["rev-parse", "--short", "HEAD"]))
    mlflow.set_tag("task", "SMILES-2026-Hallucination-Detection")

    for key, value in config.items():
        value = _to_jsonable(value)

        if isinstance(value, (list, dict, tuple)):
            mlflow.log_param(key, json.dumps(value, ensure_ascii=False))
        else:
            mlflow.log_param(key, value)

    return run


def log_dataset_info(
    n_samples: int,
    n_hallucinated: int,
    n_truthful: int,
    feature_dim: int | None = None,
) -> None:
    if not MLFLOW_AVAILABLE:
        return

    mlflow.log_param("n_samples", n_samples)
    mlflow.log_param("n_hallucinated", n_hallucinated)
    mlflow.log_param("n_truthful", n_truthful)

    if feature_dim is not None:
        mlflow.log_param("feature_dim", feature_dim)


def log_fold_results(fold_results: list[dict[str, Any]]) -> None:
    if not MLFLOW_AVAILABLE:
        return

    for fold_result in fold_results:
        fold = int(fold_result["fold"])

        for key, value in fold_result.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                mlflow.log_metric(f"fold_{fold}/{key}", float(value))


def log_results_json(path: str = "results.json") -> None:
    if not MLFLOW_AVAILABLE:
        return

    path_obj = Path(path)

    if not path_obj.exists():
        print(f"[MLflow] {path} not found, skip logging.")
        return

    with path_obj.open("r", encoding="utf-8") as f:
        results = json.load(f)

    flat = _flatten_dict(results)

    for key, value in flat.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(key, float(value))

    mlflow.log_artifact(str(path_obj))


def log_artifacts(paths: list[str]) -> None:
    if not MLFLOW_AVAILABLE:
        return

    for path in paths:
        path_obj = Path(path)

        if path_obj.exists():
            mlflow.log_artifact(str(path_obj))


def end_mlflow_run() -> None:
    if not MLFLOW_AVAILABLE:
        return

    mlflow.end_run()
