from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def flatten_dict(
    data: dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten nested dictionary."""
    flattened: dict[str, Any] = {}

    for key, value in data.items():
        full_key = f"{parent_key}{separator}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            flattened.update(
                flatten_dict(
                    data=value,
                    parent_key=full_key,
                    separator=separator,
                )
            )
        else:
            flattened[full_key] = value

    return flattened


def log_config_params(config: DictConfig) -> None:
    """Log flattened Hydra config as MLflow params."""
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        return

    flattened_config = flatten_dict(config_dict)

    for key, value in flattened_config.items():
        if isinstance(value, (str | int | float | bool)) or value is None:
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, str(value))


def log_metrics(metrics: dict[str, float]) -> None:
    """Log metrics to MLflow."""
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)


def log_artifacts(paths: list[Path]) -> None:
    """Log files or directories to MLflow."""
    for path in paths:
        if path.is_dir():
            mlflow.log_artifacts(str(path))
        elif path.is_file():
            mlflow.log_artifact(str(path))
