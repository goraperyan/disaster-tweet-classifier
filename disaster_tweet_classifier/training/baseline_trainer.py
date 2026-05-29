from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from disaster_tweet_classifier.models.baseline import build_baseline_model
from disaster_tweet_classifier.training.metrics import (
    ClassificationMetrics,
    compute_binary_classification_metrics,
)
from disaster_tweet_classifier.training.plots import (
    save_confusion_matrix_plot,
    save_precision_recall_curve_plot,
    save_roc_curve_plot,
)
from disaster_tweet_classifier.utils.serialization import save_joblib_artifact, save_json


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Result of baseline training."""

    model: Pipeline
    metrics: ClassificationMetrics
    validation_targets: np.ndarray
    validation_predictions: np.ndarray
    validation_probabilities: np.ndarray


def split_train_validation_by_fold(
    dataframe: pd.DataFrame,
    fold_column: str,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and validation parts using a fold column."""
    train_dataframe = dataframe[dataframe[fold_column] != validation_fold].reset_index(drop=True)
    validation_dataframe = dataframe[dataframe[fold_column] == validation_fold].reset_index(
        drop=True
    )

    return train_dataframe, validation_dataframe


def train_baseline_model(
    config: DictConfig,
    dataframe: pd.DataFrame,
    validation_fold: int = 0,
) -> BaselineTrainingResult:
    """Train baseline model and compute validation metrics."""
    train_dataframe, validation_dataframe = split_train_validation_by_fold(
        dataframe=dataframe,
        fold_column=config.data.fold_column,
        validation_fold=validation_fold,
    )

    model = build_baseline_model(config=config)

    model.fit(
        train_dataframe[config.data.clean_text_column],
        train_dataframe[config.data.target_column],
    )

    validation_texts = validation_dataframe[config.data.clean_text_column]
    validation_targets = validation_dataframe[config.data.target_column].to_numpy()

    validation_predictions = model.predict(validation_texts)
    validation_probabilities = model.predict_proba(validation_texts)[:, 1]

    metrics = compute_binary_classification_metrics(
        targets=validation_targets,
        predictions=validation_predictions,
        probabilities=validation_probabilities,
    )

    return BaselineTrainingResult(
        model=model,
        metrics=metrics,
        validation_targets=validation_targets,
        validation_predictions=validation_predictions,
        validation_probabilities=validation_probabilities,
    )


def save_baseline_plots(
    result: BaselineTrainingResult,
    plots_dir: Path,
) -> list[Path]:
    """Save baseline validation plots."""
    confusion_matrix_path = plots_dir / "confusion_matrix.png"
    roc_curve_path = plots_dir / "roc_curve.png"
    precision_recall_curve_path = plots_dir / "precision_recall_curve.png"

    save_confusion_matrix_plot(
        targets=result.validation_targets,
        predictions=result.validation_predictions,
        output_path=confusion_matrix_path,
    )
    save_roc_curve_plot(
        targets=result.validation_targets,
        probabilities=result.validation_probabilities,
        output_path=roc_curve_path,
    )
    save_precision_recall_curve_plot(
        targets=result.validation_targets,
        probabilities=result.validation_probabilities,
        output_path=precision_recall_curve_path,
    )

    return [
        confusion_matrix_path,
        roc_curve_path,
        precision_recall_curve_path,
    ]


def train_and_save_baseline_model(
    config: DictConfig,
    input_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    plots_dir: Path,
    validation_fold: int = 0,
) -> BaselineTrainingResult:
    """Train baseline model and save model, metrics and plots."""
    dataframe = pd.read_csv(input_path)

    result = train_baseline_model(
        config=config,
        dataframe=dataframe,
        validation_fold=validation_fold,
    )

    save_joblib_artifact(artifact=result.model, output_path=model_output_path)
    save_json(data=result.metrics.to_dict(), output_path=metrics_output_path)
    save_baseline_plots(result=result, plots_dir=plots_dir)

    return result
