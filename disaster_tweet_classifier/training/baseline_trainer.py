from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from disaster_tweet_classifier.models.baseline import build_baseline_model
from disaster_tweet_classifier.training.metrics import (
    ClassificationMetrics,
    compute_binary_classification_metrics,
)
from disaster_tweet_classifier.utils.serialization import save_joblib_artifact, save_json


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
) -> tuple[Pipeline, ClassificationMetrics]:
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

    return model, metrics


def train_and_save_baseline_model(
    config: DictConfig,
    input_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    validation_fold: int = 0,
) -> ClassificationMetrics:
    """Train baseline model and save model plus metrics."""
    dataframe = pd.read_csv(input_path)

    model, metrics = train_baseline_model(
        config=config,
        dataframe=dataframe,
        validation_fold=validation_fold,
    )

    save_joblib_artifact(artifact=model, output_path=model_output_path)
    save_json(data=metrics.to_dict(), output_path=metrics_output_path)

    return metrics
