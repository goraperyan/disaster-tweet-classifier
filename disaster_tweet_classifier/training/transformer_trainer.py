from __future__ import annotations

from pathlib import Path

import lightning as L  # noqa: N812
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from disaster_tweet_classifier.data.lightning_datamodule import TweetDataModule
from disaster_tweet_classifier.models.transformer_classifier import TransformerTweetClassifier
from disaster_tweet_classifier.tracking.git import get_git_commit_hash
from disaster_tweet_classifier.tracking.mlflow_tracking import configure_mlflow
from disaster_tweet_classifier.utils.serialization import save_json


def build_transformer_model(config: DictConfig) -> TransformerTweetClassifier:
    """Build transformer classifier from config."""
    return TransformerTweetClassifier(
        pretrained_model_name=config.model.pretrained_model_name,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        freeze_backbone=config.model.freeze_backbone,
    )


def build_lightning_callbacks(config: DictConfig) -> list:
    """Build Lightning callbacks."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.training.outputs.checkpoints_dir,
        filename=config.training.checkpoint.filename,
        monitor=config.training.checkpoint.monitor,
        mode=config.training.checkpoint.mode,
        save_top_k=config.training.checkpoint.save_top_k,
    )

    early_stopping_callback = EarlyStopping(
        monitor=config.training.early_stopping.monitor,
        mode=config.training.early_stopping.mode,
        patience=config.training.early_stopping.patience,
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    return [
        checkpoint_callback,
        early_stopping_callback,
        learning_rate_monitor,
    ]


def train_transformer_model(config: DictConfig, data_path: Path) -> dict[str, float | str]:
    """Train transformer model with PyTorch Lightning."""
    configure_mlflow(
        tracking_uri=config.training.mlflow.tracking_uri,
        experiment_name=config.training.mlflow.experiment_name,
    )

    datamodule = TweetDataModule(config=config, data_path=data_path)
    model = build_transformer_model(config=config)
    callbacks = build_lightning_callbacks(config=config)

    logger = MLFlowLogger(
        experiment_name=config.training.mlflow.experiment_name,
        run_name=config.training.mlflow.run_name,
        tracking_uri=config.training.mlflow.tracking_uri,
    )

    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=config.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
    )

    git_commit_hash = get_git_commit_hash()
    logger.experiment.log_param(logger.run_id, "git_commit_hash", git_commit_hash)

    trainer.fit(model=model, datamodule=datamodule)

    metrics = {
        key.replace("/", "_"): float(value)
        for key, value in trainer.callback_metrics.items()
        if hasattr(value, "item")
    }

    best_checkpoint_path = ""
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_checkpoint_path = callback.best_model_path

    metrics["best_checkpoint_path"] = best_checkpoint_path

    save_json(data=metrics, output_path=Path(config.training.outputs.metrics_path))

    return metrics
