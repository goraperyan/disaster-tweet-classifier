"""FastAPI application for disaster tweet classification."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from disaster_tweet_classifier.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from disaster_tweet_classifier.inference.model_service import BERTweetInferenceService


def load_api_config(
    config_path: str = "configs",
    config_name: str = "config",
) -> DictConfig:
    """Load Hydra config for API runtime.

    The API serves the trained BERTweet model, so it explicitly composes the
    BERTweet training config instead of relying on the project default.
    """
    absolute_config_path = Path(config_path).resolve()

    with initialize_config_dir(
        version_base=None,
        config_dir=str(absolute_config_path),
    ):
        config = compose(
            config_name=config_name,
            overrides=[
                "training=bertweet",
            ],
        )

    OmegaConf.resolve(config)
    return config


def get_checkpoint_path_from_config(config: DictConfig) -> Path:
    """Get checkpoint directory from Hydra config.

    The BERTweet training config stores inference checkpoint path under
    training.inference.checkpoint_path and output checkpoint directory under
    training.outputs.checkpoints_dir.
    """
    candidate_keys = [
        "training.inference.checkpoint_path",
        "training.outputs.checkpoints_dir",
        "training.checkpoint.checkpoint_path",
        "training.checkpoint.checkpoints_dir",
        "training.checkpoint.checkpoint_dir",
        "training.checkpoint_path",
        "training.checkpoints_dir",
        "training.checkpoint_dir",
        "training.output_dir",
        "training.model_dir",
    ]

    for key in candidate_keys:
        value = OmegaConf.select(config, key)
        if value is None:
            continue

        path = Path(str(value))

        if key.endswith("output_dir") or key.endswith("model_dir"):
            return path / "checkpoints"

        return path

    raise ValueError(
        "Could not find checkpoint path in config. Expected one of: " + ", ".join(candidate_keys)
    )


@lru_cache(maxsize=1)
def get_config() -> DictConfig:
    """Create and cache Hydra config."""
    return load_api_config()


@lru_cache(maxsize=1)
def get_model_service() -> BERTweetInferenceService:
    """Create and cache BERTweet inference service."""
    config = get_config()
    checkpoint_path = get_checkpoint_path_from_config(config=config)

    return BERTweetInferenceService(
        config=config,
        checkpoint_path=checkpoint_path,
    )


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Disaster Tweet Classifier",
        description="FastAPI service for disaster-related tweet classification.",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health-check endpoint."""
        return HealthResponse(status="ok")

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        """Return model metadata without loading the model."""
        config = get_config()
        checkpoint_path = get_checkpoint_path_from_config(config=config)

        return ModelInfoResponse(
            model_name=str(config.model.pretrained_model_name),
            checkpoint_path=str(checkpoint_path),
            threshold=float(config.training.inference.threshold),
            max_length=int(config.model.tokenizer.max_length),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Predict disaster label for a single tweet."""
        service = get_model_service()
        prediction = service.predict_one(text=request.text)

        return PredictionResponse(
            text=request.text,
            label=prediction.label,
            label_name=prediction.label_name,
            probability=prediction.probability,
            threshold=prediction.threshold,
        )

    return app


app = create_app()
