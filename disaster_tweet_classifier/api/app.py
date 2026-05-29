from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI

from disaster_tweet_classifier.api.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictRequest,
)
from disaster_tweet_classifier.commands import load_config
from disaster_tweet_classifier.inference.model_service import BERTweetInferenceService


@lru_cache(maxsize=1)
def get_inference_service() -> BERTweetInferenceService:
    """Initialize and cache inference service."""
    config = load_config(
        overrides=[
            "model=bertweet",
            "training=bertweet",
            "preprocessing=bertweet",
        ]
    )

    checkpoint_path = Path(config.training.inference.checkpoint_path)

    return BERTweetInferenceService(
        config=config,
        checkpoint_path=checkpoint_path,
    )


app = FastAPI(
    title="Disaster Tweet Classifier API",
    description=(
        "Production-style REST API for binary classification of tweets "
        "into disaster and non-disaster classes."
    ),
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model metadata."""
    service = get_inference_service()
    return ModelInfoResponse(**service.get_model_info())


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    """Predict class for one tweet."""
    service = get_inference_service()
    prediction = service.predict_one(request.text)

    return PredictionResponse(
        label=prediction.label,
        label_name=prediction.label_name,
        probability=prediction.probability,
        threshold=prediction.threshold,
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
    """Predict classes for multiple tweets."""
    service = get_inference_service()
    predictions = service.predict_batch(request.texts)

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                label=prediction.label,
                label_name=prediction.label_name,
                probability=prediction.probability,
                threshold=prediction.threshold,
            )
            for prediction in predictions
        ]
    )
