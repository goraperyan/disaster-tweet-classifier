from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., examples=["ok"])


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_type: str
    pretrained_model_name: str
    checkpoint_path: str
    device: str
    threshold: float
    max_length: int


class PredictRequest(BaseModel):
    """Single prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        examples=["Forest fire near La Ronge Sask. Canada"],
    )


class PredictionResponse(BaseModel):
    """Single prediction response."""

    label: int = Field(..., examples=[1])
    label_name: str = Field(..., examples=["disaster"])
    probability: float = Field(..., ge=0.0, le=1.0, examples=[0.91])
    threshold: float = Field(..., ge=0.0, le=1.0, examples=[0.5])


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        examples=[
            [
                "Forest fire near La Ronge Sask. Canada",
                "I love sunny days",
            ]
        ],
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]
