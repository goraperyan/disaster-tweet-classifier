"""Pydantic schemas for FastAPI inference service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single tweet prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        description="Tweet text to classify.",
        examples=["Forest fire near La Ronge Sask. Canada"],
    )


class PredictionResponse(BaseModel):
    """Single tweet prediction response."""

    text: str
    label: int
    label_name: str
    probability: float
    threshold: float


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    checkpoint_path: str
    threshold: float
    max_length: int
