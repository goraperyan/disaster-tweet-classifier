"""Tests for FastAPI application."""

from __future__ import annotations

from dataclasses import dataclass

from disaster_tweet_classifier.api.app import create_app
from fastapi.testclient import TestClient


@dataclass
class DummyPrediction:
    """Dummy prediction object for API tests."""

    label: int
    label_name: str
    probability: float
    threshold: float


class DummyModelService:
    """Dummy inference service for API tests."""

    def predict_one(self, text: str) -> DummyPrediction:
        """Return deterministic prediction."""
        return DummyPrediction(
            label=1,
            label_name="disaster",
            probability=0.9,
            threshold=0.5,
        )


def test_health_endpoint() -> None:
    """Health endpoint should return ok status."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(monkeypatch) -> None:
    """Predict endpoint should return prediction response."""
    from disaster_tweet_classifier.api import app as app_module

    monkeypatch.setattr(
        app_module,
        "get_model_service",
        lambda: DummyModelService(),
    )

    app = app_module.create_app()
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"text": "Forest fire near La Ronge Sask. Canada"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["text"] == "Forest fire near La Ronge Sask. Canada"
    assert payload["label"] == 1
    assert payload["label_name"] == "disaster"
    assert payload["probability"] == 0.9
    assert payload["threshold"] == 0.5


def test_predict_endpoint_rejects_empty_text() -> None:
    """Predict endpoint should reject empty text."""
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"text": ""},
    )

    assert response.status_code == 422
