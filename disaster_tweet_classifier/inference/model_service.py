from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from disaster_tweet_classifier.inference.predictor import find_latest_checkpoint
from disaster_tweet_classifier.models.transformer_classifier import TransformerTweetClassifier
from disaster_tweet_classifier.preprocessing.text_cleaning import clean_text


@dataclass(frozen=True)
class PredictionResult:
    """Single prediction result."""

    label: int
    label_name: str
    probability: float
    threshold: float


class BERTweetInferenceService:
    """Inference service for BERTweet disaster tweet classifier."""

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: Path,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.threshold = float(config.training.inference.threshold)
        self.checkpoint_path = find_latest_checkpoint(checkpoint_path=checkpoint_path)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.pretrained_model_name,
            use_fast=False,
        )

        self.model = TransformerTweetClassifier.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            pretrained_model_name=config.model.pretrained_model_name,
            num_labels=config.model.num_labels,
            dropout=config.model.dropout,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_ratio=config.training.warmup_ratio,
            freeze_backbone=config.model.freeze_backbone,
        )

        self.model.to(self.device)
        self.model.eval()

    def predict_one(self, text: str) -> PredictionResult:
        """Predict label for one tweet."""
        probabilities = self.predict_probabilities([text])
        probability = probabilities[0]
        label = int(probability >= self.threshold)

        return PredictionResult(
            label=label,
            label_name=self._label_name(label),
            probability=probability,
            threshold=self.threshold,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Predict labels for multiple tweets."""
        probabilities = self.predict_probabilities(texts)

        return [
            PredictionResult(
                label=int(probability >= self.threshold),
                label_name=self._label_name(int(probability >= self.threshold)),
                probability=probability,
                threshold=self.threshold,
            )
            for probability in probabilities
        ]

    def predict_probabilities(self, texts: list[str]) -> list[float]:
        """Predict disaster probabilities for input texts."""
        cleaned_texts = [clean_text(text=text, config=self.config.preprocessing) for text in texts]

        encoded = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.config.model.tokenizer.max_length,
            return_tensors="pt",
        )

        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

        return probabilities.detach().cpu().tolist()

    def get_model_info(self) -> dict[str, str | float | int]:
        """Return model metadata."""
        return {
            "model_type": "BERTweet",
            "pretrained_model_name": str(self.config.model.pretrained_model_name),
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "threshold": self.threshold,
            "max_length": int(self.config.model.tokenizer.max_length),
        }

    @staticmethod
    def _label_name(label: int) -> str:
        if label == 1:
            return "disaster"
        return "not_disaster"
