from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from disaster_tweet_classifier.inference.predictor import find_latest_checkpoint
from disaster_tweet_classifier.models.transformer_classifier import TransformerTweetClassifier


@dataclass(frozen=True)
class ONNXExportMetadata:
    """Metadata for exported ONNX model."""

    pretrained_model_name: str
    checkpoint_path: str
    onnx_path: str
    opset_version: int
    max_length: int
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]]


class TransformerONNXWrapper(torch.nn.Module):
    """Thin wrapper around Lightning model for ONNX export."""

    def __init__(self, model: TransformerTweetClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return logits."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


def load_model_for_onnx_export(
    config: DictConfig,
    checkpoint_path: Path,
) -> TransformerONNXWrapper:
    """Load trained model checkpoint and wrap it for ONNX export."""
    resolved_checkpoint_path = find_latest_checkpoint(checkpoint_path=checkpoint_path)

    model = TransformerTweetClassifier.load_from_checkpoint(
        checkpoint_path=str(resolved_checkpoint_path),
        map_location=torch.device("cpu"),
        pretrained_model_name=config.model.pretrained_model_name,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        freeze_backbone=config.model.freeze_backbone,
    )

    model.to(torch.device("cpu"))
    model.eval()

    wrapper = TransformerONNXWrapper(model=model)
    wrapper.to(torch.device("cpu"))
    wrapper.eval()

    return wrapper


def build_dummy_inputs(config: DictConfig) -> dict[str, torch.Tensor]:
    """Build dummy tokenizer inputs for tracing ONNX graph."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_model_name,
        use_fast=False,
    )

    encoded = tokenizer(
        ["Forest fire near La Ronge Sask. Canada"],
        padding="max_length",
        truncation=True,
        max_length=int(config.model.tokenizer.max_length),
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"].to(torch.device("cpu")),
        "attention_mask": encoded["attention_mask"].to(torch.device("cpu")),
    }


def export_bertweet_to_onnx(
    config: DictConfig,
    checkpoint_path: Path,
    output_path: Path,
    metadata_path: Path,
    opset_version: int = 17,
) -> None:
    """Export trained BERTweet classifier to ONNX format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = load_model_for_onnx_export(
        config=config,
        checkpoint_path=checkpoint_path,
    )

    dummy_inputs = build_dummy_inputs(config=config)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {
            0: "batch_size",
            1: "sequence_length",
        },
        "attention_mask": {
            0: "batch_size",
            1: "sequence_length",
        },
        "logits": {
            0: "batch_size",
        },
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (
                dummy_inputs["input_ids"],
                dummy_inputs["attention_mask"],
            ),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    resolved_checkpoint_path = find_latest_checkpoint(checkpoint_path=checkpoint_path)

    metadata = ONNXExportMetadata(
        pretrained_model_name=str(config.model.pretrained_model_name),
        checkpoint_path=str(resolved_checkpoint_path),
        onnx_path=str(output_path),
        opset_version=opset_version,
        max_length=int(config.model.tokenizer.max_length),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2),
        encoding="utf-8",
    )
