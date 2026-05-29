from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from transformers import AutoModel, get_linear_schedule_with_warmup


class TransformerTweetClassifier(LightningModule):
    """Transformer-based tweet classifier."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_function = nn.CrossEntropyLoss()

        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.validation_accuracy = BinaryAccuracy()
        self.validation_precision = BinaryPrecision()
        self.validation_recall = BinaryRecall()
        self.validation_f1 = BinaryF1Score()
        self.validation_auroc = BinaryAUROC()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run forward pass."""
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**model_inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]

        logits = self.classifier(self.dropout(pooled_output))
        return logits

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        """Run one training step."""
        labels = batch["labels"]
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        loss = self.loss_function(logits, labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_index: int,
    ) -> torch.Tensor:
        """Run one validation step."""
        labels = batch["labels"]
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        loss = self.loss_function(logits, labels)

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

        self.validation_accuracy.update(predictions, labels)
        self.validation_precision.update(predictions, labels)
        self.validation_recall.update(predictions, labels)
        self.validation_f1.update(predictions, labels)
        self.validation_auroc.update(probabilities, labels)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        accuracy = self.validation_accuracy.compute()
        precision = self.validation_precision.compute()
        recall = self.validation_recall.compute()
        f1 = self.validation_f1.compute()
        auroc = self.validation_auroc.compute()

        self.log("val/accuracy", accuracy, prog_bar=False)
        self.log("val/precision", precision, prog_bar=False)
        self.log("val/recall", recall, prog_bar=False)
        self.log("val/f1", f1, prog_bar=True)
        self.log("val/roc_auc", auroc, prog_bar=False)

        self.validation_accuracy.reset()
        self.validation_precision.reset()
        self.validation_recall.reset()
        self.validation_f1.reset()
        self.validation_auroc.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        trainer = self.trainer
        total_steps = trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
