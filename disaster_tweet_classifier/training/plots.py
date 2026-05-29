from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def save_confusion_matrix_plot(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """Save confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay.from_predictions(
        y_true=targets,
        y_pred=predictions,
        display_labels=["not_disaster", "disaster"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("Baseline confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)


def save_roc_curve_plot(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save ROC curve plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = RocCurveDisplay.from_predictions(
        y_true=targets,
        y_score=probabilities,
        name="baseline_logreg",
    )
    display.ax_.set_title("Baseline ROC curve")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)


def save_precision_recall_curve_plot(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save precision-recall curve plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = PrecisionRecallDisplay.from_predictions(
        y_true=targets,
        y_score=probabilities,
        name="baseline_logreg",
    )
    display.ax_.set_title("Baseline precision-recall curve")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)
