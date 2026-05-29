from pathlib import Path

import numpy as np
from disaster_tweet_classifier.training.plots import (
    save_confusion_matrix_plot,
    save_precision_recall_curve_plot,
    save_roc_curve_plot,
)


def test_save_training_plots(tmp_path: Path) -> None:
    targets = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.6, 0.9])

    confusion_matrix_path = tmp_path / "confusion_matrix.png"
    roc_curve_path = tmp_path / "roc_curve.png"
    precision_recall_curve_path = tmp_path / "precision_recall_curve.png"

    save_confusion_matrix_plot(
        targets=targets,
        predictions=predictions,
        output_path=confusion_matrix_path,
    )
    save_roc_curve_plot(
        targets=targets,
        probabilities=probabilities,
        output_path=roc_curve_path,
    )
    save_precision_recall_curve_plot(
        targets=targets,
        probabilities=probabilities,
        output_path=precision_recall_curve_path,
    )

    assert confusion_matrix_path.exists()
    assert roc_curve_path.exists()
    assert precision_recall_curve_path.exists()
