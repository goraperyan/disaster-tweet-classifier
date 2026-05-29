from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from disaster_tweet_classifier.commands import load_config
from disaster_tweet_classifier.inference.model_service import BERTweetInferenceService


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained BERTweet disaster tweet classifier.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Single tweet text for prediction.",
    )
    input_group.add_argument(
        "--input-path",
        type=Path,
        help="Path to CSV file with at least a 'text' column.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/predictions/predictions.csv"),
        help="Path to save predictions CSV for batch inference.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in input CSV.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Optional ID column name in input CSV.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Path to checkpoint file or checkpoints directory. "
            "If not provided, value from Hydra config is used."
        ),
    )

    return parser.parse_args()


def build_service(checkpoint_path: Path | None = None) -> BERTweetInferenceService:
    """Build inference service from project config."""
    config = load_config(
        overrides=[
            "model=bertweet",
            "training=bertweet",
            "preprocessing=bertweet",
        ]
    )

    resolved_checkpoint_path = checkpoint_path
    if resolved_checkpoint_path is None:
        resolved_checkpoint_path = Path(config.training.inference.checkpoint_path)

    return BERTweetInferenceService(
        config=config,
        checkpoint_path=resolved_checkpoint_path,
    )


def run_single_prediction(service: BERTweetInferenceService, text: str) -> None:
    """Run prediction for a single text and print result."""
    prediction = service.predict_one(text)

    print(
        {
            "text": text,
            "label": prediction.label,
            "label_name": prediction.label_name,
            "probability": prediction.probability,
            "threshold": prediction.threshold,
        }
    )


def run_batch_prediction(
    service: BERTweetInferenceService,
    input_path: Path,
    output_path: Path,
    text_column: str,
    id_column: str,
) -> None:
    """Run batch prediction for CSV input."""
    data = pd.read_csv(input_path)

    if text_column not in data.columns:
        msg = f"Input file must contain '{text_column}' column. Found: {list(data.columns)}"
        raise ValueError(msg)

    texts = data[text_column].fillna("").astype(str).tolist()
    predictions = service.predict_batch(texts)

    output = data.copy()

    output["label"] = [prediction.label for prediction in predictions]
    output["label_name"] = [prediction.label_name for prediction in predictions]
    output["probability"] = [prediction.probability for prediction in predictions]
    output["threshold"] = [prediction.threshold for prediction in predictions]

    if id_column in output.columns:
        ordered_columns = [
            id_column,
            text_column,
            "label",
            "label_name",
            "probability",
            "threshold",
        ]
        other_columns = [column for column in output.columns if column not in ordered_columns]
        output = output[ordered_columns + other_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")


def main() -> None:
    """Run inference entrypoint."""
    args = parse_args()
    service = build_service(checkpoint_path=args.checkpoint_path)

    if args.text is not None:
        run_single_prediction(service=service, text=args.text)
        return

    run_batch_prediction(
        service=service,
        input_path=args.input_path,
        output_path=args.output_path,
        text_column=args.text_column,
        id_column=args.id_column,
    )


if __name__ == "__main__":
    main()
