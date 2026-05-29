from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_joblib_artifact(artifact: Any, output_path: Path) -> None:
    """Save Python artifact using joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


def load_joblib_artifact(input_path: Path) -> Any:
    """Load Python artifact using joblib."""
    return joblib.load(input_path)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save dictionary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
