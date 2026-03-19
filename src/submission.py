"""Save predictions, model metadata, and create submission zips."""

import json
import logging
import zipfile
from pathlib import Path

from src.models import ModelInfo, Prediction

logger = logging.getLogger(__name__)


def save_predictions(predictions: list[Prediction], output_path: Path) -> None:
    """Write predictions to a text file in competition format.

    Format: '{index} {answer}' per line, sorted by index, no header.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_preds = sorted(predictions, key=lambda p: p.index)
    lines = [f"{p.index} {p.answer}" for p in sorted_preds]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %d predictions → %s", len(predictions), output_path)


def save_model_info(model_info: ModelInfo, output_path: Path) -> None:
    """Write model.json in competition format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model_info.model_dump(), indent=2),
        encoding="utf-8",
    )
    logger.info("Saved model info → %s", output_path)


def create_submission_zip(
    predictions_path: Path,
    model_info_path: Path,
    zip_path: Path,
) -> Path:
    """Create result.zip containing predictions file + model.json.

    Args:
        predictions_path: Path to predictions.txt (Task I) or result.txt (Task II).
        model_info_path: Path to model.json.
        zip_path: Output zip file path.

    Returns:
        Path to the created zip file.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, predictions_path.name)
        zf.write(model_info_path, model_info_path.name)
    logger.info("Created submission zip → %s", zip_path)
    return zip_path
