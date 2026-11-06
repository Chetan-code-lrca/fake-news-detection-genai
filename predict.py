"""Bounded CLI plus compatibility helpers for trusted local news-model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import joblib

from news_classifier import (
    MAX_PREDICTION_CHARS,
    load_artifacts as load_model_dir_artifacts,
    predict_text,
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_artifacts(
    model_path: str | Path | None = None,
    vectorizer_path: str | Path | None = None,
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> tuple[Any, Any]:
    """Load model artifacts from a directory or explicit legacy file paths.

    Explicit paths are supported for users of the earlier offline inference CLI.
    Joblib files are pickle-based: only load artifacts from sources you trust.
    """
    if model_path is None and vectorizer_path is None:
        return load_model_dir_artifacts(model_dir)
    if model_path is None or vectorizer_path is None:
        raise ValueError("Provide both --model and --vectorizer, or neither.")

    model_file = Path(model_path)
    vectorizer_file = Path(vectorizer_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not vectorizer_file.is_file():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_file}")

    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Loaded model does not provide a callable predict() method.")
    if not callable(getattr(vectorizer, "transform", None)):
        raise TypeError("Loaded vectorizer does not provide a callable transform() method.")
    return model, vectorizer


def predict_news(text: str, model: Any, vectorizer: Any) -> str:
    """Compatibility helper returning the dataset label only."""
    label, _numeric_label = predict_text(text, model, vectorizer)
    return label


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify news text with trusted, local model artifacts."
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="News article text (quote it when it contains shell-special characters).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="directory containing fake_news_model.pkl and tfidf_vectorizer.pkl",
    )
    parser.add_argument("--model", type=Path, help="explicit path to a trusted model file")
    parser.add_argument(
        "--vectorizer",
        type=Path,
        help="explicit path to the matching trusted vectorizer file",
    )
    args = parser.parse_args(argv)

    if (args.model is None) != (args.vectorizer is None):
        parser.error("provide both --model and --vectorizer, or use --model-dir")

    text = " ".join(args.text)
    if len(text) > MAX_PREDICTION_CHARS:
        parser.error(f"text exceeds the {MAX_PREDICTION_CHARS:,}-character limit")

    try:
        model, vectorizer = load_artifacts(
            args.model,
            args.vectorizer,
            model_dir=args.model_dir,
        )
        label, numeric_label = predict_text(text, model, vectorizer)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    print(f"Predicted dataset label: {label} ({numeric_label})")
    print("This prediction is not a fact-check or proof that the claim is true or false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
