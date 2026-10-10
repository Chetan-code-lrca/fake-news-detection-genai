"""Offline inference with the repository's saved fake-news model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fake_news_model.pkl"
DEFAULT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
LABEL_NAMES = {0: "Real", 1: "Fake"}


def load_artifacts(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
) -> tuple[Any, Any]:
    """Load the trusted, repository-provided classifier and matching vectorizer."""
    model_file = Path(model_path)
    vectorizer_file = Path(vectorizer_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not vectorizer_file.is_file():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_file}")

    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Loaded model does not provide a callable predict() method")
    if not callable(getattr(vectorizer, "transform", None)):
        raise TypeError("Loaded vectorizer does not provide a callable transform() method")
    return model, vectorizer


def predict_news(text: str, model: Any, vectorizer: Any) -> str:
    """Classify one article as the model's 'Real' or 'Fake' label.

    This is a statistical classification result, not independent fact-checking.
    """
    if not isinstance(text, str):
        raise TypeError("news text must be a string")
    if not text.strip():
        raise ValueError("news text must not be empty")
    if not callable(getattr(model, "predict", None)):
        raise TypeError("model must provide a callable predict() method")
    if not callable(getattr(vectorizer, "transform", None)):
        raise TypeError("vectorizer must provide a callable transform() method")

    features = vectorizer.transform([text])
    predictions = model.predict(features)
    if len(predictions) != 1:
        raise ValueError("model must return exactly one prediction for one article")

    label = predictions[0]
    if label not in LABEL_NAMES:
        raise ValueError(f"model returned unsupported label {label!r}; expected 0 or 1")
    return LABEL_NAMES[int(label)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify a news article with the saved local model."
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="news article text (quote it if it contains shell-special characters)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="path to the trusted trained model (default: models/fake_news_model.pkl)",
    )
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=DEFAULT_VECTORIZER_PATH,
        help="path to its matching TF-IDF vectorizer (default: models/tfidf_vectorizer.pkl)",
    )
    args = parser.parse_args(argv)

    try:
        model, vectorizer = load_artifacts(args.model, args.vectorizer)
        prediction = predict_news(" ".join(args.text), model, vectorizer)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    print(f"Model prediction: {prediction}")
    print("This is a model classification, not proof that the article is true or false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
