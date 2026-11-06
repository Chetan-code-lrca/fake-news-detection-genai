"""Reproducible baseline pipeline for the repository's news-label dataset."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

LABELS = {0: "Real", 1: "Fake"}
MAX_PREDICTION_CHARS = 20_000


def clean_text(text: str) -> str:
    """Apply the simple English-text normalization used by the original notebook."""
    if not isinstance(text, str):
        raise TypeError("News text must be a string.")
    return re.sub(r"[^a-z\s]", " ", text.lower()).strip()


def _read_labeled_csv(path: str | Path, label: int) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    frame = pd.read_csv(source)
    if "text" not in frame.columns:
        raise ValueError(f"{source.name} must contain a 'text' column.")
    frame = frame[["text"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["label"] = label
    return frame


def load_training_data(fake_csv: str | Path, true_csv: str | Path) -> tuple[list[str], list[int]]:
    """Load and deduplicate the two CSV files used by the baseline notebook."""
    fake = _read_labeled_csv(fake_csv, 1)
    real = _read_labeled_csv(true_csv, 0)
    frame = pd.concat([fake, real], ignore_index=True)
    frame["clean_text"] = frame["text"].map(clean_text)
    frame = frame[frame["clean_text"].str.len() > 0]

    # Remove exact-text conflicts before splitting so the same article cannot
    # appear in both classes and contaminate train/test evaluation.
    label_counts = frame.groupby("clean_text")["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)
    frame = frame[~frame["clean_text"].isin(conflicting_texts)]
    frame = frame.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    if frame["label"].nunique() != 2:
        raise ValueError("Training requires both Fake.csv and True.csv classes with non-empty text.")
    if frame["label"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two distinct records for a stratified split.")

    return frame["clean_text"].tolist(), frame["label"].astype(int).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_evaluate(
    fake_csv: str | Path,
    true_csv: str | Path,
    output_dir: str | Path = "models",
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train TF-IDF only on the training split and save artifacts plus metadata."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if not isinstance(random_state, int):
        raise TypeError("random_state must be an integer.")

    texts, labels = load_training_data(fake_csv, true_csv)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(max_features=5_000, stop_words="english")
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1_000, random_state=random_state)
    model.fit(x_train_features, y_train)
    predictions = model.predict(x_test_features)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "fake_news_model.pkl"
    vectorizer_path = destination / "tfidf_vectorizer.pkl"
    metadata_path = destination / "training_metadata.json"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    fake_path = Path(fake_csv)
    true_path = Path(true_csv)
    metadata = {
        "task": "binary news-dataset label classification",
        "label_mapping": {"0": "Real", "1": "Fake"},
        "algorithm": "LogisticRegression",
        "features": "TF-IDF, max_features=5000, English stop words",
        "test_size": test_size,
        "random_state": random_state,
        "stratified_split": True,
        "vectorizer_fit_on_training_data_only": True,
        "records_after_deduplication": len(texts),
        "train_records": len(x_train),
        "test_records": len(x_test),
        "scikit_learn_version": sklearn.__version__,
        "datasets": {
            fake_path.name: _sha256(fake_path),
            true_path.name: _sha256(true_path),
        },
        "metrics": metrics,
        "warning": "A dataset label prediction is not fact-checking or a determination of truth.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_artifacts(model_dir: str | Path = "models") -> tuple[Any, Any]:
    """Load trusted local joblib artifacts. Never load artifacts from unknown sources."""
    directory = Path(model_dir)
    model_path = directory / "fake_news_model.pkl"
    vectorizer_path = directory / "tfidf_vectorizer.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vectorizer_path.is_file():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    if not hasattr(model, "predict") or not hasattr(vectorizer, "transform"):
        raise TypeError("Saved artifacts do not expose the expected prediction interface.")
    return model, vectorizer


def predict_text(text: str, model: Any, vectorizer: Any) -> tuple[str, int]:
    """Return the predicted training-dataset class, with a strict input bound."""
    if not isinstance(text, str):
        raise TypeError("News text must be a string.")
    if not text.strip():
        raise ValueError("News text must not be empty.")
    if len(text) > MAX_PREDICTION_CHARS:
        raise ValueError(f"News text exceeds the {MAX_PREDICTION_CHARS:,}-character limit.")

    features = vectorizer.transform([clean_text(text)])
    prediction = int(model.predict(features)[0])
    if prediction not in LABELS:
        raise ValueError(f"Unsupported predicted label: {prediction}.")
    return LABELS[prediction], prediction
