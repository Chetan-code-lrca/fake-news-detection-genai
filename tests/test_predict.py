from pathlib import Path

import pytest

from predict import load_artifacts, predict_news


class FakeVectorizer:
    def __init__(self):
        self.received = None

    def transform(self, texts):
        self.received = texts
        return texts


class FakeModel:
    def __init__(self, label=1):
        self.label = label

    def predict(self, features):
        return [self.label]


def test_predict_news_returns_dataset_label():
    assert predict_news("A verified report for readers.", FakeModel(0), FakeVectorizer()) == "Real"
    assert predict_news("A fabricated report for readers.", FakeModel(1), FakeVectorizer()) == "Fake"


def test_legacy_preprocessor_matches_saved_artifact_training():
    vectorizer = FakeVectorizer()
    predict_news("State-of-the-art, 2026!", FakeModel(1), vectorizer)
    assert vectorizer.received == ["stateoftheart "]


def test_predict_news_rejects_empty_or_oversized_input():
    from news_classifier import MAX_PREDICTION_CHARS

    model, vectorizer = FakeModel(), FakeVectorizer()
    with pytest.raises(ValueError, match="must not be empty"):
        predict_news("   ", model, vectorizer)
    with pytest.raises(ValueError, match="character limit"):
        predict_news("x" * (MAX_PREDICTION_CHARS + 1), model, vectorizer)


def test_explicit_artifact_paths_report_missing_model(tmp_path: Path):
    missing_model = tmp_path / "missing-model.pkl"
    vectorizer = tmp_path / "vectorizer.pkl"
    vectorizer.write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_artifacts(missing_model, vectorizer)


def test_repository_artifacts_load_and_predict():
    model, vectorizer = load_artifacts()
    result = predict_news(
        "The report describes an event and includes details for readers.",
        model,
        vectorizer,
    )
    assert result in {"Real", "Fake"}
