from __future__ import annotations

from pathlib import Path

import pytest

from predict import load_artifacts, predict_news


class FakeVectorizer:
    def __init__(self) -> None:
        self.received: list[str] | None = None

    def transform(self, texts: list[str]) -> list[str]:
        self.received = texts
        return texts


class FakeModel:
    def __init__(self, label: int = 1) -> None:
        self.label = label
        self.received: list[str] | None = None

    def predict(self, features: list[str]) -> list[int]:
        self.received = features
        return [self.label]


def test_predict_news_maps_zero_to_real_and_passes_one_text() -> None:
    model = FakeModel(label=0)
    vectorizer = FakeVectorizer()

    result = predict_news("  A sample article  ", model, vectorizer)

    assert result == "Real"
    assert vectorizer.received == ["  A sample article  "]
    assert model.received == ["  A sample article  "]


def test_predict_news_maps_one_to_fake() -> None:
    assert predict_news("Example claim", FakeModel(label=1), FakeVectorizer()) == "Fake"


@pytest.mark.parametrize("text", ["", "   ", "\\n\\t"])
def test_predict_news_rejects_empty_text(text: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        predict_news(text, FakeModel(), FakeVectorizer())


def test_predict_news_rejects_non_string_text() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        predict_news(None, FakeModel(), FakeVectorizer())  # type: ignore[arg-type]


def test_predict_news_rejects_unexpected_model_label() -> None:
    with pytest.raises(ValueError, match="expected 0 or 1"):
        predict_news("Example claim", FakeModel(label=2), FakeVectorizer())


def test_load_artifacts_reports_missing_model(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing-model.pkl"
    vectorizer = tmp_path / "vectorizer.pkl"
    vectorizer.write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_artifacts(missing_model, vectorizer)


def test_repository_artifacts_load_and_predict() -> None:
    model, vectorizer = load_artifacts()

    result = predict_news(
        "The report describes an event and includes details for readers.",
        model,
        vectorizer,
    )

    assert result in {"Real", "Fake"}
