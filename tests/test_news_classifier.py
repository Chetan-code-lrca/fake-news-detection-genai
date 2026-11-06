from __future__ import annotations

import string

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from news_classifier import (
    MAX_PREDICTION_CHARS,
    clean_text,
    load_artifacts,
    load_training_data,
    predict_text,
    train_and_evaluate,
)


def token_for(number: int) -> str:
    alphabet = string.ascii_lowercase
    first = alphabet[number // (26 * 26)]
    second = alphabet[(number // 26) % 26]
    third = alphabet[number % 26]
    return f"feature{first}{second}{third}"


def write_datasets(tmp_path):
    fake = pd.DataFrame({
        "text": [f"fabricated report invented event {token_for(i)}" for i in range(10)]
    })
    real = pd.DataFrame({
        "text": [f"verified report confirmed event {token_for(i + 10)}" for i in range(10)]
    })
    fake_path = tmp_path / "Fake.csv"
    real_path = tmp_path / "True.csv"
    fake.to_csv(fake_path, index=False)
    real.to_csv(real_path, index=False)
    return fake_path, real_path, fake["text"].tolist() + real["text"].tolist()


def test_clean_text_is_consistent_and_rejects_non_text():
    assert clean_text("Reuters: A verified 2026 report!") == "reuters a verified report"
    with pytest.raises(TypeError):
        clean_text(None)


def test_training_fits_vectorizer_only_on_training_split(tmp_path):
    fake_path, real_path, all_texts = write_datasets(tmp_path)
    labels = [1] * 10 + [0] * 10
    indices = list(range(len(labels)))
    _, test_indices = train_test_split(
        indices, test_size=0.25, random_state=42, stratify=labels
    )

    output_dir = tmp_path / "artifacts"
    metadata = train_and_evaluate(
        fake_path, real_path, output_dir, test_size=0.25, random_state=42
    )

    vectorizer = joblib.load(output_dir / "tfidf_vectorizer.pkl")
    test_only_tokens = {
        token_for(i) for i in test_indices
    }
    assert test_only_tokens.isdisjoint(vectorizer.vocabulary_)
    assert metadata["vectorizer_fit_on_training_data_only"] is True
    assert metadata["stratified_split"] is True
    assert metadata["train_records"] + metadata["test_records"] == 20
    assert 0.0 <= metadata["metrics"]["accuracy"] <= 1.0
    assert (output_dir / "training_metadata.json").is_file()
    assert (output_dir / "fake_news_model.pkl").is_file()


def test_training_rejects_invalid_test_size(tmp_path):
    fake_path, real_path, _ = write_datasets(tmp_path)
    with pytest.raises(ValueError, match="test_size"):
        train_and_evaluate(fake_path, real_path, tmp_path / "artifacts", test_size=1.0)


def test_loading_and_prediction_from_saved_artifacts(tmp_path):
    fake_path, real_path, _ = write_datasets(tmp_path)
    train_and_evaluate(fake_path, real_path, tmp_path / "artifacts")
    model, vectorizer = load_artifacts(tmp_path / "artifacts")

    label, numeric_label = predict_text(
        "A fabricated invented report about an event.", model, vectorizer
    )

    assert label in {"Fake", "Real"}
    assert numeric_label in {0, 1}


def test_prediction_rejects_empty_and_oversized_input(tmp_path):
    fake_path, real_path, _ = write_datasets(tmp_path)
    train_and_evaluate(fake_path, real_path, tmp_path / "artifacts")
    model, vectorizer = load_artifacts(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="must not be empty"):
        predict_text(" ", model, vectorizer)
    with pytest.raises(ValueError, match="character limit"):
        predict_text("x" * (MAX_PREDICTION_CHARS + 1), model, vectorizer)


def test_missing_dataset_column_is_reported(tmp_path):
    fake_path = tmp_path / "Fake.csv"
    real_path = tmp_path / "True.csv"
    pd.DataFrame({"headline": ["not the expected field"]}).to_csv(fake_path, index=False)
    pd.DataFrame({"text": ["verified report"]}).to_csv(real_path, index=False)

    with pytest.raises(ValueError, match="text"):
        load_training_data(fake_path, real_path)
