from __future__ import annotations

import argparse

from news_classifier import MAX_PREDICTION_CHARS, load_artifacts, predict_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify one news-text input with the saved baseline model.")
    parser.add_argument("text", help="News text to classify.")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    if len(args.text) > MAX_PREDICTION_CHARS:
        parser.error(f"text exceeds the {MAX_PREDICTION_CHARS:,}-character limit")

    model, vectorizer = load_artifacts(args.model_dir)
    label, numeric_label = predict_text(args.text, model, vectorizer)
    print(f"Predicted dataset label: {label} ({numeric_label})")
    print("This prediction is not a fact-check or proof that the claim is true or false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
