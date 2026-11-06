from __future__ import annotations

import argparse
import json

from news_classifier import train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the TF-IDF news-label baseline.")
    parser.add_argument("--fake", default="data/raw/Fake.csv", help="CSV containing fake-labelled examples.")
    parser.add_argument("--real", default="data/raw/True.csv", help="CSV containing real-labelled examples.")
    parser.add_argument("--output-dir", default="models", help="Directory for the model, vectorizer, and metadata.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metadata = train_and_evaluate(
        args.fake,
        args.real,
        args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps({
        "accuracy": metadata["metrics"]["accuracy"],
        "train_records": metadata["train_records"],
        "test_records": metadata["test_records"],
        "output_dir": args.output_dir,
    }, indent=2))
    print("Reminder: this is dataset-label classification, not fact-checking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
