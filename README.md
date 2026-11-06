# News Text Classification Baseline

A reproducible TF-IDF + Logistic Regression baseline for classifying news articles using the labels in the repository's `Fake.csv` and `True.csv` datasets.

> **This is not a fact-checking system or a GenAI application.** It does not verify claims against primary sources, query trusted news outlets, use retrieval, or call an LLM. It predicts patterns learned from historical dataset labels.

## What works

- Loads the two labeled CSV files and validates their required `text` column.
- Normalizes English text and removes blank records.
- Removes exact duplicate articles and drops text that appears with conflicting labels before splitting.
- Uses a stratified train/test split.
- Fits TF-IDF vocabulary on the training split only; test text is transformed with that fitted vocabulary.
- Trains Logistic Regression and reports accuracy, per-class precision/recall/F1, and a confusion matrix.
- Saves the classifier, vectorizer, and a JSON training manifest containing dataset SHA-256 hashes and the scikit-learn version.
- Provides a bounded local prediction CLI and a Python API.

## Repository layout

```text
fake-news-detection-genai/
├── data/raw/                 # Existing datasets; see DATA_PROVENANCE.md
├── models/                   # Existing model/vectorizer artifacts
├── notebooks/
│   └── 01_data_loading.ipynb # Original exploratory notebook
├── news_classifier.py        # Reusable loading, training, and inference code
├── train.py                  # Training/evaluation CLI
├── predict.py                # Single-text prediction CLI
├── tests/
│   └── test_news_classifier.py
├── requirements.txt
└── .github/workflows/ci.yml
```

## Requirements and setup

Python 3.10 or newer is recommended.

Linux / macOS:

```bash
git clone https://github.com/Chetan-code-lrca/fake-news-detection-genai.git
cd fake-news-detection-genai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
git clone https://github.com/Chetan-code-lrca/fake-news-detection-genai.git
cd fake-news-detection-genai
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train and evaluate

From the repository root:

```bash
python train.py
```

The default inputs are:

- `data/raw/Fake.csv` → label `1` (Fake)
- `data/raw/True.csv` → label `0` (Real)

The default output directory is `models/`. To use different files or settings:

```bash
python train.py \
  --fake data/raw/Fake.csv \
  --real data/raw/True.csv \
  --output-dir models \
  --test-size 0.2 \
  --random-state 42
```

Training writes:

```text
models/
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
└── training_metadata.json
```

The metadata file records the split parameters, train/test record counts, evaluation metrics, library version, and SHA-256 hashes of the two CSV inputs. Re-running training replaces the corresponding model artifacts.

## Predict from the command line

After training:

```bash
python predict.py "A news article headline and body to classify."
```

The input limit is 20,000 characters. The output is the predicted dataset label, not a probability or a verified truth judgment.

The CLI also supports the repository's existing checked-in artifacts or alternate trusted paths:

```bash
python predict.py "Article text" --model models/fake_news_model.pkl --vectorizer models/tfidf_vectorizer.pkl
```

This preserves the earlier offline-inference workflow. Do not load joblib/pickle files from untrusted sources.

## Use from Python

```python
from news_classifier import load_artifacts, predict_text

model, vectorizer = load_artifacts("models")
label, numeric_label = predict_text(
    "A news article headline and body to classify.",
    model,
    vectorizer,
)
print(label, numeric_label)
```

Only load joblib/pickle artifacts from sources you trust. Deserializing an untrusted model file can execute code.

## Tests

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

GitHub Actions runs the unit tests and Python syntax checks on pushes and pull requests. Tests use small synthetic CSVs; they do not require downloading or retraining on the full dataset.

## Dataset and model provenance

Read [DATA_PROVENANCE.md](DATA_PROVENANCE.md) before reusing the included CSVs or model artifacts, especially in an organization or commercial product. The upstream sources and redistribution permissions are not fully recorded in this repository. Do not assume that public availability grants redistribution or commercial-use rights.

The original notebook is retained as an exploration record. Its earlier workflow fitted TF-IDF on the full dataset before creating the train/test split, which can leak test-set vocabulary information into training. Use `python train.py` for new experiments; the reusable pipeline fits the vectorizer on training data only.

## Limitations and responsible use

- The labels reflect the source datasets, not independently verified ground truth.
- The data is historical and may contain source, topic, time-period, and writing-style bias.
- A random row-level split does not test generalization to future events, new publishers, or unseen source domains. For serious evaluation, add source-aware and time-based holdouts, deduplication against external corpora, and independent expert-reviewed data.
- The current model is English-oriented and uses word-level TF-IDF. It does not understand context, satire, evolving events, image/video evidence, or the truth of a claim.
- The current output is a single binary class. It is not calibrated confidence, provenance, or an explanation.
- Do not use it to moderate speech, rank publishers, suppress reporting, or make consequential decisions without independent verification, bias testing, human review, and a documented appeal process.
- The repository has no supported HTTP service, authentication layer, monitoring, or production deployment configuration.

