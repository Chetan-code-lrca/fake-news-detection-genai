# Fake-News Text Classification (ML Baseline)

This repository contains an educational machine-learning workflow that predicts which historical dataset label (`fake` or `real`) an article resembles. It uses text preprocessing, TF-IDF features, and a scikit-learn classifier. **There is no generative-AI/LLM component in the current implementation.**

## What is included

- Raw datasets in `data/raw/`
- A trained classifier in `models/fake_news_model.pkl`
- The fitted TF-IDF vectorizer in `models/tfidf_vectorizer.pkl`
- A Jupyter notebook for loading data, preparing a stratified split, fitting TF-IDF on training text only, training, and evaluating in `notebooks/01_data_loading.ipynb`

There is currently no web application or supported standalone inference package. The main workflow is the notebook. The committed pickle files are historical artifacts from the earlier training workflow; regenerate both artifacts by rerunning the corrected notebook before treating them as the current model.

## Project structure

```text
fake-news-detection-genai/
├── data/
│   └── raw/
│       ├── Fake.csv
│       ├── True.csv
│       └── IFND.csv
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── 01_data_loading.ipynb
└── README.md
```

## Requirements

Use Python 3.10 or newer with:

- pandas
- NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook or JupyterLab

The repository does not include a dependency lock file, so using a virtual environment keeps the setup isolated from other Python projects.

## Run locally

Clone the repository:

```bash
git clone https://github.com/Chetan-code-lrca/fake-news-detection-genai.git
cd fake-news-detection-genai
```

Create and activate a virtual environment.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib jupyter joblib
```

Start Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Open `notebooks/01_data_loading.ipynb` and run the cells in order.

## Data

The repository contains three CSV files under `data/raw/`:

```text
data/raw/Fake.csv
data/raw/True.csv
data/raw/IFND.csv
```

The notebook currently combines `Fake.csv` and `True.csv` and assigns the labels:

```text
Fake.csv → 1
True.csv → 0
```

The notebook reads both datasets with the columns `title`, `text`, `subject`, and `date`.

## Model artifacts

The trained model and fitted vectorizer are stored here:

```text
models/fake_news_model.pkl
models/tfidf_vectorizer.pkl
```

The expected prediction flow is:

```text
News text
   ↓
TF-IDF vectorizer
   ↓
Trained classifier
   ↓
0 = real
1 = fake
```

Use the included vectorizer when preparing text for the saved classifier. Do not fit a new vectorizer on prediction text, because the resulting feature space would not match the model used during training.

### Load the saved artifacts

```python
import pickle

with open("models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
```

For a simple prediction:

```python
text = "Example news article text"
features = vectorizer.transform([text])
prediction = model.predict(features)[0]

print("Fake" if prediction == 1 else "Real")
```

## Validate the notebook structure

The repository's standard-library tests compile every code cell and check that the vectorizer is fitted only after the train/test split:

```bash
python -m unittest discover -s tests -v
```

## Reproducing the workflow

1. Clone the repository.
2. Create and activate the virtual environment.
3. Install the Python dependencies.
4. Open `notebooks/01_data_loading.ipynb`.
5. Run the notebook from the repository's `notebooks/` directory so the relative dataset paths resolve correctly.
6. Follow the notebook's preprocessing and dataset preparation steps before training a new model. The notebook now creates a stratified train/test split before fitting the TF-IDF vocabulary; the test partition is transformed but never used to fit the vectorizer.

## Data provenance and usage

The repository does not currently record authoritative source URLs, version identifiers, or redistribution permissions for `Fake.csv`, `True.csv`, and `IFND.csv`. The notebook uses only `Fake.csv` and `True.csv`; `IFND.csv` is currently unused. Verify each dataset's provenance, terms, and redistribution rights before using it in an organization or redistributing it.

The sample records visible in the notebook are from 2016–2017. This is historical data and does not represent current news sources, topics, or misinformation techniques.

## Limitations and safe interpretation

This is a machine-learning text classifier, not a fact-checking service, source-verification tool, or generative-AI system. The model learns correlations in the dataset's wording and source styles; it does not verify cited evidence, check original documents, evaluate whether a publisher is trustworthy, or determine whether a claim is true.

A `fake` output means only that the text was assigned the dataset's fake label. It is not proof that a real-world claim is false. A `real` output is not proof that a claim is true. Do not use this model to moderate content, make editorial decisions, rank publishers, or inform public safety decisions without independent evidence and domain-specific validation.

The random stratified split in the notebook is a basic evaluation baseline. It does not prevent duplicate-story leakage or source/publisher leakage, and the result should not be treated as out-of-time or out-of-source generalization. A stronger evaluation would deduplicate stories and hold out later dates and/or unseen sources.

Pickle/joblib artifacts can execute code when loaded. Load only artifacts you created or obtained from a trusted, integrity-checked source.
