# Fake News Detection using AI & GenAI

This repository contains a machine-learning workflow for classifying news articles as fake or real. The current project focuses on dataset loading, text preprocessing, TF-IDF features, and a trained scikit-learn classifier.

## What is included

- Raw datasets in `data/raw/`
- A trained classifier in `models/fake_news_model.pkl`
- The fitted TF-IDF vectorizer in `models/tfidf_vectorizer.pkl`
- A Jupyter notebook for loading and exploring the data in `notebooks/01_data_loading.ipynb`

There is currently no web application or standalone inference script in the repository. The main workflow is the notebook and the included model artifacts.

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
python -m pip install pandas numpy scikit-learn matplotlib jupyter
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

The notebook currently combines `Fake.csv` and `True.csv`, shuffles the combined data with `random_state=42`, and assigns the labels:

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

## Reproducing the workflow

1. Clone the repository.
2. Create and activate the virtual environment.
3. Install the Python dependencies.
4. Open `notebooks/01_data_loading.ipynb`.
5. Run the notebook from the repository's `notebooks/` directory so the relative dataset paths resolve correctly.
6. Follow the notebook's preprocessing and dataset preparation steps before training a new model.

## Limitations

This is a machine-learning classification project, not a fact-checking service. A prediction can be wrong because of dataset bias, changes in news sources and writing styles, missing context, or content that differs from the training data.

A `fake` prediction should not be treated as proof that a real-world claim is false, and a `real` prediction should not be treated as proof that a claim is true.
