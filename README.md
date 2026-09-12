# Fake News Detection using AI & GenAI

A machine-learning project for experimenting with **fake-news classification, text preprocessing, TF-IDF features, and trained classification models**.

## What is in this repository?

The current repository contains:

- Raw news datasets in `data/raw/`
- A trained fake-news classifier in `models/fake_news_model.pkl`
- Its TF-IDF vectorizer in `models/tfidf_vectorizer.pkl`
- A Jupyter notebook for data loading and exploration in `notebooks/01_data_loading.ipynb`

The checked-in repository does **not currently contain a web application, `requirements.txt`, `package.json`, or a Python inference script**, so this README documents the reproducible notebook/model workflow without inventing an application startup command.

## Repository Structure

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

For the notebook/model workflow, use:

- Python 3.10+
- Git
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- scikit-learn
- matplotlib, if required by later notebook cells

Because this repository currently has no dependency lock file, create an isolated virtual environment before installing packages.

## 1. Clone the Repository

```bash
git clone https://github.com/Chetan-code-lrca/fake-news-detection-genai.git
cd fake-news-detection-genai
```

## 2. Create a Python Environment

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

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the core packages:

```bash
python -m pip install pandas numpy scikit-learn matplotlib jupyter
```

## 3. Run the Notebook

Start Jupyter from the repository root:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Then open:

```text
notebooks/01_data_loading.ipynb
```

Run the notebook cells in order.

## 4. Working with the Included Datasets

The repository currently contains three raw CSV datasets:

```text
data/raw/Fake.csv
data/raw/True.csv
data/raw/IFND.csv
```

The first two files are large, while `IFND.csv` is smaller. If GitHub cloning or storage becomes inconvenient because of dataset size, consider moving large datasets to Git LFS or an external dataset-storage solution in a future revision.

## 5. Included Trained Model

Two serialized artifacts are already checked into the repository:

```text
models/fake_news_model.pkl
models/tfidf_vectorizer.pkl
```

The intended inference flow is:

```text
News article / text
        ↓
Text preprocessing
        ↓
TF-IDF vectorizer
        ↓
Trained classifier
        ↓
Fake / real prediction
```

When loading the model, use the **same preprocessing and vectorizer assumptions used during training**. Do not independently fit a new TF-IDF vectorizer on test text, because that changes the feature representation expected by the trained model.

## 6. Example Model Loading

Once scikit-learn is installed, the serialized artifacts can be loaded with Python/pickle-compatible tooling. For example:

```python
import pickle

with open("models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
```

For an actual prediction, transform input text with the **included** vectorizer before passing it to the model:

```python
text = "Example news article text"
features = vectorizer.transform([text])
prediction = model.predict(features)
print(prediction)
```

The exact label meaning (`fake`, `real`, or another encoding) should be verified from the training notebook/model rather than assumed.

## 7. Reproducing the Project

A clean reproduction workflow is:

1. Clone the repository.
2. Create a Python virtual environment.
3. Install the required Python packages.
4. Open `notebooks/01_data_loading.ipynb`.
5. Inspect the dataset columns and preprocessing steps.
6. Reproduce the training/evaluation workflow as additional notebooks or scripts are added.
7. Save updated model artifacts only when their training process is documented.

## Important Limitation

A fake-news classifier should be treated as a **research/educational classification system**, not as an authoritative fact-checker. Model predictions can be wrong because of dataset bias, distribution changes, misleading writing styles, incomplete context, or adversarial content.

A classification result should not be treated as proof that a real-world news claim is true or false.

## Contributing

```bash
git checkout -b feature/your-feature
```

Make your changes, test the notebook/workflow, then:

```bash
git add .
git commit -m "Describe your change"
git push origin feature/your-feature
```

Open a pull request on GitHub when ready.

## Status

This repository is currently best understood as an **ML experimentation/training repository**. The dataset, trained artifacts, and initial notebook are present; a documented end-to-end application/inference service is not currently part of the checked-in project structure.
