# Dataset and model provenance

This file records what is currently known from the repository itself. It does not grant permission to redistribute any third-party material.

## Files present

| File | Used by the new training CLI? | Notes |
| --- | --- | --- |
| `data/raw/Fake.csv` | Yes | The pipeline assigns label 1 to each usable record. |
| `data/raw/True.csv` | Yes | The pipeline assigns label 0 to each usable record. |
| `data/raw/IFND.csv` | No | The current baseline training pipeline does not load or evaluate this file. |

The repository does not currently record a complete upstream dataset citation, dataset version, author/publisher, or license/redistribution grant for these CSV files. The existing model and vectorizer files are generated artifacts; their exact source dataset hashes and training environment were not recorded when they were originally committed.

## Before organizational or commercial reuse

1. Identify the exact upstream source and version for every CSV file.
2. Read and retain the dataset's license and terms, including attribution and redistribution restrictions.
3. Check whether the dataset contains third-party article text, personal information, or content that the organization is not entitled to republish.
4. Record the source URL, access date, version/commit, license, and SHA-256 digest in this file or a separate dataset manifest.
5. Confirm whether derived model artifacts may be shared or deployed under the same terms.
6. Do not place the CSVs or derived model artifacts into a commercial system until the applicable rights have been reviewed.

The new training command writes SHA-256 hashes for `Fake.csv` and `True.csv` to `models/training_metadata.json`. Hashes identify the exact local inputs; they do not establish source authenticity or licensing rights.

## Model artifact safety

The committed `.pkl` files are serialized Python objects. Only load model and vectorizer files obtained from a trusted source. Deserializing untrusted pickle/joblib files can execute arbitrary code.
