# Medical Extractor

A small machine learning project for extracting medicines, medical tests, symptoms, and dates from English medical documents. MedExtract accepts pasted text, TXT files, and text-based PDFs and provides annotated text, a results table, and JSON export through Streamlit.

The pipeline uses token features and a logistic regression BIO classifier, repairs invalid BIO sequences, collapses entity spans, adds regex date matches, and resolves overlaps. Each result contains `text`, `label`, `start`, `end`, and `method`. Offsets are zero-based, end-exclusive character positions in the text passed to `extract_entities`: `text[start:end] == entity["text"]`.

## Setup

Use Python 3.10 or newer. From the `medical-extractor/` directory, create a virtual environment:

```text
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and launch the UI:

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The UI and evaluator load `models/medical_ner.pkl`. The UI automatically runs training once at startup if the model is missing, using only `data/train/`. An existing model is left untouched. To rebuild deliberately, run `python train.py`; this replaces the saved model. The evaluator does not train automatically, so run `python train.py` first if evaluating a fresh checkout.

## Deploy to Streamlit Community Cloud

Use `medical-extractor/` itself as the GitHub repository root, so `app.py`, `train.py`, `requirements.txt`, `src/`, and `data/` are directly inside the repository. Keep `.venv/`, model artifacts, and local secrets out of Git; `.gitignore` already excludes them. Include all paired TXT/JSON files in `data/train/`: the deployed app needs them to build the missing model.

1. Create an empty GitHub repository named `medical-extractor` with your chosen visibility. Do not initialize the remote with a README if using the commands below.
2. Open a terminal **inside the local `medical-extractor/` folder**. Create a repository there, independently of any Git repository in a parent folder:

   ```text
   git init -b main
   git add app.py train.py evaluate.py requirements.txt README.md .gitignore src tests data
   git commit -m "Prepare MedExtract for Streamlit Cloud"
   ```

3. Replace `YOUR_USERNAME` with your GitHub username, then connect and push:

   ```text
   git remote add origin https://github.com/YOUR_USERNAME/medical-extractor.git
   git push -u origin main
   ```

4. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, and select the GitHub repository and `main` branch. Set the main file path to **`app.py`**. If you instead keep the project inside a `medical-extractor/` repository subfolder, use **`medical-extractor/app.py`**.
5. In **Advanced settings**, select Python **3.12**, a version currently supported by Community Cloud. This project has been tested locally on Python 3.10; check the cloud build logs to validate the deployment environment. No application secrets are required.
6. Deploy. The app installs dependencies from `requirements.txt` and creates `models/medical_ner.pkl` from the training set on its first startup. A restart with no model artifact triggers training again. Once the app loads, test a fictional note and copy its `streamlit.app` URL for sharing; check the app's sharing settings for the audience you want.

Training in the cloud avoids loading a pickle produced by a different scikit-learn installation. Dependency ranges can still select newer package versions, so rerun evaluation if you need to compare a newly generated model with the metrics below. The five runtime dependencies are already declared in `requirements.txt`; no additional system packages are required by this project.

See the official [deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency placement guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Dataset

The dataset contains 25 short, fictional English clinical notes, each with four sentences and paired `.txt`/`.json` files. The split is by document: 80% training and 20% held-out evaluation.

| Split | Location | Documents | Annotated tokens |
|---|---|---:|---:|
| Training | `data/train/` | 20 | 657 |
| Evaluation | `data/eval/` | 5 | 168 |
| Total | | 25 | 825 |

Annotations use `B-`/`I-` labels for MEDICINE, TEST, and SYMPTOM, plus `O`. They cover words and punctuation with exact character offsets and include multiword entities and repeated mentions. There are 698 `O` tokens (84.6% of the dataset), including dates. DATE entities are handled separately by regex.

Training and evaluation documents remain separate. The vocabulary sets in `src/vocab.py`, token features, vectorizer, and classifier are built from training annotations only. Held-out documents must not be used for vocabulary development, rule tuning, or model fitting.

## Evaluation and final metrics

Run the full hybrid pipeline on each held-out document:

```text
python evaluate.py
```

The evaluator loads each evaluation TXT file and its matching JSON annotations, reconstructs gold entity spans, runs `extract_entities`, and prints per-type and overall tables with precision, recall, F1, gold/predicted counts, and TP/FP/FN.

Because the supplied BIO annotations mark dates as `O`, `evaluate.py` includes five independently transcribed DATE gold spans. These are separate from regex predictions and checked against document hashes and text offsets. When adding or changing evaluation notes, independently annotate their dates and update `DATE_GOLD` and its document hashes; the evaluator rejects missing or stale DATE gold entries.

The following results were reproduced on **5 November 2026**, using the saved model on all five held-out documents (168 tokens). These are held-out results, with previous predicted labels supplied during classification.

### Exact-span results

A true positive requires the entity type, start offset, and end offset to match a gold entity in the same document exactly. Boundary or type mismatches count as a false positive and a false negative. Repeated mentions are scored independently by their offsets.

| Entity | Precision | Recall | F1 | Gold entities |
|---|---:|---:|---:|---:|
| MEDICINE | 1.0000 | 0.6000 | 0.7500 | 5 |
| TEST | 0.8333 | 1.0000 | 0.9091 | 5 |
| SYMPTOM | 0.8000 | 0.6667 | 0.7273 | 6 |
| DATE | 1.0000 | 1.0000 | 1.0000 | 5 |
| **Overall** | **0.8947** | **0.8095** | **0.8500** | **21** |

Overall counts: **17 true positives, 2 false positives, and 4 false negatives** across 19 predictions.

### Token-level results

The secondary metric projects gold and predicted spans onto the gold token boundaries, compares entity types while ignoring B/I prefixes, and excludes `O` from precision, recall, and F1. Consequently, it is less strict about entity boundaries than exact-span scoring.

| Entity | Precision | Recall | F1 | Gold entity tokens |
|---|---:|---:|---:|---:|
| MEDICINE | 1.0000 | 0.5000 | 0.6667 | 8 |
| TEST | 0.8462 | 1.0000 | 0.9167 | 11 |
| SYMPTOM | 1.0000 | 0.6250 | 0.7692 | 8 |
| DATE | 1.0000 | 1.0000 | 1.0000 | 16 |
| **Overall** | **0.9474** | **0.8372** | **0.8889** | **43** |

Token accuracy including `O` is **95.8333%**. This is reported separately from token F1 because the large `O` class can make accuracy look optimistic.

Both overall F1 scores use micro averaging: TP, FP, and FN are pooled across entity types and documents before calculating the score. Precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, and F1 is `2TP / (2TP + FP + FN)`. Undefined values are reported as zero.

## Tests

Install pytest in the active environment and run the suite:

```text
python -m pip install pytest
python -m pytest -q
```

The latest run passed **87 tests**, covering dataset integrity and separation, input handling, BIO repair, exact offsets, repeated mentions, date matching, extraction behavior, and first-start model training without evaluation-data access or overwriting an existing model.

## Known limitations

- The synthetic dataset has limited vocabulary. Unseen medicine and test names may be missed; five evaluation notes are too few to establish performance on real clinical documents. The perfect DATE score covers only five mentions.
- Logistic regression makes token-level decisions using the previous-label feature. BIO repair is a heuristic rather than a learned sequence constraint, and prediction errors can affect later labels.
- Negation and historical context are not handled. For example, `no fever` may still yield a SYMPTOM entity for `fever`.
- Extracted mentions do not establish current medicine use or an active diagnosis. Diagnosis inference and clinical decision support are outside the project's scope.
- Only text-based PDFs are supported. Scanned documents, OCR, and handwriting are outside scope.
- Dosage and frequency extraction and date normalization to ISO 8601 are not implemented.
