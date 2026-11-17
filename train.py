"""Train a BIO token classifier using data/train only.

Run with: python train.py
Training accuracy uses gold previous labels; it is not held-out performance.
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.features import TOKEN_VOCABULARIES, sequence_features


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
MODEL_PATH = PROJECT_DIR / "models" / "medical_ner.pkl"
LABELS = [
    "B-MEDICINE", "I-MEDICINE", "B-TEST", "I-TEST",
    "B-SYMPTOM", "I-SYMPTOM", "O",
]


def load_training_data():
    """Load paired training documents, checking labels and character offsets."""
    paths = sorted(TRAIN_DIR.glob("*.txt"))
    if not paths:
        raise ValueError(f"No training notes found in {TRAIN_DIR}")
    if {path.stem for path in paths} != {path.stem for path in TRAIN_DIR.glob("*.json")}:
        raise ValueError("Every training note must have a matching TXT/JSON pair.")

    rows, labels = [], []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        annotations = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"Expected a nonempty annotation list for {path.name}")
        previous_end = 0
        for token in annotations:
            start, end = token["start"], token["end"]
            if (
                not isinstance(start, int) or not isinstance(end, int)
                or not previous_end <= start < end <= len(text)
                or text[start:end] != token["text"]
                or token["label"] not in LABELS
            ):
                raise ValueError(f"Invalid token annotation in {path.name}: {token!r}")
            previous_end = end
        document_labels = [token["label"] for token in annotations]
        rows.extend(sequence_features([token["text"] for token in annotations], document_labels))
        labels.extend(document_labels)
    return rows, labels, len(paths)


def main():
    rows, labels, document_count = load_training_data()
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(rows)
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(matrix, labels)
    predictions = model.predict(matrix)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "model": model,
                "vectorizer": vectorizer,
                "token_vocabularies": TOKEN_VOCABULARIES,
                "start_label": "<START>",
                "labels": LABELS,
                "sklearn_version": sklearn.__version__,
            },
            model_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Training documents: {document_count}; tokens: {len(labels)}")
    print(f"Feature matrix: {matrix.shape[0]} rows x {matrix.shape[1]} columns")
    print(f"Training accuracy (gold previous labels): {accuracy_score(labels, predictions):.4%}")
    print("Per-class training metrics (gold previous labels):")
    print(classification_report(labels, predictions, labels=LABELS, digits=4, zero_division=0))
    print(f"Class support: {dict(Counter(labels))}")
    print(f"Saved model and vectorizer: {MODEL_PATH}")


if __name__ == "__main__":
    main()
