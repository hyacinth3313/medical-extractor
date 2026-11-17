"""Extract classifier entities and regex dates from unchanged input text."""

import pickle
import re
from functools import lru_cache
from pathlib import Path

from src.bio import bio_to_entities
from src.date_rules import extract_dates
from src.features import token_features


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "medical_ner.pkl"
# Match the training annotation tokenizer, including hyphenated words and dates.
_TOKEN_PATTERN = re.compile(r"\w+(?:['-]\w+)*|[^\w\s]")


class ModelLoadError(RuntimeError):
    """The trained model bundle is unreadable or incomplete."""


@lru_cache(maxsize=1)
def _load_model() -> dict:
    """Load the trusted local pickle once; never fit or overwrite its contents."""
    try:
        with MODEL_PATH.open("rb") as stream:
            bundle = pickle.load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}. Run train.py first.") from exc
    except Exception as exc:
        raise ModelLoadError(f"Could not load trained model from {MODEL_PATH}: {exc}") from exc
    required = {"model", "vectorizer", "token_vocabularies", "start_label"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ModelLoadError(f"Incomplete trained model bundle: {MODEL_PATH}")
    return bundle


def tokenize(text: str) -> list[dict]:
    """Return tokens and half-open character offsets into the original string."""
    return [
        {"text": match[0], "start": match.start(), "end": match.end()}
        for match in _TOKEN_PATTERN.finditer(text)
    ]


def resolve_overlaps(entities: list[dict]) -> list[dict]:
    """Deduplicate and greedily retain longer spans, then return document order.

    At equal length, prefer regex DATE entities; otherwise prefer classifier
    entities over regex entities. Remaining ties use the earliest position.
    Separate mentions with identical text are retained at their own offsets.
    """
    def priority(entity):
        if entity["label"] == "DATE" and entity["method"] == "regex":
            method_rank = 0
        elif entity["label"] != "DATE" and entity["method"] == "classifier":
            method_rank = 1
        else:
            method_rank = 2
        return (
            -(entity["end"] - entity["start"]), method_rank,
            entity["start"], entity["end"], entity["label"], entity["method"],
        )

    selected = []
    for candidate in sorted(entities, key=priority):
        if not any(
            candidate["start"] < entity["end"] and entity["start"] < candidate["end"]
            for entity in selected
        ):
            selected.append(candidate.copy())
    return sorted(selected, key=lambda entity: (entity["start"], entity["end"]))


def extract_entities(text: str) -> list[dict]:
    """Return MEDICINE, TEST, SYMPTOM, and DATE spans in original text order.

    No whitespace normalization is applied: text[start:end] always equals the
    entity's text. Each call starts a fresh document and label history.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    if not text.strip():
        return []

    tokens = tokenize(text)
    bundle = _load_model()
    previous_label = bundle["start_label"]
    labels = []
    for token in tokens:
        features = token_features(
            token["text"], previous_label, vocabularies=bundle["token_vocabularies"]
        )
        matrix = bundle["vectorizer"].transform([features])
        label = str(bundle["model"].predict(matrix)[0])
        labels.append(label)
        previous_label = label

    classifier_entities = bio_to_entities(text, tokens, labels)
    return resolve_overlaps(classifier_entities + extract_dates(text))
