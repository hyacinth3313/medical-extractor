"""Dataset integrity and training-only vocabulary checks."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from src.vocab import (
    MEDICINE_VOCAB,
    SYMPTOM_VOCAB,
    TEST_VOCAB,
    load_training_vocab,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABELS = {"O"} | {
    f"{prefix}-{entity}"
    for prefix in ("B", "I")
    for entity in ("MEDICINE", "TEST", "SYMPTOM")
}
TOKEN_PATTERN = re.compile(r"\w+(?:['-]\w+)*|[^\w\s]")


def test_document_split_and_token_balance():
    counts = Counter()
    texts = {}
    for split, expected_count in (("train", 20), ("eval", 5)):
        paths = sorted((DATA_DIR / split).glob("*.txt"))
        assert len(paths) == expected_count
        assert {path.stem for path in paths} == {
            path.stem for path in (DATA_DIR / split).glob("*.json")
        }
        texts[split] = {path.read_text(encoding="utf-8") for path in paths}
        assert len(texts[split]) == expected_count
        for path in paths:
            tokens = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            counts.update(token["label"] for token in tokens)
    assert texts["train"].isdisjoint(texts["eval"])
    assert 600 <= sum(counts.values()) <= 1000
    assert counts["O"] > sum(counts.values()) / 2
    assert set(counts) == LABELS


@pytest.mark.parametrize("path", sorted(DATA_DIR.glob("*/*.txt")), ids=lambda p: p.stem)
def test_annotations_cover_tokens_with_exact_offsets_and_valid_bio(path):
    text = path.read_text(encoding="utf-8")
    tokens = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert 4 <= len(re.findall(r"[.!?](?:\s|$)", text)) <= 10
    assert len(tokens) == len(list(TOKEN_PATTERN.finditer(text)))
    previous_label = "O"
    for token, match in zip(tokens, TOKEN_PATTERN.finditer(text)):
        assert set(token) == {"text", "label", "start", "end"}
        assert token["text"] == text[token["start"]:token["end"]] == match[0]
        assert (token["start"], token["end"]) == match.span()
        label = token["label"]
        assert label in LABELS
        if label.startswith("I-"):
            assert previous_label in {"B-" + label[2:], label}
        previous_label = label
    # All numeric and written dates remain outside BIO entity spans.
    date_pattern = r"\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4}|\d{1,2} [A-Z][a-z]+ \d{4}|[A-Z][a-z]{2} \d{1,2}, \d{4}"
    dates = list(re.finditer(date_pattern, text))
    assert dates
    for date in dates:
        assert all(
            token["label"] == "O"
            for token in tokens
            if token["start"] < date.end() and token["end"] > date.start()
        )


def test_vocabulary_contains_complete_training_spans_and_excludes_eval_only_terms():
    vocab = load_training_vocab()
    assert vocab == {
        "MEDICINE": MEDICINE_VOCAB,
        "TEST": TEST_VOCAB,
        "SYMPTOM": SYMPTOM_VOCAB,
    }
    assert "Ferrous sulfate" in MEDICINE_VOCAB
    assert "complete blood count" in TEST_VOCAB
    assert "shortness of breath" in SYMPTOM_VOCAB
    assert "Ferrous" not in MEDICINE_VOCAB
    assert "Doxycycline" not in MEDICINE_VOCAB
    assert "Oral rehydration salts" not in MEDICINE_VOCAB
    assert "productive cough" not in SYMPTOM_VOCAB
    assert "abdominal cramps" not in SYMPTOM_VOCAB


def test_vocabulary_loader_reads_only_training_files(monkeypatch):
    read_text = Path.read_text
    seen = []

    def track_read(path, *args, **kwargs):
        seen.append(path.resolve())
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", track_read)
    load_training_vocab()
    assert len(seen) == 40
    assert all(path.parent == (DATA_DIR / "train").resolve() for path in seen)
