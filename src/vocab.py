"""Collect exact entity surface forms from training annotations only.

Vocabulary sets retain the spelling and case in the source notes. Dataset
tokens are words (including internal hyphens/apostrophes) or punctuation;
whitespace is not tokenized. Multiword BIO spans become one vocabulary entry.
"""

import json
from pathlib import Path


_TRAIN_DIR = Path(__file__).resolve().parents[1] / "data" / "train"
_ENTITY_TYPES = ("MEDICINE", "TEST", "SYMPTOM")


def load_training_vocab() -> dict[str, set[str]]:
    """Read paired TXT/JSON training files and return three surface-form sets.

    Evaluation files are never read. Invalid BIO continuations raise ValueError
    rather than silently creating an incorrect vocabulary entry.
    """
    vocab = {entity_type: set() for entity_type in _ENTITY_TYPES}
    for annotation_path in sorted(_TRAIN_DIR.glob("*.json")):
        text = annotation_path.with_suffix(".txt").read_text(encoding="utf-8")
        tokens = json.loads(annotation_path.read_text(encoding="utf-8"))
        entity_type = None
        start = end = 0
        for token in tokens + [{"label": "O"}]:
            label = token["label"]
            if label == "O" or label.startswith("B-"):
                if entity_type is not None:
                    vocab[entity_type].add(text[start:end])
                    entity_type = None
                if label == "O":
                    continue
            prefix, _, current_type = label.partition("-")
            if current_type not in vocab or prefix not in {"B", "I"}:
                raise ValueError(f"Invalid BIO label {label!r} in {annotation_path}")
            if prefix == "B":
                entity_type = current_type
                start = token["start"]
            elif entity_type != current_type:
                raise ValueError(f"Invalid BIO continuation in {annotation_path}")
            end = token["end"]
    return vocab


_vocab = load_training_vocab()
MEDICINE_VOCAB = _vocab["MEDICINE"]
TEST_VOCAB = _vocab["TEST"]
SYMPTOM_VOCAB = _vocab["SYMPTOM"]
