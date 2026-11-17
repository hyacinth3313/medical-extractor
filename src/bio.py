"""Repair BIO labels and turn token spans into classifier entities."""

from collections.abc import Mapping, Sequence


ENTITY_TYPES = {"MEDICINE", "TEST", "SYMPTOM"}


def repair_bio_sequence(labels: Sequence[str]) -> list[str]:
    """Convert orphan or mismatched I-X labels to B-X without changing the input."""
    repaired = []
    previous = "O"
    for label in labels:
        if label != "O":
            prefix, _, entity_type = label.partition("-")
            if prefix not in {"B", "I"} or entity_type not in ENTITY_TYPES:
                raise ValueError(f"Unsupported BIO label: {label!r}")
            if prefix == "I" and previous not in {f"B-{entity_type}", f"I-{entity_type}"}:
                label = f"B-{entity_type}"
        repaired.append(label)
        previous = label
    return repaired


def bio_to_entities(
    text: str, tokens: Sequence[Mapping], labels: Sequence[str]
) -> list[dict]:
    """Repair labels, then collapse each B-X/I-X run using original text offsets.

    A new B label always starts a new mention, even when its type is unchanged.
    Entity text is sliced from the input so internal spacing is preserved.
    """
    if len(tokens) != len(labels):
        raise ValueError("Tokens and labels must have the same length.")
    repaired = repair_bio_sequence(labels)
    entities = []
    active = None
    previous_end = 0
    for token, label in zip(tokens, repaired):
        start, end = token["start"], token["end"]
        if (
            not isinstance(start, int) or not isinstance(end, int)
            or not previous_end <= start < end <= len(text)
            or text[start:end] != token["text"]
        ):
            raise ValueError(f"Invalid token offsets: {token!r}")
        previous_end = end
        if label == "O" or label.startswith("B-"):
            if active is not None:
                active["text"] = text[active["start"]:active["end"]]
                entities.append(active)
                active = None
        if label.startswith("B-"):
            active = {
                "text": "", "label": label[2:], "start": start, "end": end,
                "method": "classifier",
            }
        elif label.startswith("I-"):
            active["end"] = end
    if active is not None:
        active["text"] = text[active["start"]:active["end"]]
        entities.append(active)
    return entities
