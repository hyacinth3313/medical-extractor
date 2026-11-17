"""Evaluate the frozen extraction pipeline on the held-out documents.

Run with: python evaluate.py
Exact spans require matching type, start, and end within the same document.
Token F1 compares entity types on gold token boundaries, ignoring B/I prefixes
and excluding O. Overall scores are micro averages, not averages of class F1s.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from src.extractor import extract_entities


EVAL_DIR = Path(__file__).resolve().parent / "data" / "eval"
ENTITY_TYPES = ("MEDICINE", "TEST", "SYMPTOM", "DATE")

# Independently transcribed from the five held-out notes before scoring.
# Dates deliberately have O labels in the supplied token annotations. These
# supplemental gold spans must NEVER be generated from extract_dates output.
# Each entry contains a document SHA256 and (surface, start, end) annotations.
# The digest prevents changed/new notes from silently inheriting incomplete gold.
DATE_GOLD = {
    "note_21.txt": (
        "e5a872c16d583e6626992c0304d36a4f73f49bfc3f82ba2f729ebab157c2056c",
        [("2025-09-03", 8, 18)],
    ),
    "note_22.txt": (
        "3889e6e665c443862114412fa37a049159c7b70c57fa080d16cb877acaf62781",
        [("10/10/2025", 12, 22)],
    ),
    "note_23.txt": (
        "2fc282e0c8614eb9301a7b67c67f48ea48bac3d008bc54574a3320933ea3e323",
        [("Nov 06, 2025", 8, 20)],
    ),
    "note_24.txt": (
        "81f8f1ad26102cf9e9828b9732046c4f253c7c68d43f340852da9603f932af66",
        [("2025-12-09", 12, 22)],
    ),
    "note_25.txt": (
        "fb71d8702e19265901e5cb74b83488a08ce0c71657344e4b40d0f31d446b2e69",
        [("01/12/2026", 8, 18)],
    ),
}


def gold_bio_spans(text: str, tokens: list[dict]) -> list[dict]:
    """Validate gold annotations and reconstruct spans without repairing gold."""
    if not isinstance(tokens, list):
        raise ValueError("Gold annotations must be a list of tokens.")
    entities = []
    active = None
    previous_end = 0
    for token in tokens:
        start, end, label = token["start"], token["end"], token["label"]
        if (
            type(start) is not int or type(end) is not int
            or not previous_end <= start < end <= len(text)
            or text[start:end] != token["text"]
            or text[previous_end:start].strip()
        ):
            raise ValueError(f"Invalid or incomplete gold token offsets: {token!r}")
        previous_end = end
        if label == "O":
            active = None
            continue
        prefix, _, entity_type = label.partition("-")
        if prefix not in {"B", "I"} or entity_type not in ENTITY_TYPES[:3]:
            raise ValueError(f"Invalid gold BIO label: {label!r}")
        if prefix == "B":
            active = {"label": entity_type, "start": start, "end": end}
            entities.append(active)
        elif active is None or active["label"] != entity_type:
            raise ValueError(f"Invalid gold BIO continuation: {token!r}")
        else:
            active["end"] = end
    if text[previous_end:].strip():
        raise ValueError("Gold annotations omit trailing text.")
    return entities


def load_eval_documents() -> list[dict]:
    """Load paired eval files, adding independently annotated date gold spans."""
    paths = sorted(EVAL_DIR.glob("*.txt"))
    if not paths:
        raise ValueError(f"No evaluation documents found in {EVAL_DIR}")
    if {path.stem for path in paths} != {path.stem for path in EVAL_DIR.glob("*.json")}:
        raise ValueError("Every evaluation note must have a matching TXT/JSON pair.")
    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        tokens = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        gold = gold_bio_spans(text, tokens)
        if path.name not in DATE_GOLD:
            raise ValueError(f"Independent DATE gold annotations are needed for {path.name}.")
        digest, dates = DATE_GOLD[path.name]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
            raise ValueError(f"{path.name} changed; review its independent DATE gold annotations.")
        for surface, start, end in dates:
            if text[start:end] != surface:
                raise ValueError(f"Invalid DATE gold offsets in {path.name}.")
            gold.append({"label": "DATE", "start": start, "end": end})
        documents.append({"name": path.name, "text": text, "tokens": tokens, "gold": gold})
    return documents


def token_entity_labels(tokens: list[dict], entities: list[dict]) -> list[str]:
    """Project entity spans onto the fixed gold tokens; uncovered tokens are O."""
    labels = []
    for token in tokens:
        covering = [
            entity for entity in entities
            if entity["start"] < token["end"] and token["start"] < entity["end"]
        ]
        if not covering:
            labels.append("O")
        elif len(covering) == 1 and (
            covering[0]["start"] <= token["start"] < token["end"] <= covering[0]["end"]
        ):
            labels.append(covering[0]["label"])
        else:
            raise ValueError("Entity spans overlap or do not align with gold token boundaries.")
    return labels


def accumulate_counts(counts: dict, gold: list[tuple], predicted: list[tuple]) -> None:
    """Count matching items within one document; tuple[0] is the entity type.

    Counters preserve duplicate predictions as false positives. Accumulating
    separately per document prevents identical offsets in different documents
    from matching each other.
    """
    gold_counts, predicted_counts = Counter(gold), Counter(predicted)
    for category, items in (
        ("tp", gold_counts & predicted_counts),
        ("fp", predicted_counts - gold_counts),
        ("fn", gold_counts - predicted_counts),
    ):
        for item, count in items.items():
            counts[item[0]][category] += count


def precision_recall_f1(counts: Counter) -> tuple[float, float, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def evaluate() -> dict:
    """Run the full frozen pipeline once per held-out document and aggregate."""
    documents = load_eval_documents()
    span_counts = {entity: Counter() for entity in ENTITY_TYPES}
    token_counts = {entity: Counter() for entity in ENTITY_TYPES}
    correct_tokens = total_tokens = 0
    for document in documents:
        text, tokens, gold = document["text"], document["tokens"], document["gold"]
        predicted = extract_entities(text)
        for entity in predicted:
            if (
                entity["label"] not in ENTITY_TYPES
                or not 0 <= entity["start"] < entity["end"] <= len(text)
                or text[entity["start"]:entity["end"]] != entity["text"]
            ):
                raise ValueError(f"Invalid prediction for {document['name']}: {entity!r}")
        accumulate_counts(
            span_counts,
            [(entity["label"], entity["start"], entity["end"]) for entity in gold],
            [(entity["label"], entity["start"], entity["end"]) for entity in predicted],
        )
        gold_labels = token_entity_labels(tokens, gold)
        predicted_labels = token_entity_labels(tokens, predicted)
        accumulate_counts(
            token_counts,
            [(label, index) for index, label in enumerate(gold_labels) if label != "O"],
            [(label, index) for index, label in enumerate(predicted_labels) if label != "O"],
        )
        correct_tokens += sum(gold_label == predicted_label for gold_label, predicted_label in zip(gold_labels, predicted_labels))
        total_tokens += len(tokens)
    return {
        "documents": len(documents), "tokens": total_tokens,
        "span_counts": span_counts, "token_counts": token_counts,
        "token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
    }


def print_table(title: str, counts: dict) -> None:
    print(f"\n{title}")
    print(f"{'Entity':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Gold':>6} {'Pred':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 87)
    overall = sum(counts.values(), Counter())
    for entity, row in [*counts.items(), ("Overall", overall)]:
        precision, recall, f1 = precision_recall_f1(row)
        print(
            f"{entity:<12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} "
            f"{row['tp'] + row['fn']:>6} {row['tp'] + row['fp']:>6} "
            f"{row['tp']:>5} {row['fp']:>5} {row['fn']:>5}"
        )


def main() -> None:
    results = evaluate()
    print(f"Held-out evaluation: {results['documents']} documents, {results['tokens']} tokens")
    print("Hybrid pipeline: classifier + date regex; no fitting or vocabulary updates.")
    print("DATE gold: independent manual spans in evaluate.py, verified against document hashes.")
    print_table("Exact-span metrics (label + start + end must match)", results["span_counts"])
    print_table("Token-level metrics (entity types, B/I ignored, O excluded)", results["token_counts"])
    print(f"\nToken accuracy including O: {results['token_accuracy']:.4%}")
    print("Overall uses micro averaging. Undefined precision/recall/F1 are reported as 0.")


if __name__ == "__main__":
    main()
