import hashlib
from pathlib import Path

import pytest

from src import extractor
from src.bio import bio_to_entities, repair_bio_sequence
from src.date_rules import extract_dates
from src.extractor import extract_entities
from src.input_handler import InputProcessingError, load_input


def test_empty_string_returns_empty_text():
    assert load_input("") == ""


def test_multiline_text_is_cleaned_and_paragraphs_are_preserved():
    text = "  First   line\r\nsecond line\n\n\n  New paragraph  "

    assert load_input(text) == "First line\nsecond line\n\nNew paragraph"


def test_nonexistent_text_file_has_clear_error(tmp_path):
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_input(missing_file)


def test_empty_text_file_returns_empty_text(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    assert load_input(empty_file) == ""


def test_text_file_falls_back_for_legacy_encoding(tmp_path):
    text_file = tmp_path / "legacy.txt"
    text_file.write_bytes(b"caf\xe9")

    assert load_input(text_file) == "café"


def test_corrupted_pdf_raises_clear_custom_error(tmp_path):
    pdf_file = tmp_path / "broken.pdf"
    pdf_file.write_bytes(b"not a PDF")

    with pytest.raises(InputProcessingError, match="Could not read PDF file"):
        load_input(pdf_file)


@pytest.mark.parametrize("labels,expected", [
    (["I-TEST", "I-TEST"], ["B-TEST", "I-TEST"]),
    (["O", "I-SYMPTOM"], ["O", "B-SYMPTOM"]),
    (["B-MEDICINE", "I-TEST"], ["B-MEDICINE", "B-TEST"]),
    (["B-TEST", "I-TEST", "I-SYMPTOM"], ["B-TEST", "I-TEST", "B-SYMPTOM"]),
    ([], []),
])
def test_invalid_bio_sequences_are_repaired(labels, expected):
    original = labels.copy()
    assert repair_bio_sequence(labels) == expected
    assert labels == original


def test_bio_spans_preserve_original_spacing_and_offsets():
    text = "é: complete  blood\ncount"
    tokens = [
        {"text": "complete", "start": 3, "end": 11},
        {"text": "blood", "start": 13, "end": 18},
        {"text": "count", "start": 19, "end": 24},
    ]
    assert bio_to_entities(text, tokens, ["I-TEST", "I-TEST", "I-TEST"]) == [{
        "text": text[3:24], "label": "TEST", "start": 3, "end": 24,
        "method": "classifier",
    }]


def test_adjacent_b_labels_are_separate_mentions():
    tokens = [
        {"text": "fever", "start": 0, "end": 5},
        {"text": "fever", "start": 6, "end": 11},
    ]
    entities = bio_to_entities("fever fever", tokens, ["B-SYMPTOM", "B-SYMPTOM"])
    assert [(entity["start"], entity["end"]) for entity in entities] == [(0, 5), (6, 11)]
    assert bio_to_entities("", [], []) == []


def test_bio_rejects_bad_labels_lengths_and_offsets():
    with pytest.raises(ValueError, match="Unsupported BIO"):
        repair_bio_sequence(["B-DATE"])
    with pytest.raises(ValueError, match="same length"):
        bio_to_entities("word", [], ["O"])
    with pytest.raises(ValueError, match="offsets"):
        bio_to_entities("word", [{"text": "word", "start": 0, "end": 3}], ["O"])


@pytest.mark.parametrize("date_text", [
    "15/01/2024", "01-15-2024", "2024-01-15", "15 January 2024",
    "Jan 15, 2024", "September 9, 2024", "9 Sept 2024", "JAN. 15, 2024",
    "29/02/2024",
])
def test_date_formats_have_exact_offsets(date_text):
    text = f"Review: {date_text}."
    assert extract_dates(text) == [{
        "text": date_text, "label": "DATE", "start": 8,
        "end": 8 + len(date_text), "method": "regex",
    }]


@pytest.mark.parametrize("text", [
    "", "31/02/2024", "2023-02-29", "13-15-2024", "January 32, 2024",
    "id2024-01-15", "2024-01-150", "x15/01/2024", "12024-01-15",
])
def test_invalid_dates_and_embedded_identifiers_are_not_matched(text):
    assert extract_dates(text) == []


def test_repeated_dates_keep_both_mentions():
    dates = extract_dates("2024-01-15 and 2024-01-15")
    assert len(dates) == 2
    assert [entity["start"] for entity in dates] == [0, 15]


@pytest.mark.parametrize("text", ["", " ", "\t\r\n"])
def test_empty_extraction_does_not_load_a_model(monkeypatch, text):
    def unexpected_load():
        pytest.fail("Empty text should not load a model")

    monkeypatch.setattr(extractor, "_load_model", unexpected_load)
    assert extractor.extract_entities(text) == []


def test_nonstring_extraction_has_clear_error():
    with pytest.raises(TypeError, match="must be a string"):
        extractor.extract_entities(None)


def test_complete_pipeline_extracts_all_entity_types():
    text = "Paracetamol helped headache. A complete blood count was requested on 15 January 2024."
    entities = extractor.extract_entities(text)
    assert [(entity["text"], entity["label"], entity["method"]) for entity in entities] == [
        ("Paracetamol", "MEDICINE", "classifier"),
        ("headache", "SYMPTOM", "classifier"),
        ("complete blood count", "TEST", "classifier"),
        ("15 January 2024", "DATE", "regex"),
    ]
    for entity in entities:
        assert set(entity) == {"text", "label", "start", "end", "method"}
        assert type(entity["start"]) is type(entity["end"]) is int
        assert text[entity["start"]:entity["end"]] == entity["text"]


def test_pipeline_preserves_unicode_whitespace_and_multiword_offsets():
    text = "  Café: complete  blood\ncount on Jan 15, 2024."
    entities = extractor.extract_entities(text)
    assert [entity["text"] for entity in entities] == ["complete  blood\ncount", "Jan 15, 2024"]
    assert all(text[entity["start"]:entity["end"]] == entity["text"] for entity in entities)


@pytest.mark.parametrize("mention,label", [
    ("Paracetamol", "MEDICINE"),
    ("headache", "SYMPTOM"),
])
def test_pipeline_keeps_repeated_entity_mentions(mention, label):
    text = f"{mention} and {mention}."
    entities = extract_entities(text)
    mentions = [entity for entity in entities if entity["label"] == label]
    second_start = len(mention) + len(" and ")
    assert mentions == [
        {"text": mention, "label": label, "start": 0, "end": len(mention), "method": "classifier"},
        {"text": mention, "label": label, "start": second_start,
         "end": second_start + len(mention), "method": "classifier"},
    ]


def test_pipeline_collapses_complete_blood_count_into_one_test_entity():
    text = "A complete blood count was requested."
    entities = extract_entities(text)
    start = text.index("complete blood count")
    end = start + len("complete blood count")
    overlapping = [entity for entity in entities if entity["start"] < end and entity["end"] > start]
    assert overlapping == [{
        "text": "complete blood count", "label": "TEST", "start": start,
        "end": end, "method": "classifier",
    }]


@pytest.mark.parametrize("text", [
    "Paracetamol helped headache. A complete blood count was requested on 15 January 2024.",
    "  Café: complete  blood\ncount on Jan 15, 2024.",
    "\tParacetamol and Paracetamol.\r\nReview on 2024-01-15.  ",
])
def test_every_pipeline_entity_has_exact_original_character_offsets(text):
    entities = extract_entities(text)
    assert entities, "Offset assertions require nonempty extraction results"
    for entity in entities:
        assert type(entity["start"]) is int
        assert type(entity["end"]) is int
        assert 0 <= entity["start"] < entity["end"] <= len(text)
        assert text[entity["start"]:entity["end"]] == entity["text"]


@pytest.mark.parametrize("date_text", ["2024-01-15", "15 January 2024", "Jan 15, 2024"])
def test_pipeline_extracts_dates_using_the_regex_path(date_text):
    text = f"Review on {date_text}."
    dates = [entity for entity in extract_entities(text) if entity["label"] == "DATE"]
    start = text.index(date_text)
    assert dates == [{
        "text": date_text, "label": "DATE", "start": start,
        "end": start + len(date_text), "method": "regex",
    }]


@pytest.mark.parametrize("text,predicted_labels,expected_spans", [
    pytest.param(
        "Review headache", ["O", "I-SYMPTOM"],
        [("headache", "SYMPTOM")], id="I-after-O",
    ),
    pytest.param(
        "Paracetamol headache", ["B-MEDICINE", "I-SYMPTOM"],
        [("Paracetamol", "MEDICINE"), ("headache", "SYMPTOM")], id="I-after-different-B",
    ),
    pytest.param(
        "complete blood headache", ["B-TEST", "I-TEST", "I-SYMPTOM"],
        [("complete blood", "TEST"), ("headache", "SYMPTOM")], id="I-after-different-I",
    ),
])
def test_pipeline_repairs_invalid_classifier_sequences(monkeypatch, text, predicted_labels, expected_spans):
    # Force otherwise unpredictable classifier errors, retaining the real saved
    # vectorizer, feature extraction, BIO repair, and public extraction function.
    model = extractor._load_model()["model"]
    predictions = iter(predicted_labels)
    calls = []

    def predict(matrix):
        calls.append(matrix.shape)
        return [next(predictions)]

    monkeypatch.setattr(model, "predict", predict)
    entities = extract_entities(text)
    assert len(calls) == len(predicted_labels)
    assert entities == [
        {"text": surface, "label": label, "start": text.index(surface),
         "end": text.index(surface) + len(surface), "method": "classifier"}
        for surface, label in expected_spans
    ]


def test_pipeline_uses_saved_vocab_and_previous_predictions_per_document(monkeypatch):
    seen_features = []

    class RecordingVectorizer:
        def transform(self, rows):
            seen_features.extend(rows)
            return rows

    class FixedModel:
        def predict(self, matrix):
            return [["I-TEST", "I-TEST", "O"][(len(seen_features) - 1) % 3]]

    bundle = {
        "model": FixedModel(), "vectorizer": RecordingVectorizer(),
        "start_label": "<START>",
        "token_vocabularies": {entity: set() for entity in ("MEDICINE", "TEST", "SYMPTOM")},
    }
    monkeypatch.setattr(extractor, "_load_model", lambda: bundle)
    for _ in range(2):
        entities = extractor.extract_entities("Paracetamol beta gamma")
        assert entities == [{
            "text": "Paracetamol beta", "label": "TEST", "start": 0,
            "end": 16, "method": "classifier",
        }]
    assert [row["previous_token_label"] for row in seen_features] == [
        "<START>", "I-TEST", "I-TEST", "<START>", "I-TEST", "I-TEST",
    ]
    assert not seen_features[0]["is_in_medicine_vocab"]


def test_overlap_resolution_prefers_length_then_date_or_classifier():
    def entity(start, end, label="TEST", method="classifier"):
        return {"text": "x" * (end - start), "label": label, "start": start, "end": end, "method": method}

    long_date = entity(0, 10, "DATE", "regex")
    short_classifier = entity(0, 5)
    separate = entity(10, 14)
    assert extractor.resolve_overlaps([short_classifier, long_date, long_date.copy(), separate]) == [long_date, separate]
    long_classifier = entity(0, 12)
    assert extractor.resolve_overlaps([long_date, long_classifier]) == [long_classifier]
    equal_classifier = entity(0, 10)
    assert extractor.resolve_overlaps([equal_classifier, long_date]) == [long_date]
    regex_test = entity(0, 10, "TEST", "regex")
    assert extractor.resolve_overlaps([regex_test, equal_classifier]) == [equal_classifier]


def test_tokenizer_matches_training_convention_without_losing_offsets():
    text = "Café: C-reactive protein on 2024-01-15."
    tokens = extractor.tokenize(text)
    assert [token["text"] for token in tokens] == [
        "Café", ":", "C-reactive", "protein", "on", "2024-01-15", ".",
    ]
    assert all(text[token["start"]:token["end"]] == token["text"] for token in tokens)


def test_missing_or_corrupt_model_has_clear_error(monkeypatch, tmp_path):
    model_path = tmp_path / "missing.pkl"
    monkeypatch.setattr(extractor, "MODEL_PATH", model_path)
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        extractor._load_model.__wrapped__()
    model_path.write_bytes(b"not a model")
    with pytest.raises(extractor.ModelLoadError, match="Could not load trained model"):
        extractor._load_model.__wrapped__()


def test_extraction_preserves_model_and_training_files():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "models" / "medical_ner.pkl"] + sorted((root / "data" / "train").glob("*.*"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    extractor.extract_entities("Paracetamol helped headache on 2024-01-15.")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after
