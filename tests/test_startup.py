"""Cloud startup must create missing artifacts without changing existing ones."""

import hashlib
import pickle
from pathlib import Path
from unittest.mock import Mock

import pytest

import app
import train


@pytest.fixture(autouse=True)
def clear_startup_cache():
    app.ensure_model.clear()
    yield
    app.ensure_model.clear()


def test_existing_model_is_not_retrained_or_overwritten(monkeypatch, tmp_path):
    model_path = tmp_path / "existing.pkl"
    model_path.write_bytes(b"existing artifact")
    monkeypatch.setattr(app, "MODEL_PATH", model_path)
    training = Mock(side_effect=AssertionError("An existing model must not be retrained"))
    monkeypatch.setattr(train, "main", training)

    assert app.ensure_model() == str(model_path)
    assert model_path.read_bytes() == b"existing artifact"
    training.assert_not_called()


def test_missing_model_trains_once_without_reading_eval(monkeypatch, tmp_path):
    original_model = Path(app.MODEL_PATH)
    original_digest = hashlib.sha256(original_model.read_bytes()).hexdigest() if original_model.exists() else None
    model_path = tmp_path / "models" / "medical_ner.pkl"
    monkeypatch.setattr(app, "MODEL_PATH", model_path)
    monkeypatch.setattr(train, "MODEL_PATH", model_path)
    training = Mock(wraps=train.main)
    monkeypatch.setattr(train, "main", training)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert "eval" not in path.parts, f"Startup accessed evaluation data: {path}"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    app.ensure_model()
    app.ensure_model()
    training.assert_called_once()
    with model_path.open("rb") as stream:
        bundle = pickle.load(stream)
    assert {"model", "vectorizer", "token_vocabularies"} <= bundle.keys()
    assert len(bundle["model"].classes_) == 7
    if original_digest is not None:
        assert hashlib.sha256(original_model.read_bytes()).hexdigest() == original_digest


def test_startup_does_not_report_success_without_an_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "MODEL_PATH", tmp_path / "missing.pkl")
    training = Mock()
    monkeypatch.setattr(train, "main", training)
    with pytest.raises(RuntimeError, match="without creating"):
        app.ensure_model()
    training.assert_called_once()
