"""Load plain text, TXT files, and text-based PDFs."""

import re
from os import PathLike
from pathlib import Path
from typing import Union

class InputProcessingError(ValueError):
    """Raised when an input file cannot be read or processed."""


InputSource = Union[str, PathLike[str]]


def _clean_text(text: str) -> str:
    """Trim lines and excess spacing while retaining paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Latin-1 maps every byte, so legacy text files remain readable.
        try:
            return path.read_text(encoding="latin-1")
        except OSError as exc:
            raise InputProcessingError(f"Could not read text file '{path}': {exc}") from exc
    except OSError as exc:
        raise InputProcessingError(f"Could not read text file '{path}': {exc}") from exc


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise InputProcessingError(f"Could not read PDF file '{path}': {exc}") from exc


def load_input(source: InputSource) -> str:
    """Return clean text from a raw string or a .txt/.pdf file path.

    String values are treated as paths when they name an existing file or have
    a .txt/.pdf suffix; other strings are treated as raw document text.
    """
    if isinstance(source, PathLike):
        path = Path(source)
        is_path = True
    elif isinstance(source, str):
        if not source.strip():
            return ""
        path = Path(source)
        is_path = path.exists() or path.suffix.lower() in {".txt", ".pdf"}
    else:
        raise TypeError("Input must be a string or a path-like object.")

    if not is_path:
        return _clean_text(source)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: '{path}'")
    if not path.is_file():
        raise InputProcessingError(f"Input path is not a file: '{path}'")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = _read_txt(path)
    elif suffix == ".pdf":
        text = _read_pdf(path)
    else:
        raise InputProcessingError(
            f"Unsupported file type '{suffix or '[no extension]'}'; expected .txt or .pdf."
        )

    return _clean_text(text)
