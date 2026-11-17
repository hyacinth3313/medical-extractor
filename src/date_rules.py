"""Recognize numeric and English written dates, retaining original offsets."""

import re
from datetime import date


_DAY = r"(?P<day>0?[1-9]|[12]\d|3[01])"
_MONTH = r"(?P<month>0?[1-9]|1[0-2])"
_YEAR = r"(?P<year>\d{4})"
_MONTH_WORD = (
    r"(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_MONTH_NUMBERS = {
    name: number for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_PATTERNS = [
    re.compile(r"(?<![\w/-])" + pattern + r"(?![\w/-])", re.IGNORECASE)
    for pattern in (
        rf"{_DAY}/{_MONTH}/{_YEAR}",
        rf"{_MONTH}-{_DAY}-{_YEAR}",
        rf"{_YEAR}-{_MONTH}-{_DAY}",
        rf"{_DAY}\s+{_MONTH_WORD}\s+{_YEAR}",
        rf"{_MONTH_WORD}\s+{_DAY},?\s+{_YEAR}",
    )
]


def extract_dates(text: str) -> list[dict]:
    """Return calendar-valid DATE entities; no normalization or BIO tagging."""
    entities = {}
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            month = match["month"]
            month_number = int(month) if month.isdigit() else _MONTH_NUMBERS[month[:3].lower()]
            try:
                date(int(match["year"]), month_number, int(match["day"]))
            except ValueError:
                continue
            start, end = match.span()
            entities[(start, end)] = {
                "text": text[start:end], "label": "DATE", "start": start,
                "end": end, "method": "regex",
            }
    return [entities[key] for key in sorted(entities)]
