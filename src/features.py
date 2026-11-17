"""Token features for supervised BIO training and later sequential prediction."""

import re
from collections.abc import Mapping, Sequence

from sklearn.feature_extraction import DictVectorizer

from src.vocab import MEDICINE_VOCAB, SYMPTOM_VOCAB, TEST_VOCAB


# A token matches any word in a training entity surface form, ignoring case.
# Keep these lookup sets with the model so later inference uses the same vocab.
TOKEN_VOCABULARIES = {
    entity: frozenset(
        word.casefold()
        for surface in surfaces
        for word in re.findall(r"\w+(?:['-]\w+)*|[^\w\s]", surface)
    )
    for entity, surfaces in (
        ("MEDICINE", MEDICINE_VOCAB),
        ("TEST", TEST_VOCAB),
        ("SYMPTOM", SYMPTOM_VOCAB),
    )
}


def token_features(
    token: str,
    previous_token_label: str = "<START>",
    vocabularies: Mapping | None = None,
) -> dict:
    """Describe one token; supply the previous prediction during inference."""
    vocabularies = TOKEN_VOCABULARIES if vocabularies is None else vocabularies
    lower = token.lower()
    features = {
        "lowercase": lower,
        "is_capitalized": bool(token) and token[0].isupper(),
        "is_all_caps": token.isupper(),
        "contains_digits": any(character.isdigit() for character in token),
        "previous_token_label": previous_token_label,
    }
    for length in range(2, 5):
        features[f"prefix_{length}"] = lower[:length]
        features[f"suffix_{length}"] = lower[-length:]
    for entity in ("MEDICINE", "TEST", "SYMPTOM"):
        features[f"is_in_{entity.lower()}_vocab"] = token.casefold() in vocabularies[entity]
    return features


def sequence_features(tokens: Sequence[str], labels: Sequence[str]) -> list[dict]:
    """Use previous gold BIO labels for one training document, never its next label."""
    if len(tokens) != len(labels):
        raise ValueError("Tokens and labels must have the same length.")
    return [
        token_features(token, "<START>" if index == 0 else labels[index - 1])
        for index, token in enumerate(tokens)
    ]


def feature_matrix(
    tokens: Sequence[str],
    labels: Sequence[str],
    vectorizer: DictVectorizer | None = None,
):
    """Return (sparse matrix, vectorizer), fitting only when none is supplied.

    This convenience function accepts one labeled document. For multiple
    documents, concatenate sequence_features results before fitting a shared
    DictVectorizer, preserving each document's independent label history.
    """
    rows = sequence_features(tokens, labels)
    if not rows:
        raise ValueError("Cannot build a feature matrix from an empty token sequence.")
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        return vectorizer.fit_transform(rows), vectorizer
    return vectorizer.transform(rows), vectorizer
