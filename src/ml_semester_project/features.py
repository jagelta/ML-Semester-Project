"""Shared text features for every model in the comparison.

All text models should turn a review into numbers with the same rules, so that a
difference in accuracy comes from the model and not from its input. This module
keeps those rules in one place:

* A review is lowercased and IMDb's HTML line breaks (``<br />``) become spaces.
* A word is a run of two or more letters, digits, or underscores. This is
  scikit-learn's default pattern, so punctuation and one-letter tokens are ignored.
* Stop words are kept, because words such as "not" and "no" carry sentiment.
* An n-gram enters the vocabulary only if it appears in at least
  ``min_document_frequency`` training reviews.
* The vocabulary and the IDF weights are learned from the training partition only.
  Call ``fit_transform`` on training reviews and ``transform`` on evaluation reviews.
"""

from dataclasses import dataclass
import re
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

WORD_PATTERN = r"\b\w\w+\b"
TOKEN_PATTERN = r"(?u)" + WORD_PATTERN
HTML_LINE_BREAK_PATTERN = r"<br\s*/?>"
_HTML_LINE_BREAK = re.compile(HTML_LINE_BREAK_PATTERN, flags=re.IGNORECASE)

Weighting = Literal["tfidf", "count"]


@dataclass(frozen=True)
class FeatureConfig:
    """Feature settings that an experiment is allowed to change.

    ``ngram_range`` (1, 1) uses single words; (1, 2) adds pairs of adjacent words.
    ``min_document_frequency`` is the number of training reviews an n-gram must
    appear in. Five keeps the unigram+bigram vocabulary small enough to train an
    MLP on a laptop CPU in about two minutes.
    """

    ngram_range: tuple[int, int] = (1, 1)
    min_document_frequency: int = 5

    @property
    def name(self) -> str:
        names = {(1, 1): "unigram", (1, 2): "unigram+bigram"}
        return names.get(self.ngram_range, f"ngram{self.ngram_range}")


REFERENCE_FEATURES = FeatureConfig()
UNIGRAM_BIGRAM_FEATURES = FeatureConfig(ngram_range=(1, 2))


def normalize_text(text: str) -> str:
    """Lowercase a review and replace HTML line breaks with spaces."""
    return _HTML_LINE_BREAK.sub(" ", text).lower()


def build_vectorizer(
    features: FeatureConfig = REFERENCE_FEATURES,
    weighting: Weighting = "tfidf",
) -> CountVectorizer:
    """Return an unfitted vectorizer that follows the shared feature rules.

    ``weighting="tfidf"`` gives L2-normalized TF-IDF with sublinear term frequency
    (1 + log tf), so a word repeated twenty times in a long review does not count
    twenty times. ``weighting="count"`` gives raw counts, the classic input for
    multinomial Naive Bayes. Tokens, n-grams, and vocabulary are the same in both.
    """
    shared_settings = {
        "preprocessor": normalize_text,
        "token_pattern": TOKEN_PATTERN,
        "ngram_range": features.ngram_range,
        "min_df": features.min_document_frequency,
    }
    if weighting == "tfidf":
        return TfidfVectorizer(
            sublinear_tf=True,
            dtype=np.float32,
            **shared_settings,
        )
    if weighting == "count":
        return CountVectorizer(**shared_settings)
    raise ValueError(f"Unknown weighting: {weighting}")
