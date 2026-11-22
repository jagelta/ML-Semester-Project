"""Small synthetic sentiment data for fast tests (no IMDb files needed)."""

import numpy as np
import pandas as pd

POSITIVE_WORDS = ["great", "wonderful", "loved", "excellent", "brilliant", "enjoyable"]
NEGATIVE_WORDS = ["awful", "boring", "hated", "terrible", "dull", "waste"]
NEUTRAL_WORDS = ["movie", "film", "plot", "actor", "scene", "story", "the", "and"]


def make_reviews(rows: int, seed: int = 0) -> pd.DataFrame:
    """Balanced reviews whose label is decided by the sentiment words they contain."""
    rng = np.random.default_rng(seed)
    records = []
    for review_id in range(rows):
        label = review_id % 2
        sentiment_words = POSITIVE_WORDS if label == 1 else NEGATIVE_WORDS
        words = list(rng.choice(sentiment_words, size=4)) + list(
            rng.choice(NEUTRAL_WORDS, size=8)
        )
        rng.shuffle(words)
        records.append(
            {"review_id": review_id, "text": " ".join(words), "label": label}
        )
    return pd.DataFrame(records)
