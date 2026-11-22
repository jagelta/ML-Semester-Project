import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier


def run_model(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    seed: int,
    features: object = None,
) -> np.ndarray:
    # The majority-class baseline never reads the text, so the seed and the
    # feature settings do not change its predictions.
    del seed, features

    model = DummyClassifier(strategy="most_frequent")
    training_reviews = training_data["text"].tolist()
    training_labels = training_data["label"].to_numpy()
    evaluation_reviews = evaluation_data["text"].tolist()

    model.fit(training_reviews, training_labels)
    return model.predict(evaluation_reviews)
