"""Neural model: a multilayer perceptron (MLP) with one to three hidden layers.

The model reads the shared TF-IDF features (see ``features.py``) and is trained
with scikit-learn's ``MLPClassifier``. The vectorizer and the network live in one
pipeline that is fit on the training partition only, so no information from the
evaluation reviews (vocabulary, IDF weights, labels) can reach the model.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ..features import REFERENCE_FEATURES, FeatureConfig, build_vectorizer

# Settings that are the same in every run. They are not tuned.
#
# Early stopping holds out 10% of the *training* reviews, checks accuracy on them
# after every epoch, stops after four epochs without improvement, and restores the
# best weights. It picks the number of epochs separately for each run, so one
# configuration can be reused for every training size and stress condition.
TRAINING_SETTINGS: dict[str, Any] = {
    "activation": "relu",
    "solver": "adam",
    "batch_size": "auto",  # min(200, number of training reviews)
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 3,
    "max_iter": 50,
    "tol": 1e-4,
}

# Tuning budget: nine configurations. The first one is the starting point and
# each of the others changes exactly one hyperparameter, so the table of results
# shows the effect of depth, width, learning rate, and L2 penalty one at a time.
BASE_PARAMS: dict[str, Any] = {
    "hidden_layer_sizes": (64,),
    "learning_rate_init": 1e-3,
    "alpha": 1e-4,
}
TUNING_CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    BASE_PARAMS,
    {**BASE_PARAMS, "hidden_layer_sizes": (64, 64)},
    {**BASE_PARAMS, "hidden_layer_sizes": (64, 64, 64)},
    {**BASE_PARAMS, "hidden_layer_sizes": (32,)},
    {**BASE_PARAMS, "hidden_layer_sizes": (128,)},
    {**BASE_PARAMS, "learning_rate_init": 3e-4},
    {**BASE_PARAMS, "learning_rate_init": 3e-3},
    {**BASE_PARAMS, "alpha": 1e-2},
    {**BASE_PARAMS, "alpha": 1.0},
)

# The configuration with the best mean validation accuracy in
# output/mlp-tuning-summary.csv (0.9035 over three seeds). Eight of the nine
# configurations are within 0.7 points of each other, which is about the sampling
# error of a 2,500-review validation set, so the smallest network is also the
# safest choice. Re-run `python -m ml_semester_project.mlp_tuning` to reproduce it.
SELECTED_PARAMS: dict[str, Any] = {
    "hidden_layer_sizes": (32,),
    "learning_rate_init": 1e-3,
    "alpha": 1e-4,
}


def fit_model(
    training_data: pd.DataFrame,
    seed: int,
    features: FeatureConfig = REFERENCE_FEATURES,
    params: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Fit the vectorizer and the MLP on the training reviews and return both."""
    model_params = SELECTED_PARAMS if params is None else params
    pipeline = Pipeline(
        [
            ("features", build_vectorizer(features, weighting="tfidf")),
            (
                "mlp",
                MLPClassifier(
                    **TRAINING_SETTINGS,
                    **model_params,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(training_data["text"], training_data["label"].to_numpy())
    return pipeline


def run_model(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    seed: int,
    features: FeatureConfig = REFERENCE_FEATURES,
    params: Mapping[str, Any] | None = None,
) -> np.ndarray:
    pipeline = fit_model(training_data, seed, features, params)
    return pipeline.predict(evaluation_data["text"])
