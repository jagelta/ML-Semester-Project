"""Helpers shared by the analysis experiments that run on top of the harness.

The MLP tuning, the training-size run, the vocabulary stress test, the n-gram
ablation, and the error analysis all need the same three things: the data loaded
the way the main entry point loads it, the same train/validation/test partitions,
and one way of turning per-seed rows into mean and standard deviation.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .experiment_runner import ExperimentPlan, _split_training_data

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "output"

# Every partition of IMDb has as many positive as negative reviews, so accuracy
# is easy to read (always guessing one class gives 0.5). It is the primary metric
# and the one used to select hyperparameters. The other metrics are recorded too.
PRIMARY_METRIC = "accuracy"
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
EVALUATION_SETS = ("validation", "test")

PREDICTION_ID_COLUMNS = [
    "evaluation_set",
    "model_name",
    "seed",
    "review_id",
    "true_label",
    "predicted_label",
]


def load_reviews(split: str) -> pd.DataFrame:
    """Load one data file exactly as the main entry point does.

    ``review_id`` is the row position inside that file, so the same ID means the
    same review in every experiment.
    """
    return (
        pd.read_parquet(DATA_DIR / f"{split}-00000-of-00001.parquet")
        .reset_index()
        .rename(columns={"index": "review_id"})
    )


def training_and_evaluation_data(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    evaluation_set: str,
    plan: ExperimentPlan,
) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    """Return the training reviews, the evaluation reviews, and the split seed.

    "validation" trains on 90% of the training file and evaluates on the other
    10%. "test" trains on the whole training file and evaluates on the test file.
    Both follow the main harness, so results can be compared across experiments.
    """
    if evaluation_set == "validation":
        train_pool, validation_data = _split_training_data(training_data, plan)
        return train_pool, validation_data, plan.split_seed
    if evaluation_set == "test":
        return training_data, test_data, None
    raise ValueError(f"Unknown evaluation set: {evaluation_set}")


def summarize(
    results: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str] = METRICS,
) -> pd.DataFrame:
    """Mean and standard deviation of each metric over the repeat seeds.

    The standard deviation is the sample standard deviation (n - 1 in the
    denominator). A deterministic model gives the same result for every seed, so
    its standard deviation is zero.
    """
    completed = results
    if "status" in results.columns:
        completed = results.loc[results["status"].eq("completed")]
    grouped = completed.groupby(list(group_columns), sort=False)
    summary = grouped.size().rename("runs").to_frame()
    for metric in metrics:
        summary[f"{metric}_mean"] = grouped[metric].mean()
        summary[f"{metric}_std"] = grouped[metric].std(ddof=1)
    return summary.reset_index()


def compact_predictions(predictions: pd.DataFrame, **condition: object) -> pd.DataFrame:
    """Keep one small row per review: who predicted what, without the review text.

    The text can be joined back from the data files through ``review_id``, which
    keeps the saved prediction files small enough for the repository.
    """
    compact = predictions[PREDICTION_ID_COLUMNS].copy()
    for position, (name, value) in enumerate(condition.items(), start=1):
        compact.insert(position, name, value)
    return compact


def save_table(table: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    if path.suffix == ".parquet":
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False)
    return path
