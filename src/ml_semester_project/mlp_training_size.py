"""Training-size experiment for the MLP, saved in its own files.

Run with:

    uv run python -m ml_semester_project.mlp_training_size

This calls the same harness functions as the main entry point (same split, same
ten training fractions, same three seeds) but only for the MLP, and writes
``mlp-training-size-*`` files. The shared all-model outputs of the main entry
point are left untouched, so branches that add other models do not conflict on
generated files. After all models are merged, ``uv run ml-semester-project``
produces the same MLP numbers together with the other models.
"""

from dataclasses import replace

import pandas as pd

from .experiment_runner import (
    EXPERIMENT_PLAN,
    ExperimentPlan,
    run_final_test_evaluations,
    run_validation_experiments,
)
from .experiment_support import (
    METRICS,
    compact_predictions,
    load_reviews,
    save_table,
    summarize,
)

MODEL_NAME = "mlp"


def run_mlp_training_size(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    plan: ExperimentPlan = EXPERIMENT_PLAN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation sweep over training fractions, then the final test evaluation."""
    mlp_plan = replace(plan, model_names=(MODEL_NAME,))
    validation_results, validation_predictions = run_validation_experiments(
        training_data, mlp_plan
    )
    print(f"Finished {len(validation_results)} validation runs.", flush=True)
    test_results, test_predictions = run_final_test_evaluations(
        training_data, test_data, mlp_plan
    )
    print(f"Finished {len(test_results)} test runs.", flush=True)

    results = pd.concat([validation_results, test_results], ignore_index=True)
    predictions = pd.concat(
        [
            compact_predictions(frame, train_fraction=frame["train_fraction"])
            for frame in (validation_predictions, test_predictions)
        ],
        ignore_index=True,
    )
    return results, predictions


def summarize_training_size(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every training fraction."""
    group_columns = ["evaluation_set", "model_name", "train_fraction", "training_rows"]
    return summarize(results, group_columns, METRICS)


def main() -> None:
    results, predictions = run_mlp_training_size(
        load_reviews("train"), load_reviews("test")
    )
    summary = summarize_training_size(results)
    paths = [
        save_table(results, "mlp-training-size-results.csv"),
        save_table(summary, "mlp-training-size-summary.csv"),
        save_table(predictions, "mlp-training-size-predictions.parquet"),
    ]
    print(
        summary[
            ["evaluation_set", "train_fraction", "training_rows", "runs"]
            + ["accuracy_mean", "accuracy_std"]
        ].to_string(index=False)
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
