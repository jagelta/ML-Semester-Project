"""Hyperparameter search for the MLP with a fixed budget, selected on validation data.

Run with:

    uv run python -m ml_semester_project.mlp_tuning

Every configuration in ``models.mlp.TUNING_CONFIGURATIONS`` is trained on the
harness training partition (90% of the training file) with each repeat seed and
scored on the harness validation partition (the other 10%). The test file is
never opened here. The configuration with the highest mean validation accuracy is
the one to record as ``models.mlp.SELECTED_PARAMS``.
"""

import time
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .experiment_runner import (
    EXPERIMENT_PLAN,
    ExperimentPlan,
    _split_training_data,
    _stratified_training_subset,
)
from .experiment_support import PRIMARY_METRIC, load_reviews, save_table, summarize
from .features import REFERENCE_FEATURES
from .models.mlp import BASE_PARAMS, SELECTED_PARAMS, TUNING_CONFIGURATIONS, fit_model

HYPERPARAMETERS = ["hidden_layer_sizes", "learning_rate_init", "alpha"]


def describe_configuration(params: dict[str, Any]) -> dict[str, object]:
    """Flat, CSV-friendly description of one configuration."""
    changed = [name for name in HYPERPARAMETERS if params[name] != BASE_PARAMS[name]]
    hidden_layers = params["hidden_layer_sizes"]
    return {
        "changed_hyperparameter": changed[0] if changed else "none (starting point)",
        "hidden_layers": len(hidden_layers),
        "hidden_layer_sizes": "-".join(str(units) for units in hidden_layers),
        "learning_rate_init": params["learning_rate_init"],
        "alpha": params["alpha"],
    }


def run_tuning(
    training_data: pd.DataFrame,
    plan: ExperimentPlan = EXPERIMENT_PLAN,
    configurations: tuple[dict[str, Any], ...] = TUNING_CONFIGURATIONS,
) -> pd.DataFrame:
    """Train and score every configuration with every repeat seed."""
    train_pool, validation_data = _split_training_data(training_data, plan)
    actual = validation_data["label"].to_numpy()
    rows = []
    for number, params in enumerate(configurations, start=1):
        for seed in plan.repeat_seeds:
            training_subset = _stratified_training_subset(train_pool, 1.0, seed)
            started = time.perf_counter()
            pipeline = fit_model(training_subset, seed, REFERENCE_FEATURES, params)
            fit_seconds = time.perf_counter() - started
            predictions = pipeline.predict(validation_data["text"])
            network = pipeline["mlp"]
            row = {
                "configuration": number,
                **describe_configuration(params),
                "seed": seed,
                "evaluation_set": "validation",
                "split_seed": plan.split_seed,
                "validation_fraction": plan.validation_fraction,
                "training_rows": len(training_subset),
                "input_features": network.coefs_[0].shape[0],
                "trainable_parameters": sum(
                    weights.size for weights in network.coefs_ + network.intercepts_
                ),
                "epochs": network.n_iter_,
                "fit_seconds": round(fit_seconds, 1),
                "accuracy": accuracy_score(actual, predictions),
                "f1_macro": f1_score(actual, predictions, average="macro"),
            }
            rows.append(row)
            print(
                f"configuration {number}/{len(configurations)} "
                f"({row['changed_hyperparameter']}: layers {row['hidden_layer_sizes']}, "
                f"lr {row['learning_rate_init']}, alpha {row['alpha']}) seed {seed}: "
                f"accuracy {row['accuracy']:.4f}, {row['epochs']} epochs, "
                f"{fit_seconds:.0f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_tuning(results: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration, ranked by mean validation accuracy.

    Ties are broken in favor of the configuration listed first, which is the
    starting point and then the smaller changes.
    """
    description_columns = [
        "configuration",
        "changed_hyperparameter",
        "hidden_layers",
        "hidden_layer_sizes",
        "learning_rate_init",
        "alpha",
        "input_features",
        "trainable_parameters",
    ]
    summary = summarize(results, description_columns, ["accuracy", "f1_macro"])
    effort = results.groupby("configuration")[["epochs", "fit_seconds"]].mean()
    summary["epochs_mean"] = summary["configuration"].map(effort["epochs"]).round(1)
    summary["fit_seconds_mean"] = (
        summary["configuration"].map(effort["fit_seconds"]).round(1)
    )
    ranked = summary.sort_values(
        [f"{PRIMARY_METRIC}_mean", "configuration"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    ranked["selected"] = ranked["rank"].eq(1)
    return ranked


def selected_configuration(summary: pd.DataFrame) -> dict[str, Any]:
    """The hyperparameters of the best-ranked configuration."""
    best = summary.loc[summary["selected"]].iloc[0]
    return dict(TUNING_CONFIGURATIONS[int(best["configuration"]) - 1])


def main() -> None:
    results = run_tuning(load_reviews("train"))
    summary = summarize_tuning(results)
    results_path = save_table(results, "mlp-tuning-results.csv")
    summary_path = save_table(summary, "mlp-tuning-summary.csv")

    print()
    print(
        summary[
            [
                "rank",
                "configuration",
                "changed_hyperparameter",
                "hidden_layer_sizes",
                "learning_rate_init",
                "alpha",
                "accuracy_mean",
                "accuracy_std",
                "epochs_mean",
                "fit_seconds_mean",
            ]
        ].to_string(index=False)
    )
    best = selected_configuration(summary)
    print(f"\nBest configuration on validation data: {best}")
    if best != SELECTED_PARAMS:
        print(
            "This differs from models.mlp.SELECTED_PARAMS. Update SELECTED_PARAMS "
            "before running the other experiments."
        )
    print(f"Saved per-run results to {results_path}")
    print(f"Saved the ranked summary to {summary_path}")


if __name__ == "__main__":
    main()
