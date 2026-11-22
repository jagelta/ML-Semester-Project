"""Feature ablation: unigram features versus unigram plus bigram features.

Run with:

    uv run python -m ml_semester_project.ngram_ablation

Question it answers: how much does each model gain from seeing pairs of adjacent
words (for example "not good") in addition to single words?

Only the n-gram range changes between the two arms. Both arms keep an n-gram when
it appears in at least five training reviews, so the unigram+bigram arm contains
every unigram feature plus the bigram features. Training size, input text, and
each model's selected hyperparameters stay at the reference setting.

A model takes part only if its ``run_model`` function accepts a ``features``
argument and builds its vectorizer from it. Other models are recorded as
``not_implemented`` with the reason, exactly as the main harness does.
"""

import time

import pandas as pd

from .experiment_runner import (
    EXPERIMENT_PLAN,
    ExperimentConfig,
    ExperimentPlan,
    run_experiment,
)
from .experiment_support import (
    EVALUATION_SETS,
    METRICS,
    PRIMARY_METRIC,
    compact_predictions,
    load_reviews,
    save_table,
    summarize,
    training_and_evaluation_data,
)
from .features import (
    REFERENCE_FEATURES,
    UNIGRAM_BIGRAM_FEATURES,
    FeatureConfig,
    build_vectorizer,
)

FEATURE_ARMS = (REFERENCE_FEATURES, UNIGRAM_BIGRAM_FEATURES)
CONDITION_COLUMNS = ["features", "min_document_frequency", "shared_feature_count"]


def run_ngram_ablation(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    plan: ExperimentPlan = EXPERIMENT_PLAN,
    arms: tuple[FeatureConfig, ...] = FEATURE_ARMS,
    evaluation_sets: tuple[str, ...] = EVALUATION_SETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every model with every feature arm, on validation and then on test."""
    result_rows = []
    prediction_frames = []
    for evaluation_set in evaluation_sets:
        training_reviews, evaluation_reviews, split_seed = training_and_evaluation_data(
            training_data, test_data, evaluation_set, plan
        )
        for features in arms:
            vectorizer = build_vectorizer(features)
            vectorizer.fit(training_reviews["text"])
            condition = {
                "features": features.name,
                "min_document_frequency": features.min_document_frequency,
                "shared_feature_count": len(vectorizer.vocabulary_),
            }
            for model_name in plan.model_names:
                for seed in plan.repeat_seeds:
                    config = ExperimentConfig(model_name, 1.0, seed)
                    started = time.perf_counter()
                    metrics, predictions = run_experiment(
                        config,
                        plan,
                        training_reviews,
                        evaluation_reviews,
                        evaluation_set,
                        split_seed,
                        features,
                    )
                    seconds = time.perf_counter() - started
                    result_rows.append(
                        {**metrics, **condition, "run_seconds": round(seconds, 1)}
                    )
                    if predictions is not None:
                        prediction_frames.append(
                            compact_predictions(predictions, features=features.name)
                        )
                    print(
                        f"{evaluation_set} | {features.name} "
                        f"({condition['shared_feature_count']:,} shared features) | "
                        f"{model_name} seed {seed}: "
                        + (
                            f"{PRIMARY_METRIC} {metrics[PRIMARY_METRIC]:.4f}, "
                            f"{seconds:.0f}s"
                            if metrics["status"] == "completed"
                            else str(metrics["status"])
                        ),
                        flush=True,
                    )

    results = pd.DataFrame(result_rows)
    leading = ["model_name", "evaluation_set", *CONDITION_COLUMNS, "seed"]
    results = results[leading + [c for c in results.columns if c not in leading]]
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return results, predictions


def summarize_ngram_ablation(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation per model and arm, plus the gain from bigrams.

    The gain is computed seed by seed (same seed, same data, only the features
    differ) and then averaged, so its standard deviation describes how stable the
    effect of adding bigrams is across runs.
    """
    group_columns = ["evaluation_set", "model_name", *CONDITION_COLUMNS]
    summary = summarize(results, group_columns, METRICS)
    if summary.empty:
        return summary
    seconds = (
        results.loc[results["status"].eq("completed")]
        .groupby(["evaluation_set", "model_name", "features"])["run_seconds"]
        .mean()
        .round(1)
        .rename("run_seconds_mean")
        .reset_index()
    )
    summary = summary.merge(
        seconds, on=["evaluation_set", "model_name", "features"], how="left"
    )

    completed = results.loc[results["status"].eq("completed")]
    per_seed = completed.pivot_table(
        index=["evaluation_set", "model_name", "seed"],
        columns="features",
        values=PRIMARY_METRIC,
    )
    arm_names = [REFERENCE_FEATURES.name, UNIGRAM_BIGRAM_FEATURES.name]
    if all(name in per_seed.columns for name in arm_names):
        gain = (per_seed[arm_names[1]] - per_seed[arm_names[0]]).dropna()
        gain_summary = (
            gain.groupby(["evaluation_set", "model_name"])
            .agg(["mean", "std"])
            .rename(
                columns={
                    "mean": f"{PRIMARY_METRIC}_gain_from_bigrams_mean",
                    "std": f"{PRIMARY_METRIC}_gain_from_bigrams_std",
                }
            )
            .reset_index()
        )
        gain_summary["features"] = arm_names[1]
        summary = summary.merge(
            gain_summary, on=["evaluation_set", "model_name", "features"], how="left"
        )
    return summary


def main() -> None:
    results, predictions = run_ngram_ablation(
        load_reviews("train"), load_reviews("test")
    )
    summary = summarize_ngram_ablation(results)
    paths = [
        save_table(results, "ngram-ablation-results.csv"),
        save_table(summary, "ngram-ablation-summary.csv"),
        save_table(predictions, "ngram-ablation-predictions.parquet"),
    ]
    skipped = results["status"].ne("completed").sum()
    print(f"\nFinished {len(results) - skipped} runs; skipped {skipped}.")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
