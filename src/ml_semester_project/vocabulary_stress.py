"""Vocabulary stress test: remove rare words from the reviews, retrain, and evaluate.

Run with:

    uv run python -m ml_semester_project.vocabulary_stress

Question it answers: how well can each model learn when rare words are absent
from the data? The change is applied to BOTH the training and the evaluation
reviews, and every model is retrained at every severity level (the "training and
test data" setting of the project guidelines).

How it works:

* A word is "rare" at level k if it appears in fewer than k training reviews.
  Level 1 is the reference condition: nothing is removed.
* Rare words are deleted from the review text itself, before any model sees it.
  The models keep their normal ``run_model(training_data, evaluation_data, seed)``
  interface, so every registered model is stressed in exactly the same way.
* Document frequencies are counted on the training partition only. A word that
  appears only in evaluation reviews has frequency zero and is removed as well.
* Everything else stays at the reference setting: full training data, unigram
  features, and each model's selected hyperparameters.
"""

from collections import Counter
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
import re
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
    HTML_LINE_BREAK_PATTERN,
    REFERENCE_FEATURES,
    WORD_PATTERN,
    normalize_text,
)

# Severity levels: the minimum number of training reviews a word must appear in
# to stay in the text. The same levels are used for every model. The shared
# vectorizer already ignores words seen in fewer than 5 training reviews, so the
# first level that changes its input is 10.
MIN_DOCUMENT_FREQUENCY_LEVELS = (1, 10, 30, 100, 300, 1000, 3000)
REFERENCE_LEVEL = 1

_WORD = re.compile(WORD_PATTERN)
# Matches an HTML line break or a word. Line breaks are matched first so that the
# "br" inside "<br />" is never treated as a word, as in the shared features.
_BREAK_OR_WORD = re.compile(
    f"(?P<line_break>{HTML_LINE_BREAK_PATTERN})|{WORD_PATTERN}",
    flags=re.IGNORECASE,
)
CONDITION_COLUMNS = [
    "min_document_frequency",
    "kept_vocabulary_size",
    "shared_feature_count",
    "training_words_removed_share",
    "evaluation_words_removed_share",
]


def word_document_frequencies(texts: Iterable[str]) -> Counter[str]:
    """Count, for every word, the number of reviews that contain it.

    Words are found the same way the shared vectorizer finds them: lowercase
    text, HTML line breaks dropped, runs of two or more word characters.
    """
    frequencies: Counter[str] = Counter()
    for text in texts:
        frequencies.update(set(_WORD.findall(normalize_text(text))))
    return frequencies


def remove_words(
    texts: pd.Series,
    kept_words: AbstractSet[str],
) -> tuple[pd.Series, float]:
    """Delete every word that is not in ``kept_words``.

    The rest of each review (kept words, punctuation, spacing) is left untouched.
    Returns the new texts and the share of word occurrences that were deleted.
    """
    counts = {"words": 0, "removed": 0}

    def keep_or_delete(match: re.Match[str]) -> str:
        word = match.group(0)
        if match.group("line_break") is not None:
            return word
        counts["words"] += 1
        if word.lower() in kept_words:
            return word
        counts["removed"] += 1
        return ""

    stressed = pd.Series(
        [_BREAK_OR_WORD.sub(keep_or_delete, text) for text in texts],
        index=texts.index,
        name=texts.name,
    )
    return stressed, counts["removed"] / max(counts["words"], 1)


def apply_rare_word_removal(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    frequencies: Counter[str],
    min_document_frequency: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Remove rare words from both partitions and describe what was removed.

    ``frequencies`` must come from ``training_data`` so that the evaluation
    reviews never influence which words count as rare.
    """
    kept_words = {
        word
        for word, frequency in frequencies.items()
        if frequency >= min_document_frequency
    }
    if min_document_frequency <= REFERENCE_LEVEL:
        stressed_training, stressed_evaluation = training_data, evaluation_data
        training_removed = evaluation_removed = 0.0
    else:
        training_text, training_removed = remove_words(
            training_data["text"], kept_words
        )
        evaluation_text, evaluation_removed = remove_words(
            evaluation_data["text"], kept_words
        )
        stressed_training = training_data.assign(text=training_text)
        stressed_evaluation = evaluation_data.assign(text=evaluation_text)

    # Number of unigram features that the shared vectorizer builds from the
    # stressed training reviews: it also drops words below its own threshold.
    shared_threshold = max(
        min_document_frequency, REFERENCE_FEATURES.min_document_frequency
    )
    shared_feature_count = sum(
        frequency >= shared_threshold for frequency in frequencies.values()
    )
    condition = {
        "min_document_frequency": min_document_frequency,
        "kept_vocabulary_size": len(kept_words),
        "shared_feature_count": shared_feature_count,
        "training_words_removed_share": training_removed,
        "evaluation_words_removed_share": evaluation_removed,
    }
    return stressed_training, stressed_evaluation, condition


def run_vocabulary_stress(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    plan: ExperimentPlan = EXPERIMENT_PLAN,
    levels: tuple[int, ...] = MIN_DOCUMENT_FREQUENCY_LEVELS,
    evaluation_sets: tuple[str, ...] = EVALUATION_SETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every model at every severity level, on validation and then on test."""
    result_rows = []
    prediction_frames = []
    for evaluation_set in evaluation_sets:
        training_reviews, evaluation_reviews, split_seed = training_and_evaluation_data(
            training_data, test_data, evaluation_set, plan
        )
        frequencies = word_document_frequencies(training_reviews["text"])
        for level in levels:
            stressed_training, stressed_evaluation, condition = apply_rare_word_removal(
                training_reviews, evaluation_reviews, frequencies, level
            )
            for model_name in plan.model_names:
                for seed in plan.repeat_seeds:
                    config = ExperimentConfig(model_name, 1.0, seed)
                    started = time.perf_counter()
                    metrics, predictions = run_experiment(
                        config,
                        plan,
                        stressed_training,
                        stressed_evaluation,
                        evaluation_set,
                        split_seed,
                    )
                    seconds = time.perf_counter() - started
                    result_rows.append(
                        {**metrics, **condition, "run_seconds": round(seconds, 1)}
                    )
                    if predictions is not None:
                        prediction_frames.append(
                            compact_predictions(
                                predictions, min_document_frequency=level
                            )
                        )
                    print(
                        f"{evaluation_set} | min document frequency {level} "
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


def prediction_change_rates(predictions: pd.DataFrame) -> pd.DataFrame:
    """Share of reviews whose predicted label differs from the reference run.

    Each stressed run is compared with the reference run of the same model, seed,
    and evaluation set. Accuracy can stay flat while individual predictions flip,
    so this measures stability directly.
    """
    run_keys = ["evaluation_set", "model_name", "seed"]
    reference = predictions.loc[
        predictions["min_document_frequency"].eq(REFERENCE_LEVEL),
        [*run_keys, "review_id", "predicted_label"],
    ].rename(columns={"predicted_label": "reference_label"})
    merged = predictions.merge(reference, on=[*run_keys, "review_id"], how="left")
    merged["prediction_changed"] = merged["predicted_label"].ne(
        merged["reference_label"]
    )
    return (
        merged.groupby([*run_keys, "min_document_frequency"], sort=False)[
            "prediction_changed"
        ]
        .mean()
        .rename("prediction_change_rate")
        .reset_index()
    )


def seed_disagreement_rates(predictions: pd.DataFrame) -> pd.DataFrame:
    """Share of reviews on which two reference runs with different seeds disagree.

    This is the yardstick for the prediction change rate: a stochastic model
    already changes some predictions when only its seed changes, so a stress level
    is destabilizing only to the extent that it changes more predictions than that.
    The value is the mean over all pairs of seeds; it is zero for a deterministic
    model.
    """
    reference = predictions.loc[
        predictions["min_document_frequency"].eq(REFERENCE_LEVEL)
    ]
    rows = []
    for (evaluation_set, model_name), runs in reference.groupby(
        ["evaluation_set", "model_name"], sort=False
    ):
        by_seed = runs.pivot(
            index="review_id", columns="seed", values="predicted_label"
        )
        seeds = list(by_seed.columns)
        pair_rates = [
            by_seed[first].ne(by_seed[second]).mean()
            for position, first in enumerate(seeds)
            for second in seeds[position + 1 :]
        ]
        rows.append(
            {
                "evaluation_set": evaluation_set,
                "model_name": model_name,
                "reference_seed_disagreement_rate": (
                    sum(pair_rates) / len(pair_rates) if pair_rates else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_vocabulary_stress(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Mean and standard deviation per model and level, plus change from reference."""
    group_columns = ["evaluation_set", "model_name", *CONDITION_COLUMNS]
    summary = summarize(results, group_columns, METRICS)
    if summary.empty or predictions.empty:
        return summary

    model_keys = ["evaluation_set", "model_name"]
    reference = summary.loc[
        summary["min_document_frequency"].eq(REFERENCE_LEVEL),
        [*model_keys, f"{PRIMARY_METRIC}_mean"],
    ].rename(columns={f"{PRIMARY_METRIC}_mean": "reference_mean"})
    summary = summary.merge(reference, on=model_keys, how="left")
    summary[f"{PRIMARY_METRIC}_change_from_reference"] = (
        summary[f"{PRIMARY_METRIC}_mean"] - summary["reference_mean"]
    )
    summary = summary.drop(columns="reference_mean")

    changes = prediction_change_rates(predictions)
    change_summary = (
        changes.groupby([*model_keys, "min_document_frequency"], sort=False)[
            "prediction_change_rate"
        ]
        .agg(prediction_change_rate_mean="mean", prediction_change_rate_std="std")
        .reset_index()
    )
    summary = summary.merge(
        change_summary, on=[*model_keys, "min_document_frequency"], how="left"
    )
    return summary.merge(
        seed_disagreement_rates(predictions), on=model_keys, how="left"
    )


def main() -> None:
    results, predictions = run_vocabulary_stress(
        load_reviews("train"), load_reviews("test")
    )
    summary = summarize_vocabulary_stress(results, predictions)
    paths = [
        save_table(results, "vocabulary-stress-results.csv"),
        save_table(summary, "vocabulary-stress-summary.csv"),
        save_table(predictions, "vocabulary-stress-predictions.parquet"),
    ]
    skipped = results["status"].ne("completed").sum()
    print(f"\nFinished {len(results) - skipped} runs; skipped {skipped}.")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
