"""Error analysis: which reviews do the models get wrong, and why?

Run after the vocabulary stress test and the n-gram ablation, because it reads
the predictions they saved:

    uv run python -m ml_semester_project.error_analysis

Three properties are measured for every evaluation review. They are computed
from the text alone, with rules fixed before looking at any model error:

* ``negation``: negation cues ("not", "never", "n't", ...) per 100 words.
* ``rare_words``: share of the review's words that the reference features cannot
  use, because they appear in fewer than five training reviews (or in none).
* ``mixed_polarity``: how evenly the review mixes words typical of positive
  reviews with words typical of negative reviews (0 = one-sided, 0.5 = evenly
  mixed). The two word lists are learned from the training labels only.

For each property the evaluation reviews are split into four groups of equal
size, from the lowest values (Q1) to the highest (Q4), and the error rate of every
model is reported per group, as mean and standard deviation over the repeat
seeds. A property matters when the error rate grows from Q1 to Q4.

The same table is produced for the unigram+bigram arm of the ablation, which
links the error categories to a quantitative result: if bigrams help with
negation, the error rate should drop most in the high-negation group.
"""

from collections import Counter
import math
import re

import numpy as np
import pandas as pd

from .experiment_runner import EXPERIMENT_PLAN, ExperimentPlan
from .experiment_support import (
    EVALUATION_SETS,
    OUTPUT_DIR,
    load_reviews,
    save_table,
    training_and_evaluation_data,
)
from .features import (
    HTML_LINE_BREAK_PATTERN,
    REFERENCE_FEATURES,
    UNIGRAM_BIGRAM_FEATURES,
    WORD_PATTERN,
    normalize_text,
)
from .vocabulary_stress import REFERENCE_LEVEL, word_document_frequencies

NEGATION_CUES = (
    "not",
    "no",
    "never",
    "none",
    "nothing",
    "nobody",
    "nowhere",
    "neither",
    "nor",
    "cannot",
    "without",
    "hardly",
    "barely",
    "scarcely",
)
# A polarity word must appear in at least this many training reviews, so that its
# class balance is a stable estimate, and each list keeps the most one-sided words.
POLARITY_MIN_DOCUMENT_FREQUENCY = 100
POLARITY_LEXICON_SIZE = 300

CATEGORIES = {
    "negation": "negation_rate",
    "rare_words": "rare_word_share",
    "mixed_polarity": "mixed_polarity_share",
}
GROUP_LABELS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
REFERENCE_CONDITION = "reference"
BIGRAM_CONDITION = UNIGRAM_BIGRAM_FEATURES.name
BASELINE_MODEL = "control"

# Representative errors are short enough to read and quote in the report.
EXAMPLES_PER_CATEGORY = 5
EXAMPLE_WORD_RANGE = (40, 150)

_WORD = re.compile(WORD_PATTERN)
_NEGATION = re.compile(r"\b(?:" + "|".join(NEGATION_CUES) + r")\b|\b\w+n['’]t\b")
_LINE_BREAK = re.compile(HTML_LINE_BREAK_PATTERN, flags=re.IGNORECASE)


def build_polarity_lexicon(
    training_data: pd.DataFrame,
    size: int | None = None,
    min_document_frequency: int | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Words most typical of positive and of negative training reviews.

    Each word is scored by the log-odds of appearing in a positive review versus a
    negative review (with add-one smoothing). The ``size`` highest scores form the
    positive list and the ``size`` lowest scores form the negative list.
    """
    if size is None:
        size = POLARITY_LEXICON_SIZE
    if min_document_frequency is None:
        min_document_frequency = POLARITY_MIN_DOCUMENT_FREQUENCY
    positive = training_data.loc[training_data["label"].eq(1), "text"]
    negative = training_data.loc[training_data["label"].eq(0), "text"]
    positive_frequency = word_document_frequencies(positive)
    negative_frequency = word_document_frequencies(negative)

    scores = {}
    for word in positive_frequency.keys() | negative_frequency.keys():
        in_positive = positive_frequency.get(word, 0)
        in_negative = negative_frequency.get(word, 0)
        if in_positive + in_negative < min_document_frequency:
            continue
        scores[word] = math.log((in_positive + 1) / (len(positive) + 2)) - math.log(
            (in_negative + 1) / (len(negative) + 2)
        )
    ranked = sorted(scores, key=lambda word: (scores[word], word))
    size = min(size, len(ranked) // 2)
    return frozenset(ranked[len(ranked) - size :]), frozenset(ranked[:size])


def describe_reviews(
    evaluation_data: pd.DataFrame,
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """Measure the three properties for every evaluation review.

    The training reviews are used only to decide which words are rare and which
    words signal each class; the evaluation labels are never used.
    """
    frequencies = word_document_frequencies(training_data["text"])
    positive_words, negative_words = build_polarity_lexicon(training_data)
    rare_threshold = REFERENCE_FEATURES.min_document_frequency

    rows = []
    for review_id, text in zip(evaluation_data["review_id"], evaluation_data["text"]):
        normalized = normalize_text(text)
        words = _WORD.findall(normalized)
        word_count = max(len(words), 1)
        negations = _NEGATION.findall(normalized)
        rare = [word for word in words if frequencies.get(word, 0) < rare_threshold]
        positive = [word for word in words if word in positive_words]
        negative = [word for word in words if word in negative_words]
        polarity_words = len(positive) + len(negative)
        rows.append(
            {
                "review_id": review_id,
                "word_count": len(words),
                "negation_cues": len(negations),
                "negation_rate": 100 * len(negations) / word_count,
                "rare_word_share": len(rare) / word_count,
                "positive_cue_words": len(positive),
                "negative_cue_words": len(negative),
                "mixed_polarity_share": (
                    min(len(positive), len(negative)) / polarity_words
                    if polarity_words
                    else np.nan
                ),
                "negation_examples": ", ".join(_most_common(negations)),
                "rare_word_examples": ", ".join(_most_common(rare)),
                "positive_cue_examples": ", ".join(_most_common(positive)),
                "negative_cue_examples": ", ".join(_most_common(negative)),
            }
        )
    described = pd.DataFrame(rows)
    for category, measure in CATEGORIES.items():
        described[f"{category}_group"] = quartile_groups(described[measure])
    return described


def _most_common(words: list[str], limit: int = 8) -> list[str]:
    return [word for word, _ in Counter(words).most_common(limit)]


def quartile_groups(values: pd.Series, seed: int = 0) -> pd.Series:
    """Four groups of equal size, from the lowest values to the highest.

    Ranks are used instead of raw values so that ties cannot produce groups of
    different sizes. Tied reviews are ordered at random with a fixed seed, never
    by their position in the file: the data files are sorted by label, so file
    order would push negative reviews into the lower group. Reviews without a
    value (no polarity words) get no group.
    """
    shuffled = values.sample(frac=1.0, random_state=seed)
    ranks = shuffled.rank(method="first").reindex(values.index)
    if ranks.notna().sum() < len(GROUP_LABELS):
        return pd.Series(np.nan, index=values.index, dtype="object")
    return pd.qcut(ranks, len(GROUP_LABELS), labels=GROUP_LABELS).astype("object")


def collect_predictions(
    stress_predictions: pd.DataFrame,
    ablation_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Predictions of the reference condition and of the unigram+bigram arm.

    The reference predictions come from the unstressed level of the vocabulary
    stress test, which includes every implemented model.
    """
    columns = [
        "evaluation_set",
        "condition",
        "model_name",
        "seed",
        "review_id",
        "true_label",
        "predicted_label",
    ]
    reference = stress_predictions.loc[
        stress_predictions["min_document_frequency"].eq(REFERENCE_LEVEL)
    ].assign(condition=REFERENCE_CONDITION)
    frames = [reference[columns]]
    if not ablation_predictions.empty:
        bigram = ablation_predictions.loc[
            ablation_predictions["features"].eq(BIGRAM_CONDITION)
        ].assign(condition=BIGRAM_CONDITION)
        frames.append(bigram[columns])
    predictions = pd.concat(frames, ignore_index=True)
    predictions["error"] = predictions["predicted_label"].ne(predictions["true_label"])
    return predictions


def error_rates_by_group(
    predictions: pd.DataFrame,
    described: pd.DataFrame,
) -> pd.DataFrame:
    """Error rate of every model in every group, as mean and std over seeds."""
    run_keys = ["evaluation_set", "condition", "model_name"]
    merged = predictions.merge(described, on="review_id", how="left")
    merged["positive"] = merged["true_label"].eq(1)
    tables = []
    groupings = [("all", None), *[(name, f"{name}_group") for name in CATEGORIES]]
    for category, group_column in groupings:
        frame = (
            merged.assign(group="All reviews")
            if group_column is None
            else (
                merged.dropna(subset=[group_column]).rename(
                    columns={group_column: "group"}
                )
            )
        )
        measure = CATEGORIES.get(category)
        per_seed = (
            frame.groupby([*run_keys, "group", "seed"], sort=False)["error"]
            .mean()
            .reset_index()
        )
        table = (
            per_seed.groupby([*run_keys, "group"], sort=False)["error"]
            .agg(runs="size", error_rate_mean="mean", error_rate_std="std")
            .reset_index()
        )
        # A review counts as unstable when the repeat seeds do not all agree.
        per_review = frame.groupby([*run_keys, "group", "review_id"], sort=False).agg(
            unstable=("predicted_label", lambda labels: labels.nunique() > 1),
            positive=("positive", "first"),
            measure=(measure, "first") if measure else ("error", "size"),
        )
        group_facts = per_review.groupby(level=[*run_keys, "group"], sort=False).agg(
            reviews=("unstable", "size"),
            positive_share=("positive", "mean"),
            seed_disagreement_rate=("unstable", "mean"),
            measure_min=("measure", "min"),
            measure_max=("measure", "max"),
        )
        table = table.merge(group_facts.reset_index(), on=[*run_keys, "group"])
        if measure is None:
            table[["measure_min", "measure_max"]] = np.nan
        table.insert(3, "category", category)
        table.insert(4, "measure", measure or "")
        tables.append(table)

    result = pd.concat(tables, ignore_index=True)
    reference = result.loc[
        result["condition"].eq(REFERENCE_CONDITION),
        ["evaluation_set", "model_name", "category", "group", "error_rate_mean"],
    ].rename(columns={"error_rate_mean": "reference_error_rate"})
    result = result.merge(
        reference, on=["evaluation_set", "model_name", "category", "group"], how="left"
    )
    result["error_rate_change_from_reference"] = (
        result["error_rate_mean"] - result["reference_error_rate"]
    )
    result = result.drop(columns="reference_error_rate")
    group_order = {label: position for position, label in enumerate(GROUP_LABELS)}
    result["group_order"] = result["group"].map(group_order).fillna(-1)
    result = result.sort_values(
        ["evaluation_set", "condition", "model_name", "category", "group_order"],
        kind="stable",
    ).drop(columns="group_order")
    column_order = [
        "evaluation_set",
        "condition",
        "model_name",
        "category",
        "measure",
        "group",
        "reviews",
        "measure_min",
        "measure_max",
        "positive_share",
        "runs",
        "error_rate_mean",
        "error_rate_std",
        "error_rate_change_from_reference",
        "seed_disagreement_rate",
    ]
    return result[column_order].reset_index(drop=True)


def representative_errors(
    predictions: pd.DataFrame,
    described: pd.DataFrame,
    evaluation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Short reviews that a model gets wrong with every seed, per category.

    For each model and category: take the reviews in the highest group (Q4) that
    all reference runs misclassify, keep those of readable length, and list the
    ones with the highest value of the category measure, alternating between
    negative and positive reviews so that both kinds of mistake are shown.
    """
    wrong_runs = (
        predictions.groupby(["condition", "model_name", "review_id"])["error"]
        .agg(wrong_runs="sum", runs="size")
        .reset_index()
    )
    reference = wrong_runs.loc[wrong_runs["condition"].eq(REFERENCE_CONDITION)]
    bigram = wrong_runs.loc[
        wrong_runs["condition"].eq(BIGRAM_CONDITION),
        ["model_name", "review_id", "wrong_runs"],
    ].rename(columns={"wrong_runs": "wrong_runs_with_bigrams"})
    always_wrong = reference.loc[reference["wrong_runs"].eq(reference["runs"])]
    always_wrong = always_wrong.loc[always_wrong["model_name"].ne(BASELINE_MODEL)]
    candidates = (
        always_wrong.merge(bigram, on=["model_name", "review_id"], how="left")
        .merge(described, on="review_id")
        .merge(
            evaluation_data[["review_id", "label", "text"]].rename(
                columns={"label": "true_label"}
            ),
            on="review_id",
        )
    )
    shortest, longest = EXAMPLE_WORD_RANGE
    candidates = candidates.loc[candidates["word_count"].between(shortest, longest)]
    # The data contain a few repeated reviews; show each text once per model.
    candidates = candidates.sort_values("review_id").drop_duplicates(
        ["model_name", "text"]
    )

    examples = []
    for category, measure in CATEGORIES.items():
        top_group = candidates.loc[
            candidates[f"{category}_group"].eq(GROUP_LABELS[-1])
        ].sort_values([measure, "review_id"], ascending=[False, True])
        rank_within_label = top_group.groupby(["model_name", "true_label"]).cumcount()
        chosen = (
            top_group.assign(rank_within_label=rank_within_label)
            .sort_values(["rank_within_label", "true_label"], kind="stable")
            .groupby("model_name", sort=False)
            .head(EXAMPLES_PER_CATEGORY)
            .assign(category=category, measure=measure)
        )
        chosen["measure_value"] = chosen[measure]
        examples.append(chosen)
    if not examples:
        return pd.DataFrame()
    table = pd.concat(examples, ignore_index=True)
    table["predicted_label"] = 1 - table["true_label"]
    table["text"] = [
        " ".join(_LINE_BREAK.sub(" ", text).split()) for text in table["text"]
    ]
    return table[
        [
            "model_name",
            "category",
            "measure",
            "measure_value",
            "review_id",
            "true_label",
            "predicted_label",
            "runs",
            "wrong_runs",
            "wrong_runs_with_bigrams",
            "word_count",
            "negation_examples",
            "rare_word_examples",
            "positive_cue_examples",
            "negative_cue_examples",
            "text",
        ]
    ].sort_values(
        ["model_name", "category", "measure_value"], ascending=[True, True, False]
    )


def run_error_analysis(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    stress_predictions: pd.DataFrame,
    ablation_predictions: pd.DataFrame,
    plan: ExperimentPlan = EXPERIMENT_PLAN,
    evaluation_sets: tuple[str, ...] = EVALUATION_SETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Error rates by group and representative errors for each evaluation set."""
    all_predictions = collect_predictions(stress_predictions, ablation_predictions)
    group_tables = []
    example_tables = []
    for evaluation_set in evaluation_sets:
        predictions = all_predictions.loc[
            all_predictions["evaluation_set"].eq(evaluation_set)
        ]
        if predictions.empty:
            continue
        training_reviews, evaluation_reviews, _ = training_and_evaluation_data(
            training_data, test_data, evaluation_set, plan
        )
        described = describe_reviews(evaluation_reviews, training_reviews)
        group_tables.append(error_rates_by_group(predictions, described))
        examples = representative_errors(predictions, described, evaluation_reviews)
        examples.insert(0, "evaluation_set", evaluation_set)
        example_tables.append(examples)
        print(f"Analyzed {len(described):,} {evaluation_set} reviews.", flush=True)
    return (
        pd.concat(group_tables, ignore_index=True),
        pd.concat(example_tables, ignore_index=True),
    )


def main() -> None:
    stress_path = OUTPUT_DIR / "vocabulary-stress-predictions.parquet"
    ablation_path = OUTPUT_DIR / "ngram-ablation-predictions.parquet"
    if not stress_path.exists() or not ablation_path.exists():
        raise SystemExit(
            "Run ml_semester_project.vocabulary_stress and "
            "ml_semester_project.ngram_ablation first: this analysis reads the "
            "predictions they save."
        )
    groups, examples = run_error_analysis(
        load_reviews("train"),
        load_reviews("test"),
        pd.read_parquet(stress_path),
        pd.read_parquet(ablation_path),
    )
    for path in (
        save_table(groups, "error-analysis-groups.csv"),
        save_table(examples, "error-analysis-examples.csv"),
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
