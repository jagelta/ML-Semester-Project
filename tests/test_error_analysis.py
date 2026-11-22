import contextlib
import io
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ml_semester_project import error_analysis
from ml_semester_project.experiment_runner import EXPERIMENT_PLAN, _split_training_data
from ml_semester_project.error_analysis import (
    GROUP_LABELS,
    build_polarity_lexicon,
    collect_predictions,
    describe_reviews,
    error_rates_by_group,
    quartile_groups,
    representative_errors,
    run_error_analysis,
)

from synthetic import NEGATIVE_WORDS, POSITIVE_WORDS, make_reviews


def reviews(texts: list[str], labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"review_id": range(len(texts)), "text": texts, "label": labels}
    )


def prediction_rows(
    evaluation, condition_column, condition, seed, predicted, evaluation_set="test"
):
    return pd.DataFrame(
        {
            "evaluation_set": evaluation_set,
            condition_column: condition,
            "model_name": "mlp",
            "seed": seed,
            "review_id": evaluation["review_id"].to_numpy(),
            "true_label": evaluation["label"].to_numpy(),
            "predicted_label": predicted,
        }
    )


class PolarityLexiconTest(unittest.TestCase):
    def test_lists_hold_the_words_typical_of_each_class(self):
        training = make_reviews(400, seed=4)
        positive, negative = build_polarity_lexicon(
            training, size=6, min_document_frequency=20
        )
        self.assertEqual(positive, frozenset(POSITIVE_WORDS))
        self.assertEqual(negative, frozenset(NEGATIVE_WORDS))

    def test_infrequent_words_are_left_out(self):
        training = make_reviews(400, seed=4)
        training.loc[training["label"].eq(1).idxmax(), "text"] += " uniquepraise"
        positive, _ = build_polarity_lexicon(
            training, size=8, min_document_frequency=20
        )
        self.assertNotIn("uniquepraise", positive)


class ReviewDescriptionTest(unittest.TestCase):
    def setUp(self):
        self.training = make_reviews(400, seed=4)
        self.evaluation = reviews(
            [
                "Great film, NOT boring. I don't think it was dull<br />never!",
                "great wonderful loved excellent story",
                "great awful movie zzzunseen qqqunseen",
                "the and the",
            ],
            [1, 1, 0, 0],
        )
        self.patches = [
            mock.patch.object(error_analysis, "POLARITY_LEXICON_SIZE", 6),
            mock.patch.object(error_analysis, "POLARITY_MIN_DOCUMENT_FREQUENCY", 20),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.described = describe_reviews(self.evaluation, self.training).set_index(
            "review_id"
        )

    def test_negation_cues_are_counted_per_hundred_words(self):
        first = self.described.loc[0]
        self.assertEqual(first["negation_cues"], 3)  # not, don't, never
        self.assertAlmostEqual(first["negation_rate"], 100 * 3 / first["word_count"])
        self.assertEqual(self.described.loc[1, "negation_cues"], 0)

    def test_rare_word_share_counts_words_the_reference_features_cannot_use(self):
        self.assertAlmostEqual(self.described.loc[2, "rare_word_share"], 2 / 5)
        self.assertEqual(self.described.loc[1, "rare_word_share"], 0.0)
        self.assertEqual(
            self.described.loc[2, "rare_word_examples"], "zzzunseen, qqqunseen"
        )

    def test_mixed_polarity_share_is_zero_when_one_sided_and_half_when_even(self):
        self.assertEqual(self.described.loc[1, "mixed_polarity_share"], 0.0)
        self.assertEqual(self.described.loc[2, "mixed_polarity_share"], 0.5)
        self.assertTrue(np.isnan(self.described.loc[3, "mixed_polarity_share"]))
        self.assertTrue(pd.isna(self.described.loc[3, "mixed_polarity_group"]))

    def test_evaluation_labels_do_not_change_the_description(self):
        flipped = self.evaluation.assign(label=1 - self.evaluation["label"])
        again = describe_reviews(flipped, self.training).set_index("review_id")
        pd.testing.assert_frame_equal(self.described, again)


class QuartileGroupsTest(unittest.TestCase):
    def test_groups_have_equal_size_even_with_ties(self):
        values = pd.Series([0.0] * 6 + [1.0, 2.0])
        groups = quartile_groups(values)
        self.assertEqual(groups.value_counts().tolist(), [2, 2, 2, 2])
        self.assertEqual(groups.iloc[-2:].tolist(), [GROUP_LABELS[-1]] * 2)
        self.assertEqual(sorted(groups.iloc[:6]), sorted(GROUP_LABELS[:3] * 2))

    def test_ties_are_not_split_by_position_in_the_file(self):
        # Half of the reviews are tied at zero. Position must not decide which of
        # them land in the lowest group, and the split must be repeatable.
        values = pd.Series([0.0] * 400 + list(np.linspace(0.1, 1.0, 400)))
        groups = quartile_groups(values)
        lowest = np.flatnonzero(groups.eq(GROUP_LABELS[0]).to_numpy())
        self.assertEqual(len(lowest), 200)
        self.assertLess(lowest.max(), 400)
        self.assertGreater(np.sum(lowest >= 200), 60)  # about half, not none
        self.assertLess(np.sum(lowest >= 200), 140)
        pd.testing.assert_series_equal(groups, quartile_groups(values))

    def test_missing_values_get_no_group(self):
        groups = quartile_groups(pd.Series([0.1, 0.2, np.nan, 0.3, 0.4]))
        self.assertTrue(pd.isna(groups.iloc[2]))
        self.assertEqual(groups.notna().sum(), 4)


class ErrorRatesTest(unittest.TestCase):
    def setUp(self):
        # Eight reviews; negation rate grows with the review id, so each quarter
        # holds two reviews: ids (0, 1), (2, 3), (4, 5), (6, 7).
        self.evaluation = reviews([f"text {i}" for i in range(8)], [0, 1] * 4)
        self.described = pd.DataFrame(
            {
                "review_id": range(8),
                "word_count": 60,
                "negation_rate": np.arange(8, dtype=float),
                "rare_word_share": np.arange(8, dtype=float)[::-1],
                "mixed_polarity_share": [np.nan] * 8,
                "negation_examples": "not",
                "rare_word_examples": "",
                "positive_cue_examples": "",
                "negative_cue_examples": "",
            }
        )
        for category, measure in error_analysis.CATEGORIES.items():
            self.described[f"{category}_group"] = quartile_groups(
                self.described[measure]
            )

        truth = self.evaluation["label"].to_numpy()

        def wrong_on(*review_ids):
            predicted = truth.copy()
            predicted[list(review_ids)] = 1 - predicted[list(review_ids)]
            return predicted

        stress = pd.concat(
            [
                prediction_rows(
                    self.evaluation, "min_document_frequency", 1, 1, wrong_on(6, 7)
                ),
                prediction_rows(
                    self.evaluation, "min_document_frequency", 1, 2, wrong_on(7)
                ),
                prediction_rows(
                    self.evaluation, "min_document_frequency", 10, 1, wrong_on(0, 1, 2)
                ),
            ]
        )
        ablation = pd.concat(
            [
                prediction_rows(
                    self.evaluation, "features", "unigram", 1, wrong_on(6, 7)
                ),
                prediction_rows(
                    self.evaluation, "features", "unigram+bigram", 1, wrong_on(7)
                ),
                prediction_rows(
                    self.evaluation, "features", "unigram+bigram", 2, wrong_on(7)
                ),
            ]
        )
        self.predictions = collect_predictions(stress, ablation)
        self.table = error_rates_by_group(self.predictions, self.described)

    def row(self, condition, category, group):
        match = self.table.loc[
            self.table["condition"].eq(condition)
            & self.table["category"].eq(category)
            & self.table["group"].eq(group)
        ]
        self.assertEqual(len(match), 1)
        return match.iloc[0]

    def test_only_reference_level_and_bigram_arm_are_collected(self):
        self.assertEqual(
            set(self.predictions["condition"]), {"reference", "unigram+bigram"}
        )
        self.assertEqual(len(self.predictions), 4 * 8)

    def test_error_rate_is_averaged_over_seeds_within_each_group(self):
        top = self.row("reference", "negation", "Q4 (highest)")
        self.assertEqual(top["reviews"], 2)
        self.assertEqual(top["runs"], 2)
        self.assertAlmostEqual(top["error_rate_mean"], 0.75)  # seeds: 2/2 and 1/2
        self.assertAlmostEqual(top["error_rate_std"], np.std([1.0, 0.5], ddof=1))
        self.assertAlmostEqual(top["seed_disagreement_rate"], 0.5)  # review 6
        self.assertEqual((top["measure_min"], top["measure_max"]), (6.0, 7.0))
        self.assertAlmostEqual(top["positive_share"], 0.5)
        self.assertEqual(
            self.row("reference", "negation", "Q1 (lowest)")["error_rate_mean"], 0
        )
        # Rare-word share runs the other way, so the same reviews are its Q1.
        self.assertAlmostEqual(
            self.row("reference", "rare_words", "Q1 (lowest)")["error_rate_mean"], 0.75
        )

    def test_overall_row_and_change_from_reference(self):
        overall = self.row("reference", "all", "All reviews")
        self.assertEqual(overall["reviews"], 8)
        self.assertAlmostEqual(overall["error_rate_mean"], (2 / 8 + 1 / 8) / 2)
        self.assertEqual(overall["error_rate_change_from_reference"], 0)
        with_bigrams = self.row("unigram+bigram", "negation", "Q4 (highest)")
        self.assertAlmostEqual(with_bigrams["error_rate_mean"], 0.5)
        self.assertAlmostEqual(with_bigrams["error_rate_change_from_reference"], -0.25)

    def test_a_repeated_review_text_is_listed_once(self):
        # Reviews 6 and 7 are copies of each other and both are always wrong.
        evaluation = self.evaluation.copy()
        evaluation.loc[evaluation["review_id"].isin([6, 7]), "text"] = "the same text"
        examples = representative_errors(
            self.predictions.assign(error=True), self.described, evaluation
        )
        by_category = examples.groupby("category")["review_id"].apply(list)
        self.assertEqual(by_category["negation"], [6])
        self.assertEqual(by_category["rare_words"], [0, 1])

    def test_reviews_without_a_value_are_left_out_of_that_category(self):
        self.assertFalse(self.table["category"].eq("mixed_polarity").any())

    def test_representative_errors_are_wrong_with_every_seed_and_in_the_top_group(self):
        examples = representative_errors(
            self.predictions, self.described, self.evaluation
        )
        negation = examples.loc[examples["category"].eq("negation")]
        # Review 7 is wrong with both seeds; review 6 with only one of them.
        self.assertEqual(negation["review_id"].tolist(), [7])
        self.assertEqual(negation.iloc[0]["wrong_runs"], 2)
        self.assertEqual(negation.iloc[0]["wrong_runs_with_bigrams"], 2)
        self.assertEqual(
            negation.iloc[0]["predicted_label"], 1 - negation.iloc[0]["true_label"]
        )
        self.assertTrue(examples.loc[examples["category"].eq("rare_words")].empty)


class EndToEndTest(unittest.TestCase):
    def test_tables_cover_both_evaluation_sets(self):
        training = make_reviews(400, seed=1)
        test = make_reviews(80, seed=2)
        _, validation = _split_training_data(training, EXPERIMENT_PLAN)
        frames = []
        for evaluation_set, evaluation in (("validation", validation), ("test", test)):
            wrong = evaluation["label"].to_numpy().copy()
            wrong[:5] = 1 - wrong[:5]
            for seed in (1, 2):
                frames.append(
                    prediction_rows(
                        evaluation,
                        "min_document_frequency",
                        1,
                        seed,
                        wrong,
                        evaluation_set,
                    )
                )
        stress = pd.concat(frames, ignore_index=True)
        with (
            mock.patch.object(error_analysis, "POLARITY_LEXICON_SIZE", 6),
            mock.patch.object(error_analysis, "POLARITY_MIN_DOCUMENT_FREQUENCY", 20),
            mock.patch.object(error_analysis, "EXAMPLE_WORD_RANGE", (1, 500)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            groups, examples = run_error_analysis(
                training, test, stress, pd.DataFrame()
            )
        self.assertEqual(set(groups["evaluation_set"]), {"validation", "test"})
        self.assertEqual(set(groups["condition"]), {"reference"})
        overall = groups.loc[groups["category"].eq("all")].set_index("evaluation_set")
        self.assertAlmostEqual(overall.loc["test", "error_rate_mean"], 5 / 80)
        self.assertAlmostEqual(overall.loc["validation", "error_rate_mean"], 5 / 40)
        self.assertEqual(
            set(examples["evaluation_set"]) - {"validation", "test"}, set()
        )
        self.assertTrue((examples["wrong_runs"] == examples["runs"]).all())


if __name__ == "__main__":
    unittest.main()
