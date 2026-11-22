import contextlib
import io
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ml_semester_project import experiment_runner
from ml_semester_project.experiment_runner import EXPERIMENT_PLAN
from ml_semester_project.features import TOKEN_PATTERN, build_vectorizer
from ml_semester_project.vocabulary_stress import (
    apply_rare_word_removal,
    prediction_change_rates,
    remove_words,
    run_vocabulary_stress,
    seed_disagreement_rates,
    summarize_vocabulary_stress,
    word_document_frequencies,
)

from synthetic import make_reviews


def frame(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": range(len(texts)),
            "text": texts,
            "label": [0, 1] * (len(texts) // 2),
        }
    )


class DocumentFrequencyTest(unittest.TestCase):
    def test_counts_reviews_not_occurrences(self):
        frequencies = word_document_frequencies(
            ["Good good GOOD movie", "good plot", "bad plot"]
        )
        self.assertEqual(frequencies["good"], 2)
        self.assertEqual(frequencies["plot"], 2)
        self.assertEqual(frequencies["movie"], 1)

    def test_html_line_breaks_and_one_letter_tokens_are_not_words(self):
        frequencies = word_document_frequencies(
            ["a good<br /><br />film", "I liked it"]
        )
        self.assertNotIn("br", frequencies)
        self.assertNotIn("a", frequencies)
        self.assertEqual(sorted(frequencies), ["film", "good", "it", "liked"])


class RareWordRemovalTest(unittest.TestCase):
    def setUp(self):
        self.training = frame(
            [
                "good film",
                "good plot",
                "good acting",
                "bad film",
                "bad plot",
                "odd film",
            ]
        )
        self.evaluation = frame(["Good, ODD film!<br />unseen plot", "bad acting"])
        self.frequencies = word_document_frequencies(self.training["text"])

    def test_reference_level_returns_the_original_reviews(self):
        training, evaluation, condition = apply_rare_word_removal(
            self.training, self.evaluation, self.frequencies, 1
        )
        self.assertIs(training, self.training)
        self.assertIs(evaluation, self.evaluation)
        self.assertEqual(condition["training_words_removed_share"], 0.0)
        self.assertEqual(condition["kept_vocabulary_size"], len(self.frequencies))

    def test_rare_and_unseen_words_are_deleted_from_both_partitions(self):
        training, evaluation, condition = apply_rare_word_removal(
            self.training, self.evaluation, self.frequencies, 2
        )
        # "acting" and "odd" appear in one training review; "unseen" in none.
        self.assertEqual(training["text"].tolist()[2], "good ")
        self.assertEqual(training["text"].tolist()[5], " film")
        self.assertEqual(evaluation["text"].tolist()[0], "Good,  film!<br /> plot")
        self.assertEqual(evaluation["text"].tolist()[1], "bad ")
        self.assertEqual(condition["kept_vocabulary_size"], 4)
        self.assertAlmostEqual(condition["training_words_removed_share"], 2 / 12)
        self.assertAlmostEqual(condition["evaluation_words_removed_share"], 3 / 7)

    def test_original_frames_are_not_modified(self):
        before = self.training["text"].tolist()
        apply_rare_word_removal(self.training, self.evaluation, self.frequencies, 3)
        self.assertEqual(self.training["text"].tolist(), before)

    def test_evaluation_reviews_do_not_decide_which_words_are_rare(self):
        other_evaluation = frame(["odd odd odd acting acting", "unseen unseen"])
        first, _, _ = apply_rare_word_removal(
            self.training, self.evaluation, self.frequencies, 2
        )
        second, _, _ = apply_rare_word_removal(
            self.training, other_evaluation, self.frequencies, 2
        )
        self.assertEqual(first["text"].tolist(), second["text"].tolist())

    def test_removal_equals_restricting_the_unigram_vocabulary(self):
        reviews = make_reviews(120, seed=3)
        reviews.loc[::7, "text"] += " hapaxword" + reviews.loc[::7, "review_id"].astype(
            str
        )
        frequencies = word_document_frequencies(reviews["text"])
        kept = sorted(word for word, count in frequencies.items() if count >= 2)
        stressed, removed_share = remove_words(reviews["text"], set(kept))
        self.assertGreater(removed_share, 0)

        counter = CountVectorizer(token_pattern=TOKEN_PATTERN, vocabulary=kept)
        np.testing.assert_array_equal(
            counter.transform(stressed).toarray(),
            counter.transform(reviews["text"]).toarray(),
        )
        self.assertEqual(sorted(word_document_frequencies(stressed)), kept)

    def test_levels_up_to_the_shared_minimum_leave_shared_features_unchanged(self):
        # The shared vectorizer drops words seen in fewer than 5 training reviews,
        # so deleting those words from the text first must not change its output.
        reviews = make_reviews(200, seed=5)
        reviews.loc[::9, "text"] += " seldomword"
        reviews.loc[::50, "text"] += " veryrareword"
        frequencies = word_document_frequencies(reviews["text"])
        stressed, _, _ = apply_rare_word_removal(reviews, reviews, frequencies, 5)
        self.assertNotEqual(stressed["text"].tolist(), reviews["text"].tolist())

        original_vectorizer = build_vectorizer()
        stressed_vectorizer = build_vectorizer()
        original = original_vectorizer.fit_transform(reviews["text"])
        after = stressed_vectorizer.fit_transform(stressed["text"])
        self.assertEqual(
            original_vectorizer.vocabulary_, stressed_vectorizer.vocabulary_
        )
        np.testing.assert_allclose(original.toarray(), after.toarray(), rtol=1e-6)

    def test_shared_feature_count_matches_the_fitted_shared_vectorizer(self):
        reviews = make_reviews(300, seed=8)
        reviews.loc[::4, "text"] += " sometimesword"
        reviews.loc[::40, "text"] += " seldomword"
        reviews.loc[::150, "text"] += " veryrareword<br />"
        frequencies = word_document_frequencies(reviews["text"])
        for level in (1, 5, 10, 80):
            stressed, _, condition = apply_rare_word_removal(
                reviews, reviews, frequencies, level
            )
            vectorizer = build_vectorizer().fit(stressed["text"])
            self.assertEqual(
                condition["shared_feature_count"], len(vectorizer.vocabulary_)
            )
        self.assertLess(condition["shared_feature_count"], len(frequencies))


class PredictionChangeRateTest(unittest.TestCase):
    def test_share_of_reviews_that_flip_relative_to_the_reference_run(self):
        rows = []
        for level, labels in ((1, [0, 0, 1, 1]), (10, [0, 1, 1, 0])):
            for review_id, label in enumerate(labels):
                rows.append(
                    {
                        "evaluation_set": "validation",
                        "min_document_frequency": level,
                        "model_name": "mlp",
                        "seed": 420,
                        "review_id": review_id,
                        "true_label": 0,
                        "predicted_label": label,
                    }
                )
        rates = prediction_change_rates(pd.DataFrame(rows)).set_index(
            "min_document_frequency"
        )["prediction_change_rate"]
        self.assertEqual(rates[1], 0.0)
        self.assertEqual(rates[10], 0.5)


class SeedDisagreementTest(unittest.TestCase):
    def test_mean_share_of_reviews_on_which_two_seeds_disagree(self):
        labels_by_seed = {1: [0, 0, 1, 1], 2: [0, 0, 1, 0], 3: [1, 0, 1, 0]}
        rows = [
            {
                "evaluation_set": "test",
                "min_document_frequency": level,
                "model_name": "mlp",
                "seed": seed,
                "review_id": review_id,
                "true_label": 0,
                "predicted_label": label if level == 1 else 1 - label,
            }
            for level in (1, 10)
            for seed, labels in labels_by_seed.items()
            for review_id, label in enumerate(labels)
        ]
        rates = seed_disagreement_rates(pd.DataFrame(rows))
        # Pairs (1,2), (1,3), (2,3) disagree on 1, 2, and 1 of the 4 reviews.
        self.assertEqual(len(rates), 1)
        self.assertAlmostEqual(
            rates.loc[0, "reference_seed_disagreement_rate"], (1 + 2 + 1) / (3 * 4)
        )


class StressRunTest(unittest.TestCase):
    def test_every_registered_model_runs_at_every_level_on_both_evaluation_sets(self):
        def always_positive(training_data, evaluation_data, seed):
            return np.ones(len(evaluation_data), dtype=int)

        def unfinished(training_data, evaluation_data, seed):
            raise NotImplementedError("not ready")

        plan = replace(
            EXPERIMENT_PLAN,
            repeat_seeds=(1, 2),
            model_names=("always_positive", "unfinished"),
        )
        runners = {"always_positive": always_positive, "unfinished": unfinished}
        with (
            mock.patch.dict(experiment_runner.MODEL_RUNNERS, runners),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results, predictions = run_vocabulary_stress(
                make_reviews(100, seed=1),
                make_reviews(40, seed=2),
                plan,
                levels=(1, 20),
            )
        summary = summarize_vocabulary_stress(results, predictions)

        self.assertEqual(len(results), 2 * 2 * 2 * 2)
        finished = results.loc[results["status"].eq("completed")]
        self.assertEqual(set(finished["model_name"]), {"always_positive"})
        self.assertEqual(set(finished["evaluation_set"]), {"validation", "test"})
        self.assertTrue((finished["accuracy"] == 0.5).all())
        self.assertEqual(
            results.loc[results["model_name"].eq("unfinished"), "status"]
            .unique()
            .tolist(),
            ["not_implemented"],
        )
        # Validation trains on 90% of the training file; test trains on all of it.
        rows_by_set = finished.groupby("evaluation_set")["training_rows"].first()
        self.assertEqual(rows_by_set["validation"], 90)
        self.assertEqual(rows_by_set["test"], 100)
        self.assertEqual(len(predictions), 2 * 2 * (10 + 40))
        self.assertEqual(len(summary), 4)
        self.assertTrue((summary["prediction_change_rate_mean"] == 0).all())
        self.assertTrue((summary["reference_seed_disagreement_rate"] == 0).all())
        self.assertTrue((summary["runs"] == 2).all())


if __name__ == "__main__":
    unittest.main()
