import contextlib
import io
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import pandas as pd

from ml_semester_project import experiment_runner
from ml_semester_project.experiment_runner import (
    EXPERIMENT_PLAN,
    ExperimentConfig,
    _split_training_data,
    _stratified_training_subset,
    run_experiment,
)
from ml_semester_project.experiment_support import (
    summarize,
    training_and_evaluation_data,
)
from ml_semester_project.features import REFERENCE_FEATURES, UNIGRAM_BIGRAM_FEATURES
from ml_semester_project.ngram_ablation import (
    run_ngram_ablation,
    summarize_ngram_ablation,
)

from synthetic import make_reviews


class PartitionTest(unittest.TestCase):
    def setUp(self):
        self.training = make_reviews(200, seed=1)
        self.test = make_reviews(60, seed=2)

    def test_validation_split_is_disjoint_stratified_and_repeatable(self):
        pool, validation = _split_training_data(self.training, EXPERIMENT_PLAN)
        again_pool, again_validation = _split_training_data(
            self.training, EXPERIMENT_PLAN
        )
        self.assertEqual(len(pool), 180)
        self.assertEqual(len(validation), 20)
        self.assertFalse(set(pool["review_id"]) & set(validation["review_id"]))
        self.assertEqual(validation["label"].mean(), 0.5)
        self.assertEqual(pool["review_id"].tolist(), again_pool["review_id"].tolist())
        self.assertEqual(
            validation["review_id"].tolist(), again_validation["review_id"].tolist()
        )

    def test_full_training_fraction_keeps_every_review_for_every_seed(self):
        pool, _ = _split_training_data(self.training, EXPERIMENT_PLAN)
        for seed in EXPERIMENT_PLAN.repeat_seeds:
            subset = _stratified_training_subset(pool, 1.0, seed)
            self.assertEqual(set(subset["review_id"]), set(pool["review_id"]))

    def test_validation_and_test_settings_follow_the_harness(self):
        train, evaluation, split_seed = training_and_evaluation_data(
            self.training, self.test, "validation", EXPERIMENT_PLAN
        )
        self.assertEqual((len(train), len(evaluation)), (180, 20))
        self.assertEqual(split_seed, EXPERIMENT_PLAN.split_seed)
        train, evaluation, split_seed = training_and_evaluation_data(
            self.training, self.test, "test", EXPERIMENT_PLAN
        )
        self.assertIs(train, self.training)
        self.assertIs(evaluation, self.test)
        self.assertIsNone(split_seed)
        with self.assertRaises(ValueError):
            training_and_evaluation_data(
                self.training, self.test, "dev", EXPERIMENT_PLAN
            )


class FeatureSettingsTest(unittest.TestCase):
    def setUp(self):
        self.training = make_reviews(100, seed=1)
        self.evaluation = make_reviews(20, seed=2)
        self.received = []

        def old_style(training_data, evaluation_data, seed):
            self.received.append("old_style")
            return np.zeros(len(evaluation_data), dtype=int)

        def new_style(
            training_data, evaluation_data, seed, features=REFERENCE_FEATURES
        ):
            self.received.append(features)
            return np.zeros(len(evaluation_data), dtype=int)

        self.runners = {"old_style": old_style, "new_style": new_style}

    def run_one(self, model_name, features=None):
        with mock.patch.dict(experiment_runner.MODEL_RUNNERS, self.runners):
            return run_experiment(
                ExperimentConfig(model_name, 1.0, 420),
                EXPERIMENT_PLAN,
                self.training,
                self.evaluation,
                "validation",
                420,
                features,
            )

    def test_without_feature_settings_runners_are_called_as_before(self):
        for model_name in ("old_style", "new_style"):
            metrics, predictions = self.run_one(model_name)
            self.assertEqual(metrics["status"], "completed")
            self.assertEqual(len(predictions), len(self.evaluation))
        self.assertEqual(self.received, ["old_style", REFERENCE_FEATURES])

    def test_feature_settings_are_passed_to_runners_that_accept_them(self):
        metrics, _ = self.run_one("new_style", UNIGRAM_BIGRAM_FEATURES)
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(self.received, [UNIGRAM_BIGRAM_FEATURES])

    def test_runner_without_feature_support_is_reported_not_silently_run(self):
        metrics, predictions = self.run_one("old_style", UNIGRAM_BIGRAM_FEATURES)
        self.assertEqual(metrics["status"], "not_implemented")
        self.assertIn("feature settings", metrics["reason"])
        self.assertIsNone(predictions)
        self.assertEqual(self.received, [])

    def test_ablation_runs_both_arms_and_reports_the_gain_per_seed(self):
        def better_with_bigrams(training_data, evaluation_data, seed, features):
            labels = evaluation_data["label"].to_numpy().copy()
            mistakes = 2 if features.ngram_range == (1, 2) else 6
            labels[:mistakes] = 1 - labels[:mistakes]
            return labels

        plan = replace(
            EXPERIMENT_PLAN,
            repeat_seeds=(1, 2, 3),
            model_names=("better_with_bigrams", "old_style"),
        )
        runners = {**self.runners, "better_with_bigrams": better_with_bigrams}
        with (
            mock.patch.dict(experiment_runner.MODEL_RUNNERS, runners),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results, predictions = run_ngram_ablation(
                make_reviews(200, seed=1), make_reviews(40, seed=2), plan
            )
        summary = summarize_ngram_ablation(results)

        self.assertEqual(len(results), 2 * 2 * 2 * 3)
        self.assertEqual(
            results.loc[results["model_name"].eq("old_style"), "status"]
            .unique()
            .tolist(),
            ["not_implemented"],
        )
        self.assertEqual(set(summary["model_name"]), {"better_with_bigrams"})
        self.assertEqual(set(predictions["features"]), {"unigram", "unigram+bigram"})
        test_rows = summary.loc[summary["evaluation_set"].eq("test")].set_index(
            "features"
        )
        self.assertAlmostEqual(test_rows.loc["unigram", "accuracy_mean"], 34 / 40)
        self.assertAlmostEqual(
            test_rows.loc["unigram+bigram", "accuracy_mean"], 38 / 40
        )
        self.assertAlmostEqual(
            test_rows.loc["unigram+bigram", "accuracy_gain_from_bigrams_mean"], 4 / 40
        )
        self.assertAlmostEqual(
            test_rows.loc["unigram+bigram", "accuracy_gain_from_bigrams_std"], 0.0
        )
        self.assertGreater(
            test_rows.loc["unigram+bigram", "shared_feature_count"],
            test_rows.loc["unigram", "shared_feature_count"],
        )


class SummaryTest(unittest.TestCase):
    def test_mean_and_sample_standard_deviation_over_completed_runs(self):
        results = pd.DataFrame(
            {
                "model_name": ["mlp", "mlp", "mlp", "mlp"],
                "status": ["completed", "completed", "completed", "not_implemented"],
                "accuracy": [0.80, 0.82, 0.84, None],
            }
        )
        summary = summarize(results, ["model_name"], ["accuracy"])
        self.assertEqual(summary.loc[0, "runs"], 3)
        self.assertAlmostEqual(summary.loc[0, "accuracy_mean"], 0.82)
        self.assertAlmostEqual(summary.loc[0, "accuracy_std"], 0.02)


if __name__ == "__main__":
    unittest.main()
