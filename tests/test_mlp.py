import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ml_semester_project.experiment_support import OUTPUT_DIR
from ml_semester_project.features import UNIGRAM_BIGRAM_FEATURES
from ml_semester_project.models import mlp
from ml_semester_project.models.mlp import (
    BASE_PARAMS,
    SELECTED_PARAMS,
    TUNING_CONFIGURATIONS,
    fit_model,
    run_model,
)

from synthetic import make_reviews


class MlpRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", ConvergenceWarning)
        # Large enough for several gradient steps per epoch, so that early
        # stopping sees the internal validation accuracy improve.
        cls.training = make_reviews(1500, seed=10)
        cls.evaluation = make_reviews(80, seed=11)

    def test_returns_one_label_per_evaluation_review_and_learns_the_task(self):
        predictions = run_model(self.training, self.evaluation, seed=420)
        self.assertEqual(predictions.shape, (len(self.evaluation),))
        self.assertTrue(set(predictions) <= {0, 1})
        accuracy = (predictions == self.evaluation["label"].to_numpy()).mean()
        self.assertGreater(accuracy, 0.9)

    def test_same_seed_gives_the_same_predictions(self):
        first = run_model(self.training, self.evaluation, seed=7)
        second = run_model(self.training, self.evaluation, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_seed_changes_the_initial_weights(self):
        first = fit_model(self.training, seed=1)["mlp"].coefs_[0]
        second = fit_model(self.training, seed=2)["mlp"].coefs_[0]
        self.assertFalse(np.allclose(first, second))

    def test_evaluation_labels_and_extra_columns_are_never_used(self):
        scrambled = self.evaluation.assign(
            label=1 - self.evaluation["label"], extra="ignored"
        )
        np.testing.assert_array_equal(
            run_model(self.training, self.evaluation, seed=420),
            run_model(self.training, scrambled, seed=420),
        )

    def test_vocabulary_is_fit_on_training_reviews_only(self):
        evaluation = self.evaluation.assign(
            text=self.evaluation["text"] + " onlyinevaluation"
        )
        pipeline = fit_model(self.training, seed=420)
        vocabulary = pipeline["features"].vocabulary_
        self.assertNotIn("onlyinevaluation", vocabulary)
        training_words = set(" ".join(self.training["text"]).split())
        self.assertTrue(set(vocabulary) <= training_words)
        np.testing.assert_array_equal(
            pipeline.predict(evaluation["text"]),
            pipeline.predict(self.evaluation["text"]),
        )

    def test_feature_settings_reach_the_vectorizer(self):
        unigram = fit_model(self.training, seed=420)
        bigram = fit_model(self.training, seed=420, features=UNIGRAM_BIGRAM_FEATURES)
        self.assertEqual(unigram["features"].ngram_range, (1, 1))
        self.assertEqual(bigram["features"].ngram_range, (1, 2))
        self.assertGreater(
            bigram["mlp"].coefs_[0].shape[0], unigram["mlp"].coefs_[0].shape[0]
        )
        predictions = run_model(
            self.training, self.evaluation, 420, features=UNIGRAM_BIGRAM_FEATURES
        )
        self.assertEqual(len(predictions), len(self.evaluation))

    def test_hyperparameters_reach_the_network_and_early_stopping_is_on(self):
        params = {**BASE_PARAMS, "hidden_layer_sizes": (16, 8, 4), "alpha": 0.5}
        network = fit_model(self.training, seed=420, params=params)["mlp"]
        self.assertEqual(network.hidden_layer_sizes, (16, 8, 4))
        self.assertEqual(network.n_layers_ - 2, 3)
        self.assertEqual(network.alpha, 0.5)
        self.assertEqual(network.random_state, 420)
        self.assertTrue(network.early_stopping)
        self.assertEqual(network.coefs_[0].dtype, np.float32)


class TuningBudgetTest(unittest.TestCase):
    def test_budget_is_nine_distinct_configurations_with_one_to_three_layers(self):
        self.assertEqual(len(TUNING_CONFIGURATIONS), 9)
        distinct = {tuple(sorted(config.items())) for config in TUNING_CONFIGURATIONS}
        self.assertEqual(len(distinct), 9)
        for config in TUNING_CONFIGURATIONS:
            self.assertEqual(set(config), set(BASE_PARAMS))
            self.assertIn(len(config["hidden_layer_sizes"]), (1, 2, 3))

    def test_each_configuration_changes_at_most_one_hyperparameter(self):
        self.assertEqual(TUNING_CONFIGURATIONS[0], BASE_PARAMS)
        for config in TUNING_CONFIGURATIONS[1:]:
            changed = [
                name for name in BASE_PARAMS if config[name] != BASE_PARAMS[name]
            ]
            self.assertEqual(len(changed), 1)

    def test_selected_configuration_is_inside_the_budget(self):
        self.assertIn(SELECTED_PARAMS, TUNING_CONFIGURATIONS)

    def test_selected_configuration_matches_the_saved_tuning_summary(self):
        summary_path = OUTPUT_DIR / "mlp-tuning-summary.csv"
        if not summary_path.exists():
            self.skipTest("run `python -m ml_semester_project.mlp_tuning` first")
        summary = pd.read_csv(summary_path)
        best = summary.sort_values(
            ["accuracy_mean", "configuration"], ascending=[False, True]
        ).iloc[0]
        self.assertTrue(bool(best["selected"]))
        self.assertEqual(
            mlp.TUNING_CONFIGURATIONS[int(best["configuration"]) - 1], SELECTED_PARAMS
        )


if __name__ == "__main__":
    unittest.main()
