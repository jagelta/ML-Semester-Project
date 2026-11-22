import unittest

import numpy as np

from ml_semester_project.features import (
    REFERENCE_FEATURES,
    UNIGRAM_BIGRAM_FEATURES,
    FeatureConfig,
    build_vectorizer,
    normalize_text,
)

from synthetic import make_reviews

LOW_THRESHOLD = FeatureConfig(min_document_frequency=1)


class SharedFeaturesTest(unittest.TestCase):
    def test_reference_setting_is_unigrams_seen_in_five_training_reviews(self):
        self.assertEqual(REFERENCE_FEATURES.ngram_range, (1, 1))
        self.assertEqual(REFERENCE_FEATURES.min_document_frequency, 5)
        self.assertEqual(UNIGRAM_BIGRAM_FEATURES.ngram_range, (1, 2))
        self.assertEqual(
            UNIGRAM_BIGRAM_FEATURES.min_document_frequency,
            REFERENCE_FEATURES.min_document_frequency,
        )

    def test_html_line_breaks_are_removed_and_text_is_lowercased(self):
        self.assertEqual(
            normalize_text("Great<br /><BR>Movie").split(), ["great", "movie"]
        )
        vectorizer = build_vectorizer(LOW_THRESHOLD)
        vectorizer.fit(["Great<br /><br />movie", "bad movie"])
        self.assertEqual(
            sorted(vectorizer.get_feature_names_out()), ["bad", "great", "movie"]
        )

    def test_negation_words_are_kept(self):
        vectorizer = build_vectorizer(LOW_THRESHOLD)
        vectorizer.fit(["this is not good", "no way"])
        self.assertIn("not", vectorizer.vocabulary_)
        self.assertIn("no", vectorizer.vocabulary_)

    def test_vocabulary_comes_from_training_reviews_only(self):
        training = make_reviews(60)
        vectorizer = build_vectorizer()
        vectorizer.fit(training["text"])
        vocabulary_before = dict(vectorizer.vocabulary_)
        matrix = vectorizer.transform(["great unseenword movie"])
        self.assertEqual(vectorizer.vocabulary_, vocabulary_before)
        self.assertNotIn("unseenword", vectorizer.vocabulary_)
        self.assertEqual(matrix.shape[1], len(vocabulary_before))

    def test_ngrams_below_the_document_frequency_threshold_are_dropped(self):
        texts = ["common rare"] + ["common word"] * 5
        vectorizer = build_vectorizer(FeatureConfig(min_document_frequency=5))
        vectorizer.fit(texts)
        self.assertEqual(sorted(vectorizer.vocabulary_), ["common", "word"])

    def test_bigram_arm_contains_every_unigram_feature(self):
        training = make_reviews(200)
        unigrams = set(
            build_vectorizer(REFERENCE_FEATURES).fit(training["text"]).vocabulary_
        )
        both = set(
            build_vectorizer(UNIGRAM_BIGRAM_FEATURES).fit(training["text"]).vocabulary_
        )
        self.assertTrue(unigrams < both)
        self.assertTrue(all(" " in feature for feature in both - unigrams))

    def test_tfidf_rows_have_unit_length_and_counts_are_whole_numbers(self):
        training = make_reviews(60)
        tfidf = build_vectorizer(weighting="tfidf").fit_transform(training["text"])
        counts = build_vectorizer(weighting="count").fit_transform(training["text"])
        self.assertEqual(tfidf.dtype, np.float32)
        row_lengths = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
        np.testing.assert_allclose(row_lengths, 1.0, rtol=1e-5)
        self.assertTrue(np.issubdtype(counts.dtype, np.integer))
        self.assertEqual(tfidf.shape, counts.shape)

    def test_unknown_weighting_is_rejected(self):
        with self.assertRaises(ValueError):
            build_vectorizer(weighting="binary")


if __name__ == "__main__":
    unittest.main()
