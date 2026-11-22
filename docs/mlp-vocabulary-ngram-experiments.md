# Neural model, vocabulary stress test, n-gram ablation, and error analysis

Design notes and results for the part of the project owned by Erik Vera.
Written on November 18, 2026. Checked against the team's `main` on November 22,
2026 (section 15).

Every number in this document comes from a file in
`src/ml_semester_project/output/` and can be regenerated with the commands in the
[README](../README.md). Accuracy is written as mean ± standard deviation over the
three repeat seeds (420, 421, 422).

## 1. What this covers

| Task | Code | Models it runs |
| --- | --- | --- |
| Neural model (MLP) | `models/mlp.py` | MLP |
| Hyperparameter search | `mlp_tuning.py` | MLP |
| Training-size results | `mlp_training_size.py` | MLP |
| Vocabulary-change experiment | `vocabulary_stress.py` | every registered model |
| Unigram versus unigram+bigram ablation | `ngram_ablation.py` | every registered model that accepts feature settings |
| Error analysis | `error_analysis.py` | every model with saved predictions |
| Shared text features | `features.py` | used by the MLP, offered to the other models |

**Status of the models.** The majority baseline and the MLP are implemented.
Logistic regression and Naive Bayes are still stubs, so their runs are saved with
status `not_implemented`. The experiments pick them up once they exist (section
12) and then have to be re-run. Until then, every statement below is about the
MLP and the baseline only. Nothing here says yet which model is the most
reliable.

## 2. Reference setting

Each experiment changes one row of this table and leaves the others as they are.

| Item | Reference value |
| --- | --- |
| Training data | All of it: 22,500 reviews for validation results, 25,000 for test results |
| Review text | Unchanged |
| Features | TF-IDF over unigrams that appear in at least 5 training reviews |
| MLP | 1 hidden layer of 32 units, learning rate 0.001, L2 penalty 0.0001 |
| Seeds | 420, 421, 422 (the split uses 420) |
| Primary metric | Accuracy |

Accuracy is the primary metric because every partition has exactly as many
positive as negative reviews. Always guessing one class gives 0.500, which is
what the majority baseline scores everywhere.

## 3. Shared text features (`features.py`)

The comparison is about models, so the models should read the same input. This
module holds one set of rules. The MLP uses it. The team's harness notes
(`HARNESS_REQUIREMENTS.md`, removed from the repository on November 19) said that
each model owns its preprocessing, so the other models are free to adopt it or
not (section 13).

| Decision | Reason |
| --- | --- |
| Lowercase, and turn `<br />` into a space | About 58% of the reviews contain HTML line breaks. Without this, "br" becomes one of the most frequent tokens and pollutes the bigrams. |
| A word is a run of two or more letters, digits, or underscores | It is scikit-learn's default and needs no extra tools. Punctuation and one-letter tokens are dropped. |
| Keep stop words | Standard stop-word lists contain "not", "no", and "never", which carry sentiment. |
| An n-gram needs at least 5 training reviews | A word seen once or twice gives a model almost nothing to learn from. It also keeps the unigram+bigram vocabulary trainable on a laptop: on the 22,500 training reviews there are 1,408,750 distinct unigrams and bigrams, and 142,295 of them pass the threshold. |
| TF-IDF, sublinear term frequency, rows scaled to length 1 | Training reviews run from 10 to 2,470 words. Scaling puts short and long reviews on the same footing, `1 + log(tf)` stops a word repeated twenty times from counting twenty times, and IDF lowers the weight of words that appear everywhere. |
| Vocabulary and IDF are learned from the training partition only | Anything learned from evaluation reviews would be data leakage. |

On the 22,500 training reviews there are 71,753 distinct words and 25,921 of them
pass the threshold of 5. On the full training file the numbers are 74,849 and
27,271.

None of these choices was tuned. No validation score was compared to pick them.
The threshold of 5 was chosen for training time: in a timing check that measured
only seconds, an MLP with 64 hidden units needed about 300 seconds per fit on
unigram+bigram features with a threshold of 2 (401,159 features) and about 110
seconds with a threshold of 5.

`build_vectorizer(features, weighting="count")` returns raw counts with the same
tokens and vocabulary, which is the usual input for multinomial Naive Bayes.

## 4. The MLP (`models/mlp.py`)

**Library.** scikit-learn's `MLPClassifier`. The guidelines point to
scikit-learn, the network is small, it accepts sparse TF-IDF input directly, and
one fit takes about ten seconds on a CPU. There was no reason to use PyTorch.

**Network.** 25,921 TF-IDF inputs, one hidden layer of 32 ReLU units, one output
unit with a logistic function. That is 829,537 trainable weights.

**Training.** Adam optimizer, mini-batches of 200 reviews, log-loss.

**Early stopping.** The MLP sets aside 10% of the training reviews it receives,
checks accuracy on them after every epoch, stops when four epochs in a row bring
no improvement, and goes back to the best weights. The limit is 50 epochs.

- Why: the right number of epochs depends on how much data and how many features
  there are. Early stopping picks it separately for each run, so one
  configuration can be reused for every training size and stress level without
  tuning again.
- Cost: the weights are learned from 90% of the training reviews. Logistic
  regression and Naive Bayes can use all of them.
- The reviews set aside always come from the training partition. They are never
  the harness validation set or the test set.

**What the seed controls.** The initial weights, the order of the mini-batches,
and which training reviews are set aside for early stopping. This is why the MLP
needs repeated runs and the deterministic models do not.

**What is fixed and what is tuned.** Activation, optimizer, batch size, and the
early-stopping settings are fixed. Depth, width, learning rate, and L2 penalty
are tuned (section 5).

**Behavior with little data.** With 2,250 training reviews only 225 are set aside
for early stopping, so the stopping decision is noisier: in a check on training
data, the three seeds stopped after 11, 16, and 17 epochs, against 8 epochs for
all three with the full training partition. This is one reason the results vary
more across seeds at small training sizes (section 6).

## 5. Hyperparameter search (`mlp_tuning.py`)

**Budget: nine configurations.** The guidelines define a comparable tuning budget
as the same number of configurations per model, selected on validation data. The
team has not fixed the number yet, so nine is a proposal (section 13).

**Design.** One starting configuration, plus eight that each change one
hyperparameter of it. The four hyperparameters are the number of hidden layers,
the number of units, the learning rate, and the L2 penalty. This covers all four
with a small budget and shows the effect of each one separately. It does not test
combinations.

**Protocol.**

- The nine configurations were written down before any of them was scored. No
  other configuration was tried.
- Each one was trained on the 22,500-review training partition with the three
  seeds and scored on the 2,500-review validation partition.
- The script never reads the test file.
- Selection rule: highest mean validation accuracy. A tie goes to the
  configuration listed first.
- The selected configuration is reused, unchanged, in every other experiment.

Two checks were run before the search. Neither used validation or test data: one
measured training time, the other looked at early-stopping curves on training
data.

**Results** (`mlp-tuning-summary.csv`, figure `figures/mlp-tuning`).

| Rank | What changed | Hidden layers | Learning rate | L2 penalty | Weights | Validation accuracy | Epochs | Seconds per fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | units | 32 | 0.001 | 0.0001 | 829,537 | 0.9035 ± 0.0017 | 8.0 | 8 |
| 2 | layers | 64-64 | 0.001 | 0.0001 | 1,663,233 | 0.9001 ± 0.0032 | 6.0 | 13 |
| 3 | layers | 64-64-64 | 0.001 | 0.0001 | 1,667,393 | 0.8989 ± 0.0036 | 5.3 | 12 |
| 4 | units | 128 | 0.001 | 0.0001 | 3,318,145 | 0.8989 ± 0.0017 | 7.0 | 30 |
| 5 | learning rate | 64 | 0.0003 | 0.0001 | 1,659,073 | 0.8975 ± 0.0013 | 17.7 | 30 |
| 6 | L2 penalty | 64 | 0.001 | 0.01 | 1,659,073 | 0.8971 ± 0.0023 | 8.0 | 18 |
| 7 | nothing (starting point) | 64 | 0.001 | 0.0001 | 1,659,073 | 0.8967 ± 0.0020 | 8.7 | 17 |
| 8 | learning rate | 64 | 0.003 | 0.0001 | 1,659,073 | 0.8964 ± 0.0032 | 5.7 | 13 |
| 9 | L2 penalty | 64 | 0.001 | 1 | 1,659,073 | 0.8804 ± 0.0012 | 18.3 | 51 |

**How to read it.**

- The selected configuration is one hidden layer of 32 units. It is also the
  smallest and the fastest network in the list.
- Eight of the nine configurations are within 0.7 points of each other. The
  sampling error of an accuracy near 0.90 measured on 2,500 reviews is about 0.6
  points, so these eight cannot be told apart reliably. The honest reading is
  that depth, width, learning rate, and a mild L2 penalty make no measurable
  difference here.
- Only the very strong L2 penalty clearly hurts: 1.6 points below the starting
  point.
- A second and a third hidden layer did not help. For bag-of-words sentiment, a
  small network is enough.
- The selected configuration was picked because its validation score was the
  highest, so that score is slightly optimistic. The test accuracy in section 6
  is the fair estimate.
- The seconds are rough: other jobs were running on the same 2-core machine
  during part of the search.

## 6. Training size (`mlp_training_size.py`)

This is the harness training-size sweep, run for the MLP alone and saved in its
own files, so that it does not overwrite the shared all-model outputs.

Results (`mlp-training-size-summary.csv`, figure `figures/mlp-training-size`):

| Evaluated on | Training fraction | Training reviews | Accuracy |
| --- | --- | --- | --- |
| validation | 10% | 2,250 | 0.8472 ± 0.0059 |
| validation | 20% | 4,500 | 0.8728 ± 0.0033 |
| validation | 30% | 6,750 | 0.8819 ± 0.0034 |
| validation | 40% | 9,000 | 0.8847 ± 0.0066 |
| validation | 50% | 11,250 | 0.8857 ± 0.0051 |
| validation | 60% | 13,500 | 0.8951 ± 0.0027 |
| validation | 70% | 15,749 | 0.8967 ± 0.0024 |
| validation | 80% | 18,000 | 0.8992 ± 0.0011 |
| validation | 90% | 20,250 | 0.9012 ± 0.0025 |
| validation | 100% | 22,500 | 0.9035 ± 0.0017 |
| test | 100% | 25,000 | 0.8855 ± 0.0007 |

**How to read it.**

- With one tenth of the data the MLP reaches 0.847. With all of it, 0.903. Ten
  times more data buys 5.6 points.
- Almost half of that gain comes from the first doubling, from 2,250 to 4,500
  reviews (2.6 points). After 13,500 reviews each extra 2,250 adds about 0.2
  points.
- The spread across seeds is larger with little data (up to 0.7 points at 40%)
  than with all of it (0.2 points).
- Test accuracy (0.8855) is 1.8 points below validation accuracy (0.9035).
  Section 10 explains why this is expected and not a sign of leakage.

## 7. Vocabulary stress test (`vocabulary_stress.py`)

### Design

| Decision | Choice | Reason |
| --- | --- | --- |
| Which change | Rare-word removal | It is one of the two changes the guidelines suggest. It asks a clear question about vocabulary (does a model need the long tail of rare words?) and it adds no randomness of its own. |
| Where it is applied | Training and evaluation reviews; every model is retrained | This is the "training and test data" setting of the guidelines, whose own example is rare-word removal. It answers: how well can a model learn when this information is absent from the data? It does not answer how a model trained on clean text copes with damaged text. |
| What "rare" means | The word appears in fewer than k training reviews | Counted on the training partition only. A word that appears only in evaluation reviews has a count of zero and is removed too. |
| Severity levels | k = 1 (reference), 10, 30, 100, 300, 1000, 3000 | About three times stricter at each step, the same for every model. |
| How it is done | The words are deleted from the review text before any model sees it | Every model is stressed in exactly the same way through the normal `run_model` interface, whatever vectorizer it uses. |
| Everything else | Reference setting; hyperparameters are not tuned again | One factor at a time. |

There is no level between 1 and 10 because the shared features already ignore
words seen in fewer than 5 training reviews. For a model that uses them, removing
those words from the text changes nothing. A unit test checks this.

Besides accuracy, the summary reports two stability numbers:

- **Predictions changed**: the share of evaluation reviews whose predicted label
  differs from the reference run with the same seed.
- **Seed-to-seed disagreement**: the share of reviews on which two reference runs
  with different seeds disagree. It is the yardstick for the first number,
  because a stochastic model changes some predictions even when nothing but the
  seed changes.

### Results for the MLP

Test set (`vocabulary-stress-summary.csv`, figure `figures/vocabulary-stress`):

| Level k | Unigram features left | Words removed | Accuracy | Change (points) | Predictions changed |
| --- | --- | --- | --- | --- | --- |
| 1 (reference) | 27,271 | 0.0% | 0.8855 ± 0.0007 | 0.00 | 0.0% |
| 10 | 18,522 | 3.4% | 0.8836 ± 0.0018 | -0.19 | 3.2% |
| 30 | 9,260 | 6.4% | 0.8845 ± 0.0015 | -0.09 | 4.2% |
| 100 | 3,832 | 12.2% | 0.8847 ± 0.0009 | -0.08 | 5.9% |
| 300 | 1,559 | 20.0% | 0.8729 ± 0.0017 | -1.26 | 8.6% |
| 1000 | 522 | 31.3% | 0.8429 ± 0.0005 | -4.26 | 12.7% |
| 3000 | 175 | 43.8% | 0.7570 ± 0.0004 | -12.85 | 21.7% |

Seed-to-seed disagreement on the test set: 2.8%.

Validation set:

| Level k | Unigram features left | Words removed | Accuracy | Change (points) | Predictions changed |
| --- | --- | --- | --- | --- | --- |
| 1 (reference) | 25,921 | 0.0% | 0.9035 ± 0.0017 | 0.00 | 0.0% |
| 10 | 17,492 | 3.2% | 0.9020 ± 0.0011 | -0.15 | 2.3% |
| 30 | 8,581 | 6.7% | 0.8971 ± 0.0005 | -0.64 | 3.4% |
| 100 | 3,515 | 13.0% | 0.8895 ± 0.0034 | -1.40 | 5.0% |
| 300 | 1,424 | 21.1% | 0.8748 ± 0.0034 | -2.87 | 8.3% |
| 1000 | 475 | 32.5% | 0.8303 ± 0.0023 | -7.32 | 13.3% |
| 3000 | 163 | 44.7% | 0.7525 ± 0.0029 | -15.09 | 20.8% |

Seed-to-seed disagreement on the validation set: 2.0%.

The majority baseline scores 0.500 at every level, because it never reads the
text.

### How to read it

1. **On the test set the MLP does not need the rare words.** Cutting the
   vocabulary from 27,271 to 3,832 features (level 100) costs 0.1 points, which
   is inside the noise. The sampling error on 25,000 reviews is about 0.2 points.
2. **It breaks down only when common words go too.** The loss is 1.3 points with
   1,559 features, 4.3 points with 522, and 12.9 points with 175. Even with 175
   words the MLP is far above the baseline (0.757 against 0.500).
3. **On the validation set the loss starts earlier**: 0.6 points at level 30 and
   1.4 points at level 100. A likely reason is the way the dataset was built.
   The validation reviews are a random part of the training file, so they share
   movies with the training reviews, and rare words such as titles and names can
   help there. The test file holds reviews of other movies, where those words
   are useless. We did not test this explanation directly, because the data
   files do not say which movie a review is about.
4. **Accuracy hides part of the change.** At level 100 test accuracy is
   unchanged, yet 5.9% of the predictions differ from the reference run. Two
   runs that differ only in their seed already disagree on 2.8%. So removing the
   rare words changes which reviews the model gets wrong, even while the number
   of errors stays the same.

## 8. N-gram ablation (`ngram_ablation.py`)

### Design

- Two arms: unigrams (the reference) and unigrams plus bigrams.
- Both arms keep an n-gram when it appears in at least 5 training reviews. The
  second arm therefore contains every unigram feature of the first arm plus the
  bigram features. The only thing added is the bigrams.
- Training size, review text, and hyperparameters stay at the reference setting.
- The gain is computed seed by seed and then averaged, so its standard deviation
  says how stable the effect is.
- A model takes part only if its `run_model` accepts the feature settings.
  Otherwise it is saved as `not_implemented` with the reason. It is never run
  silently with the wrong features.

### Results

(`ngram-ablation-summary.csv`, figure `figures/ngram-ablation`)

| Evaluated on | Model | Features | Number of features | Accuracy | Gain from bigrams (points) | Seconds per run |
| --- | --- | --- | --- | --- | --- | --- |
| validation | MLP | unigram | 25,921 | 0.9035 ± 0.0017 | | 8 |
| validation | MLP | unigram+bigram | 142,295 | 0.9169 ± 0.0010 | +1.35 ± 0.16 | 60 |
| test | MLP | unigram | 27,271 | 0.8855 ± 0.0007 | | 13 |
| test | MLP | unigram+bigram | 155,437 | 0.9061 ± 0.0006 | +2.06 ± 0.07 | 85 |

The majority baseline scores 0.500 in both arms.

### How to read it

- Bigrams help the MLP by 2.1 points on the test set and 1.4 points on the
  validation set. Both gains are many times larger than their spread across
  seeds.
- The price, on the test runs, is 5.7 times more input features and about 6.5
  times more training time per run.
- The MLP's hyperparameters were tuned on unigram features and reused. With
  tuning on the larger input the gain could be different.

## 9. Error analysis (`error_analysis.py`)

### Design

Three properties are measured for every evaluation review, from its text alone.
The rules were fixed before any model error on the test set was looked at, and
the code was developed on validation predictions.

| Property | Measure | How it is computed |
| --- | --- | --- |
| Negation | Negation cues per 100 words | A cue is one of "not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor", "cannot", "without", "hardly", "barely", "scarcely", or any word ending in "n't". |
| Rare words | Share of the review's words that the reference features cannot use | Words that appear in fewer than 5 training reviews, or in none. |
| Ambiguous review | Mix of positive and negative words, from 0 (one-sided) to 0.5 (evenly mixed) | Two lists of 300 words each: the words most typical of positive training reviews and the words most typical of negative training reviews. The measure is the smaller count divided by the sum of both counts. |

The two word lists are learned from the training labels only. Each word that
appears in at least 100 training reviews is scored by how much more often it
appears in positive than in negative reviews, and the 300 highest and 300 lowest
scores are kept. No outside sentiment dictionary is used.

For each property the evaluation reviews are split into four groups of equal
size, from the lowest values (Q1) to the highest (Q4). A property matters when
the error rate rises from Q1 to Q4. Ties are split at random with a fixed seed.
They must not be split by position in the file, because the data files are
sorted by label.

The same table is computed for the unigram+bigram arm. That connects the error
categories to the ablation: if bigrams repair a kind of error, the error rate
should fall most in the group where that kind of error is concentrated.

### Results for the MLP on the test set

(`error-analysis-groups.csv`, figures `figures/error-analysis-groups` and
`figures/error-analysis-bigram-effect`)

| Property | Group | Range of the measure | Positive reviews | Error rate, unigrams (%) | Error rate, with bigrams (%) | Change (points) |
| --- | --- | --- | --- | --- | --- | --- |
| All reviews | | | 50% | 11.5 ± 0.1 | 9.4 ± 0.1 | -2.1 |
| Negation | Q1 (lowest) | 0.0 to 0.9 | 66% | 11.0 ± 0.1 | 9.1 ± 0.1 | -1.8 |
| Negation | Q2 | 0.9 to 1.6 | 55% | 11.9 ± 0.1 | 9.7 ± 0.1 | -2.1 |
| Negation | Q3 | 1.6 to 2.5 | 45% | 11.4 ± 0.1 | 9.3 ± 0.2 | -2.1 |
| Negation | Q4 (highest) | 2.5 to 15.4 | 34% | 11.6 ± 0.1 | 9.4 ± 0.1 | -2.2 |
| Rare words | Q1 (lowest) | 0.000 to 0.007 | 46% | 10.0 ± 0.1 | 8.0 ± 0.0 | -1.9 |
| Rare words | Q2 | 0.007 to 0.017 | 47% | 10.6 ± 0.1 | 8.1 ± 0.1 | -2.5 |
| Rare words | Q3 | 0.017 to 0.029 | 49% | 11.2 ± 0.1 | 9.7 ± 0.0 | -1.5 |
| Rare words | Q4 (highest) | 0.029 to 0.234 | 58% | 14.0 ± 0.2 | 11.7 ± 0.1 | -2.3 |
| Mixed polarity | Q1 (lowest) | 0.000 to 0.000 | 45% | 7.8 ± 0.1 | 6.7 ± 0.0 | -1.1 |
| Mixed polarity | Q2 | 0.000 to 0.200 | 35% | 5.4 ± 0.1 | 4.6 ± 0.1 | -0.8 |
| Mixed polarity | Q3 | 0.200 to 0.333 | 55% | 12.9 ± 0.2 | 10.8 ± 0.2 | -2.1 |
| Mixed polarity | Q4 (highest) | 0.333 to 0.500 | 64% | 19.4 ± 0.1 | 15.2 ± 0.1 | -4.2 |

Each group has 6,250 reviews (6,167 or 6,168 for mixed polarity, because 330
reviews contain no word from either list and are left out).

### How to read it

1. **Mixed reviews are where the MLP fails.** The error rate is 19.4% in the most
   mixed quarter against 5.4% in Q2, more than three times higher. The seeds also
   disagree most there (8.0% of the reviews, against 1.3% in Q2).
2. **Rare words matter a little.** The error rate rises from 10.0% to 14.0%
   between the quarter with the fewest and the quarter with the most unusable
   words.
3. **Negation does not show up with this measure.** The error rate is flat, from
   11.0% to 11.9%, whether a review has few or many negation cues. Counting cues
   is a coarse measure, since it does not know which words a cue applies to, so
   this is not proof that negation is harmless. It does say that reviews with
   much negation are not harder for the MLP on average.
4. **Bigrams help everywhere, and most in mixed reviews.** Adding bigrams lowers
   the error rate by about 2 points in every negation group and every rare-word
   group. In the most mixed quarter it lowers it by 4.2 points, against about 1
   point in the two one-sided quarters. The gain from bigrams found in section 8
   is therefore not specific to reviews with many negation cues.
5. **Watch the class balance of the groups.** Negative reviews use more
   negation: only 34% of the reviews in the top negation quarter are positive.
   The column "Positive reviews" is there so that a difference between groups is
   not read as an effect of the property when it could be an effect of the label.

The validation set shows the same pattern for mixed polarity (16.6% in Q4 against
5.2% in Q2), with more noise in the other rows because each group has only about
625 reviews.

### Representative errors

`error-analysis-examples.csv` lists, for each property, five reviews from the top
quarter that the MLP gets wrong with all three seeds, alternating between
negative and positive reviews. Only reviews of 40 to 150 words are listed, so
they can be read and quoted. Each row has the cue words that were found, the full
text, and the number of seeds that still get it wrong with bigrams.

Of the 15 test-set examples, 3 are corrected by all three seeds once bigrams are
added and 10 are still wrong with all three. These reviews were picked because
they are hard, so this count says nothing about how often bigrams help overall;
the table above does.

## 10. Checks for data leakage and other sanity checks

The validation accuracy (0.9035) is higher than the test accuracy (0.8855), so
the possible leakage routes were checked one by one.

| Check | Result |
| --- | --- |
| Do the training and validation partitions share reviews? | No review ID is in both. |
| Is anything fit on evaluation data? | No. The vectorizer and the MLP are one pipeline fit on training reviews only; a unit test confirms that a word seen only in evaluation reviews never enters the vocabulary and that evaluation labels are never read. |
| Does the model learn from shuffled labels? | No. Trained on the same reviews with shuffled labels, the MLP scores 0.505 on validation, which is chance. |
| Do repeated texts inflate the scores? | Hardly. 22 validation reviews have an identical copy in the training partition; without them validation accuracy is 0.9026 instead of 0.9035. 123 test reviews have an identical copy in the training file; without them test accuracy is 0.8851 instead of 0.8855. |
| Does the search touch the test file? | No. The tuning script does not load it. |
| Are the runs repeatable? | Yes. The reference condition was run three separate times (training-size run, stress level 1, ablation unigram arm) and gave identical predictions for every review and seed. |

What explains the gap, then:

- **The dataset.** According to its README, the training and test files contain
  reviews of different movies. Our validation set is a random part of the
  training file, so it shares movies with the training reviews. A model can use
  movie-specific words on validation but not on test. The stress test agrees
  with this (section 7, point 3).
- **Selection.** The configuration was chosen for having the highest validation
  score among nine that were about equally good.
- **Sample size.** 2,500 validation reviews give a sampling error of about 0.6
  points.

So 0.8855 ± 0.0007 is the number to report for the MLP at the reference setting.

## 11. Limitations

- Only the MLP and the baseline have results so far.
- Three seeds give a rough standard deviation. It measures how much training
  randomness moves the result. It does not include the sampling error of the
  evaluation set, which is about 0.6 points on validation and 0.2 on test.
- The search tested nine configurations and no combinations of changes.
- Hyperparameters were tuned once, at the reference setting, and reused. This
  keeps the comparison controlled, but the MLP may not be at its best with
  bigrams, with little data, or under stress.
- The stress levels are absolute counts of training reviews.
- The error-analysis measures are simple. Cue counting ignores what a negation
  applies to. The two polarity lists come from the data and contain topic words
  (genres, names) as well as opinion words. Reviews with no minority-polarity
  word at all (Q1 of mixed polarity) include very short reviews with a single
  cue word, which is why Q1 has more errors than Q2.
- The times are from a shared 2-core machine and are approximate.

## 12. For teammates: taking part in these experiments

- **Vocabulary stress test:** nothing to do. It works with any `run_model`.
- **N-gram ablation:** accept the feature settings and build the vectorizer from
  them, as shown under "Adding a model to the experiments" in the README.
- **Same input as the MLP:** call `build_vectorizer(features)`.
- Then re-run `vocabulary_stress`, `ngram_ablation`, `error_analysis`, and
  `visualize_analysis`. The tables and figures fill in by themselves. Each model
  has a fixed color and point shape in every figure.

## 13. Decisions the team still has to confirm

1. **Tuning budget.** The MLP used nine configurations. For a comparable budget,
   logistic regression and Naive Bayes should also try nine, selected by mean
   validation accuracy.
2. **Shared features.** Whether logistic regression and Naive Bayes adopt
   `features.py`. Without it the models are compared on different inputs and
   those two cannot take part in the ablation.
3. **Primary metric.** Accuracy is proposed here.
4. **Reporting the stress test and the ablation on the test set.** They are saved
   for both validation and test, like the main results. The settings (levels,
   arms, error categories) were fixed before any test result was computed.

## 14. Changes this work makes to shared files

- `experiment_runner.py`: `run_experiment` and `run_sweep` take an optional
  `features` argument. Without it they behave exactly as before.
- `models/control.py`: accepts and ignores `features`, so the baseline appears in
  the ablation.
- `README.md`: setup, dataset, protocol, commands, and outputs.
- New: `requirements.txt` (exported from `uv.lock`), `.gitignore`, `tests/`.
- The two prediction files of the main entry point are no longer tracked in git.
  With the baseline and the MLP they are about 121 MB and 130 MB, and GitHub
  rejects files over 100 MB. They are still written locally on every run.
- Compiled Python files (`__pycache__`) are no longer tracked either.

## 15. What was run to check this work

Run:

- All five experiments, end to end, on the full data (MLP and baseline).
- 55 unit tests on synthetic data: features, MLP, stress test, ablation, error
  analysis, harness partitions.
- The leakage checks in section 10.
- The main entry point (`ml_semester_project.main`) with the MLP, in a copy of
  the repository: it finished in about 3 minutes and produced the MLP charts.
- A fresh `pip` environment built from `requirements.txt`: the unit tests pass.
- A clean reproduction: a fresh clone of this branch (commit `ae08f11`) with a
  fresh `uv sync` environment re-ran every command in the README. Every result
  table and prediction file came out with identical values; only the wall-clock
  columns differ. The whole sequence took 21 minutes on 2 cores, and the main
  entry point gave the same MLP numbers as the separate MLP run.
- The PNG version of every figure was inspected.
- The data files were compared with the Hugging Face repository by SHA-256.
- A check against the team's current `main` (commit `943a774`, which removed
  unused files on November 19), run on November 22 in a new environment: the
  merge has no conflicts, the 55 unit tests pass, and the main entry point gives
  the same MLP results as the saved files in all 33 runs (identical accuracy,
  precision, recall, F1, and confusion matrices).

Not run:

- Logistic regression and Naive Bayes, which do not exist yet. The all-model code
  paths were exercised with stand-in models in the unit tests and in a small dry
  run, not with the real ones.
- The HTML figures were not opened in a browser. They are generated from the
  same chart definitions as the PNG files.
- Nothing was run on another machine or operating system.

## 16. Assistance disclosure

The code, the experiment runs, and this document were produced with Claude
(Anthropic), a generative AI assistant, working at Erik Vera's direction. The
project guidelines ask for the use of generative AI to be disclosed, so this
should also be stated in the team report. The results were obtained by running
the code in this repository; they are for the team to review before they are
reported.
