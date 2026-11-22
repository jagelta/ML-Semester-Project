# ML-Semester-Project

Team 1 semester project: sentiment analysis on IMDb movie reviews.

**Research question.** Which text model remains most reliable when training data
shrink or the vocabulary changes?

**Models compared.** A majority-class baseline, logistic regression, an MLP with
one to three hidden layers, and Naive Bayes. All of them plug into one experiment
harness, so they share the data split, the seeds, and the metrics.

## Dataset

IMDb Large Movie Review Dataset (Maas et al., 2011), Hugging Face copy
[`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb),
configuration `plain_text`. The two labeled files are stored in
`src/ml_semester_project/data/`, so the experiments run offline.

| File | Reviews | Labels | SHA-256 |
| --- | --- | --- | --- |
| `train-00000-of-00001.parquet` | 25,000 | 12,500 negative, 12,500 positive | `db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c` |
| `test-00000-of-00001.parquet` | 25,000 | 12,500 negative, 12,500 positive | `b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da` |

- Columns: `text` (the review) and `label` (0 = negative, 1 = positive).
- The checksums match the files published in the Hugging Face repository
  (checked on November 18, 2026). The `unsupervised` split is not used.
- Known properties of the data: 96 training reviews and 199 test reviews repeat
  a text that appears earlier in the same file, and 123 test reviews have an
  identical copy in the training file (always with the same label). We keep the
  standard files unchanged; this affects at most 0.5% of the test set.
- According to the dataset README, the training and test files contain reviews
  of different movies. Our validation set is a random part of the training file,
  so it shares movies with the training reviews. Validation accuracy is therefore
  a little higher than test accuracy.

Citation:

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

## Setup

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/). The exact
package versions are locked in `uv.lock`.

```bash
uv sync
```

Without uv, the same versions are listed in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Versions used for the saved results: Python 3.13.18, scikit-learn 1.9.1,
NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1, PyArrow 26.0.1, Altair 6.3.0.

## Experimental protocol

| Decision | Setting |
| --- | --- |
| Primary metric | Accuracy (every partition is balanced, so chance is 0.5). Macro precision, recall, F1, and the confusion matrix are saved too. |
| Validation set | Stratified 10% of the training file, `split_seed = 420` (22,500 training reviews, 2,500 validation reviews) |
| Test set | The separate test file. It is used for final reporting only, never for tuning. |
| Repeat seeds | 420, 421, 422. Results are reported as mean ± sample standard deviation over the three runs. |
| Reference setting | Full training data, unchanged review text, unigram features, each model's selected hyperparameters |
| One factor at a time | Every experiment changes one thing from the reference setting. |
| Tuning | Hyperparameters are selected once, on validation data at the reference setting, and reused in every experiment. |

For validation results a model trains on the 22,500-review partition. For test
results it trains on all 25,000 training reviews.

## Running the experiments

Run the commands from the repository root. Times are for a 2-core CPU without a
GPU.

| Step | Command | Time | Models |
| --- | --- | --- | --- |
| Training-size sweep and final test, all models | `uv run ml-semester-project` | about 3 min with the baseline and the MLP | all |
| MLP hyperparameter search (validation only) | `uv run python -m ml_semester_project.mlp_tuning` | about 10 min | MLP |
| MLP training-size run, saved in its own files | `uv run python -m ml_semester_project.mlp_training_size` | about 4 min | MLP |
| Vocabulary stress test (rare-word removal) | `uv run python -m ml_semester_project.vocabulary_stress` | about 5 min | all |
| Unigram versus unigram+bigram ablation | `uv run python -m ml_semester_project.ngram_ablation` | about 10 min | all |
| Error analysis (reads the two previous results) | `uv run python -m ml_semester_project.error_analysis` | under 1 min | all |
| Figures | `uv run --with vl-convert-python python -m ml_semester_project.visualize_analysis` | under 1 min | all |
| Unit tests | `uv run python -m unittest discover -s tests -t tests` | seconds | |

A model that is not implemented yet is recorded with status `not_implemented`
and the experiments continue with the others. Re-running a command after a model
is added fills in its rows.

The figures command writes interactive HTML files. With the optional
`vl-convert-python` package (added by `--with` above, without changing the
project dependencies) it also writes PNG and PDF files for slides and LaTeX.

## Results

Everything is written to `src/ml_semester_project/output/`.

| Files | Content |
| --- | --- |
| `validation-results.csv`, `test-results.csv`, `master-*.html`, `<model>-*.html` | Main entry point: training-size sweep and final test evaluation for all models |
| `validation-predictions.parquet`, `test-predictions.parquet` | Main entry point: one row per review and run, with the review text. Written locally, not tracked in git (see the note below). |
| `mlp-tuning-results.csv`, `mlp-tuning-summary.csv` | The nine MLP configurations, per seed and ranked |
| `mlp-training-size-results.csv`, `mlp-training-size-summary.csv`, `mlp-training-size-predictions.parquet` | MLP accuracy for ten training fractions (validation) and at full size (test) |
| `vocabulary-stress-results.csv`, `vocabulary-stress-summary.csv`, `vocabulary-stress-predictions.parquet` | Every model at seven levels of rare-word removal |
| `ngram-ablation-results.csv`, `ngram-ablation-summary.csv`, `ngram-ablation-predictions.parquet` | Every model with unigram and with unigram+bigram features |
| `error-analysis-groups.csv`, `error-analysis-examples.csv` | Error rates by negation, rare words, and mixed polarity; representative errors |
| `figures/` | One figure per result, as HTML, PNG, and PDF |

The two prediction files of the main entry point repeat the full review text
in every row. With the baseline and the MLP they are already about 121 MB and
130 MB, above GitHub's limit of 100 MB per file, so they are listed in
`.gitignore`. They are regenerated by every run of the main entry point.

`*-results.csv` files have one row per run (model, condition, seed).
`*-summary.csv` files have the mean and standard deviation over seeds. The
prediction files of the analysis experiments store `review_id`, the true label,
and the predicted label; `review_id` is the row position in the data file, so the
text can be joined back.

The design decisions behind the MLP, the vocabulary stress test, the ablation,
and the error analysis, with the result tables, are explained in
[`docs/mlp-vocabulary-ngram-experiments.md`](docs/mlp-vocabulary-ngram-experiments.md).

## Repository layout

```
src/ml_semester_project/
  __init__.py              main entry point (training-size sweep, final test, charts)
  experiment_runner.py     experiment plan, split, training subsets, metrics
  features.py              shared text features (tokens, n-grams, vocabulary rule)
  models/                  one run_model function per model
  experiment_support.py    helpers shared by the analysis experiments
  mlp_tuning.py            MLP hyperparameter search
  mlp_training_size.py     MLP training-size run
  vocabulary_stress.py     rare-word removal stress test
  ngram_ablation.py        unigram versus unigram+bigram ablation
  error_analysis.py        error categories and representative errors
  visualize_results.py     charts of the main entry point
  visualize_analysis.py    figures of the analysis experiments
  data/                    IMDb train and test files
  output/                  result tables, predictions, and figures
tests/                     unit tests on small synthetic data
docs/                      design notes
```

## Adding a model to the experiments

Every model is a function with this interface, registered in `MODEL_RUNNERS` in
`experiment_runner.py`:

```python
def run_model(training_data, evaluation_data, seed) -> np.ndarray:  # predicted labels
```

The training-size sweep and the vocabulary stress test need nothing else. The
stress test edits the review text before the model sees it, so it works with any
model.

To take part in the n-gram ablation, a model also accepts the feature settings
and builds its vectorizer from them. Using the shared vectorizer gives every
model the same tokens and the same vocabulary:

```python
from ..features import REFERENCE_FEATURES, FeatureConfig, build_vectorizer

def run_model(training_data, evaluation_data, seed, features: FeatureConfig = REFERENCE_FEATURES):
    vectorizer = build_vectorizer(features)             # TF-IDF; weighting="count" for raw counts
    train_matrix = vectorizer.fit_transform(training_data["text"])   # fit on training reviews only
    evaluation_matrix = vectorizer.transform(evaluation_data["text"])
    ...
```

## Reproducibility notes

- All randomness comes from the seeds in `ExperimentPlan` (`experiment_runner.py`).
- With the same package versions on the same machine, repeated runs give
  identical numbers. Across machines, floating-point differences can change the
  last digits of the MLP results.
- The `fit_seconds` and `run_seconds` columns record wall-clock time, so they
  change from run to run. Every other column is reproducible.
- The saved results were produced on November 18, 2026 on a 2-core Linux machine.
