"""Figures for the MLP tuning, training-size, vocabulary, n-gram, and error results.

Run after the experiments, as often as needed (it only reads the saved tables):

    uv run python -m ml_semester_project.visualize_analysis

Every figure is saved as an interactive HTML file with tooltips. If the optional
package ``vl-convert-python`` is installed, a PNG and a PDF are saved as well, for
slides and for the LaTeX report:

    uv run --with vl-convert-python python -m ml_semester_project.visualize_analysis

Each model keeps the same color and point shape in every figure, and every value
shown here is also in the summary CSV next to it.
"""

from pathlib import Path

import altair as alt
import pandas as pd

from .experiment_support import OUTPUT_DIR
from .features import REFERENCE_FEATURES, UNIGRAM_BIGRAM_FEATURES

FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_LABELS = {
    "control": "Majority baseline",
    "logistic_regression": "Logistic regression",
    "mlp": "MLP",
    "naive_bayes": "Naive Bayes",
}
# Fixed per model, so a model looks the same in every figure. The three hues stay
# distinguishable under the common forms of color blindness; the shapes repeat the
# same information without color.
MODEL_COLORS = {
    "control": "#898781",
    "logistic_regression": "#2a78d6",
    "mlp": "#eb6834",
    "naive_bayes": "#1baf7a",
}
MODEL_SHAPES = {
    "control": "cross",
    "logistic_regression": "circle",
    "mlp": "square",
    "naive_bayes": "triangle-up",
}
EVALUATION_LABELS = {"validation": "Validation set", "test": "Test set"}
CATEGORY_LABELS = {
    "negation": "Negation cues per 100 words",
    "rare_words": "Share of rare words",
    "mixed_polarity": "Mix of positive and negative words",
}
GROUP_ORDER = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
GROUP_NOTE = (
    "Reviews are split into four groups of equal size by each property, "
    "from lowest (Q1) to highest (Q4). "
)
FEATURE_COLORS = {
    REFERENCE_FEATURES.name: "#86b6ef",
    UNIGRAM_BIGRAM_FEATURES.name: "#1c5cab",
}

ACCENT = "#2a78d6"
CONTEXT = "#898781"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"


def _read(filename: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / filename
    return pd.read_csv(path) if path.exists() else None


def _with_error_bars(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    """Add the lower and upper ends of mean +/- one standard deviation."""
    spread = summary[f"{value}_std"].fillna(0)
    return summary.assign(
        lower=summary[f"{value}_mean"] - spread,
        upper=summary[f"{value}_mean"] + spread,
    )


def _present_models(data: pd.DataFrame) -> list[str]:
    return [name for name in MODEL_LABELS if name in set(data["model_name"])]


def _model_encodings(data: pd.DataFrame) -> tuple[alt.Color, alt.Shape]:
    models = _present_models(data)
    labels = [MODEL_LABELS[name] for name in models]
    color = alt.Color(
        "model:N",
        title="Model",
        scale=alt.Scale(domain=labels, range=[MODEL_COLORS[name] for name in models]),
    )
    shape = alt.Shape(
        "model:N",
        title="Model",
        scale=alt.Scale(domain=labels, range=[MODEL_SHAPES[name] for name in models]),
    )
    return color, shape


def _label_models(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        model=data["model_name"].map(MODEL_LABELS),
        evaluation=data["evaluation_set"].map(EVALUATION_LABELS),
    )


def _styled(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return (
        chart.configure(background=SURFACE)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=GRID,
            gridWidth=1,
            domainColor=AXIS,
            tickColor=AXIS,
            labelColor=SECONDARY_INK,
            titleColor=SECONDARY_INK,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
            titlePadding=8,
        )
        .configure_legend(
            labelColor=SECONDARY_INK,
            titleColor=SECONDARY_INK,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
            symbolSize=90,
            symbolStrokeWidth=2,
        )
        .configure_header(
            labelColor=INK,
            labelFontSize=13,
            labelFontWeight="bold",
            labelAnchor="start",
            title=None,
        )
        .configure_title(
            color=INK,
            fontSize=15,
            anchor="start",
            offset=14,
            subtitleColor=SECONDARY_INK,
            subtitleFontSize=12,
            subtitlePadding=6,
        )
    )


def tuning_chart(summary: pd.DataFrame) -> alt.TopLevelMixin:
    """Validation accuracy of the nine configurations; the selected one stands out."""

    def describe(row: pd.Series) -> str:
        changed = row["changed_hyperparameter"]
        if changed == "hidden_layer_sizes":
            layers = int(row["hidden_layers"])
            noun = "layer" if layers == 1 else "layers"
            return f"{layers} hidden {noun}: {row['hidden_layer_sizes']} units"
        if changed == "learning_rate_init":
            return f"Learning rate {row['learning_rate_init']:g}"
        if changed == "alpha":
            return f"L2 penalty {row['alpha']:g}"
        return "Starting point"

    data = _with_error_bars(summary, "accuracy").assign(
        label=summary.apply(describe, axis=1),
        choice=summary["selected"].map({True: "Selected", False: "Not selected"}),
    )
    order = data.sort_values("accuracy_mean", ascending=False)["label"].tolist()
    color = alt.Color(
        "choice:N",
        title=None,
        scale=alt.Scale(domain=["Selected", "Not selected"], range=[ACCENT, CONTEXT]),
    )
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    start = data.loc[data["changed_hyperparameter"].str.startswith("none")].iloc[0]
    starting_point = (
        f"Starting point: {int(start['hidden_layers'])} hidden layer of "
        f"{start['hidden_layer_sizes']} units, learning rate "
        f"{start['learning_rate_init']:g}, L2 penalty {start['alpha']:g}. "
        "Every other row changes one of these."
    )
    tooltip = [
        alt.Tooltip("label:N", title="Configuration"),
        alt.Tooltip("accuracy_mean:Q", title="Mean accuracy", format=".4f"),
        alt.Tooltip("accuracy_std:Q", title="Std over seeds", format=".4f"),
        alt.Tooltip("epochs_mean:Q", title="Mean epochs", format=".1f"),
        alt.Tooltip("fit_seconds_mean:Q", title="Mean seconds per fit", format=".0f"),
        alt.Tooltip("trainable_parameters:Q", title="Parameters", format=","),
    ]
    base = alt.Chart(data)
    spread = base.mark_rule(strokeWidth=2).encode(
        x=alt.X(
            "lower:Q",
            title="Validation accuracy (mean ± std over 3 seeds)",
            scale=alt.Scale(zero=False, padding=12),
            axis=alt.Axis(format=".3f", tickCount=6),
        ),
        x2="upper:Q",
        y=y,
        color=color,
    )
    points = base.mark_point(
        filled=True, size=110, opacity=1, stroke=SURFACE, strokeWidth=2
    ).encode(x="accuracy_mean:Q", y=y, color=color, tooltip=tooltip)
    best = data.loc[data["selected"]]
    value = (
        alt.Chart(best)
        .mark_text(align="left", dx=10, dy=-12, fontSize=12, color=INK)
        .encode(
            x="accuracy_mean:Q", y=y, text=alt.Text("accuracy_mean:Q", format=".3f")
        )
    )
    return _styled(
        (spread + points + value).properties(
            title=alt.Title(
                "MLP tuning: nine configurations, selected on validation data",
                subtitle=starting_point,
            ),
            width=400,
            height=alt.Step(30),
        )
    )


def training_size_chart(summary: pd.DataFrame) -> alt.TopLevelMixin:
    """MLP accuracy against the number of training reviews."""
    data = _with_error_bars(summary, "accuracy").assign(
        evaluation=summary["evaluation_set"].map(EVALUATION_LABELS)
    )
    validation = data.loc[data["evaluation_set"].eq("validation")]
    color = alt.Color(
        "evaluation:N",
        title="Evaluated on",
        scale=alt.Scale(
            domain=list(EVALUATION_LABELS.values()),
            range=[MODEL_COLORS["mlp"], INK],
        ),
    )
    shape = alt.Shape(
        "evaluation:N",
        title="Evaluated on",
        scale=alt.Scale(
            domain=list(EVALUATION_LABELS.values()), range=["square", "diamond"]
        ),
    )
    x = alt.X(
        "training_rows:Q",
        title="Training reviews",
        axis=alt.Axis(format=",d", tickCount=6),
        scale=alt.Scale(domain=[0, 1.08 * data["training_rows"].max()], nice=False),
    )
    y_title = "Accuracy (mean ± std over 3 seeds)"
    tooltip = [
        alt.Tooltip("evaluation:N", title="Evaluated on"),
        alt.Tooltip("training_rows:Q", title="Training reviews", format=","),
        alt.Tooltip("accuracy_mean:Q", title="Mean accuracy", format=".4f"),
        alt.Tooltip("accuracy_std:Q", title="Std over seeds", format=".4f"),
    ]
    line = (
        alt.Chart(validation)
        .mark_line(strokeWidth=2, color=MODEL_COLORS["mlp"])
        .encode(
            x=x, y=alt.Y("accuracy_mean:Q", title=y_title, scale=alt.Scale(zero=False))
        )
    )
    spread = (
        alt.Chart(data)
        .mark_rule(strokeWidth=1.5)
        .encode(x=x, y=alt.Y("lower:Q", title=y_title), y2="upper:Q", color=color)
    )
    # A diamond looks smaller than a square of the same area, so it is drawn larger.
    size = alt.Size(
        "evaluation:N",
        legend=None,
        scale=alt.Scale(domain=list(EVALUATION_LABELS.values()), range=[110, 190]),
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, opacity=1, stroke=SURFACE, strokeWidth=2)
        .encode(
            x=x,
            y="accuracy_mean:Q",
            color=color,
            shape=shape,
            size=size,
            tooltip=tooltip,
        )
    )
    # Label only the two full-data results; the axis carries the other values.
    full_data = data.loc[data["train_fraction"].eq(1.0)]
    values = (
        alt.Chart(full_data)
        .mark_text(dy=-16, fontSize=12, color=INK)
        .encode(
            x=x, y="accuracy_mean:Q", text=alt.Text("accuracy_mean:Q", format=".3f")
        )
    )
    return _styled(
        (line + spread + points + values).properties(
            title=alt.Title(
                "MLP accuracy by training-set size",
                subtitle="Validation: trained on a share of 22,500 reviews. "
                "Test: trained on all 25,000",
            ),
            width=440,
            height=280,
        )
    )


def vocabulary_stress_chart(summary: pd.DataFrame) -> alt.TopLevelMixin:
    """Accuracy of every model as rarer and rarer words are removed."""
    data = _label_models(_with_error_bars(summary, "accuracy"))
    # The majority baseline never reads the text, so it is a flat line at its
    # accuracy. Naming it in the subtitle leaves the axis to the other models.
    baseline = data.loc[data["model_name"].eq("control"), "accuracy_mean"]
    baseline_note = (
        f" The majority baseline stays at {baseline.mean():.3f}."
        if len(baseline)
        else ""
    )
    data = data.loc[data["model_name"].ne("control")]
    color, shape = _model_encodings(data)
    levels = sorted(data["min_document_frequency"].unique())
    x = alt.X(
        "min_document_frequency:Q",
        title="Minimum training reviews per word (log scale)",
        scale=alt.Scale(
            type="log", domain=[levels[0] / 1.4, levels[-1] * 1.4], nice=False
        ),
        axis=alt.Axis(values=levels, format=",d", grid=False),
    )
    y_title = "Accuracy (mean ± std over 3 seeds)"
    tooltip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("evaluation:N", title="Evaluated on"),
        alt.Tooltip("min_document_frequency:Q", title="Minimum reviews per word"),
        alt.Tooltip(
            "shared_feature_count:Q", title="Unigram features left", format=","
        ),
        alt.Tooltip(
            "evaluation_words_removed_share:Q", title="Words removed", format=".1%"
        ),
        alt.Tooltip("accuracy_mean:Q", title="Mean accuracy", format=".4f"),
        alt.Tooltip("accuracy_std:Q", title="Std over seeds", format=".4f"),
        alt.Tooltip(
            "prediction_change_rate_mean:Q", title="Predictions changed", format=".1%"
        ),
    ]
    lines = (
        alt.Chart()
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("accuracy_mean:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=color,
        )
    )
    spread = (
        alt.Chart()
        .mark_rule(strokeWidth=1.5)
        .encode(x=x, y=alt.Y("lower:Q", title=y_title), y2="upper:Q", color=color)
    )
    points = (
        alt.Chart()
        .mark_point(filled=True, size=80, opacity=1, stroke=SURFACE, strokeWidth=2)
        .encode(x=x, y="accuracy_mean:Q", color=color, shape=shape, tooltip=tooltip)
    )
    panels = (
        alt.layer(lines, spread, points, data=data)
        .properties(width=330, height=270)
        .facet(
            column=alt.Column(
                "evaluation:N", sort=list(EVALUATION_LABELS.values()), title=None
            )
        )
    )
    return _styled(
        panels.properties(
            title=alt.Title(
                "Vocabulary stress test: rare words removed from training and "
                "evaluation reviews",
                subtitle="1 is the reference (nothing removed). Every model is "
                "retrained at every level." + baseline_note,
            )
        )
    )


def ngram_ablation_chart(summary: pd.DataFrame) -> alt.TopLevelMixin | None:
    """Accuracy with unigrams and with unigrams plus bigrams, one row per model."""
    data = _label_models(_with_error_bars(summary, "accuracy"))
    # The majority baseline ignores the text, so it is 0.5 with either feature set.
    data = data.loc[data["model_name"].ne("control")]
    if data.empty:
        return None
    models = [MODEL_LABELS[name] for name in _present_models(data)]
    arms = list(FEATURE_COLORS)
    color = alt.Color(
        "features:N",
        title="Features",
        scale=alt.Scale(domain=arms, range=list(FEATURE_COLORS.values())),
    )
    y = alt.Y("model:N", sort=models, title=None)
    x_title = "Accuracy (mean ± std over 3 seeds)"
    tooltip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("evaluation:N", title="Evaluated on"),
        alt.Tooltip("features:N", title="Features"),
        alt.Tooltip("shared_feature_count:Q", title="Number of features", format=","),
        alt.Tooltip("accuracy_mean:Q", title="Mean accuracy", format=".4f"),
        alt.Tooltip("accuracy_std:Q", title="Std over seeds", format=".4f"),
        alt.Tooltip(
            "accuracy_gain_from_bigrams_mean:Q",
            title="Gain from bigrams",
            format="+.4f",
        ),
        alt.Tooltip("run_seconds_mean:Q", title="Mean seconds per run", format=".0f"),
    ]
    connector = (
        alt.Chart()
        .mark_rule(strokeWidth=2, color=AXIS)
        .encode(
            x=alt.X(
                "min(accuracy_mean):Q",
                title=x_title,
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(format=".3f", tickCount=5),
            ),
            x2="max(accuracy_mean):Q",
            y=y,
        )
    )
    spread = (
        alt.Chart()
        .mark_rule(strokeWidth=1.5)
        .encode(x=alt.X("lower:Q", title=x_title), x2="upper:Q", y=y, color=color)
    )
    points = (
        alt.Chart()
        .mark_point(filled=True, size=120, opacity=1, stroke=SURFACE, strokeWidth=2)
        .encode(x="accuracy_mean:Q", y=y, color=color, tooltip=tooltip)
    )
    value = (
        alt.Chart()
        .mark_text(dy=-14, fontSize=11, color=SECONDARY_INK)
        .encode(
            x="accuracy_mean:Q", y=y, text=alt.Text("accuracy_mean:Q", format=".3f")
        )
    )
    panels = (
        alt.layer(connector, spread, points, value, data=data)
        .properties(width=330, height=alt.Step(64))
        .facet(
            column=alt.Column(
                "evaluation:N", sort=list(EVALUATION_LABELS.values()), title=None
            )
        )
    )
    return _styled(
        panels.properties(
            title=alt.Title(
                "Feature ablation: unigrams versus unigrams plus bigrams",
                subtitle="Same data and settings; only the n-gram range changes",
            )
        )
    )


def _error_groups(
    groups: pd.DataFrame, evaluation_set: str, condition: str
) -> pd.DataFrame:
    selected = groups.loc[
        groups["evaluation_set"].eq(evaluation_set)
        & groups["condition"].eq(condition)
        & groups["category"].isin(list(CATEGORY_LABELS))
        & groups["model_name"].ne("control")
    ]
    return _label_models(selected).assign(
        property=selected["category"].map(CATEGORY_LABELS)
    )


def error_groups_chart(
    groups: pd.DataFrame, evaluation_set: str
) -> alt.TopLevelMixin | None:
    """Error rate in the four groups of each review property (reference setting)."""
    data = _error_groups(groups, evaluation_set, "reference")
    if data.empty:
        return None
    data = _with_error_bars(data, "error_rate")
    color, shape = _model_encodings(data)
    x = alt.X("group:N", sort=GROUP_ORDER, title=None, axis=alt.Axis(labelAngle=0))
    y_title = "Error rate (mean ± std over 3 seeds)"
    tooltip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("property:N", title="Property"),
        alt.Tooltip("group:N", title="Group"),
        alt.Tooltip("reviews:Q", title="Reviews", format=","),
        alt.Tooltip("measure_min:Q", title="Lowest value in group", format=".3f"),
        alt.Tooltip("measure_max:Q", title="Highest value in group", format=".3f"),
        alt.Tooltip("positive_share:Q", title="Positive reviews", format=".1%"),
        alt.Tooltip("error_rate_mean:Q", title="Mean error rate", format=".4f"),
        alt.Tooltip("error_rate_std:Q", title="Std over seeds", format=".4f"),
    ]
    lines = (
        alt.Chart()
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y(
                "error_rate_mean:Q",
                title=y_title,
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domainMin=0),
            ),
            color=color,
        )
    )
    spread = (
        alt.Chart()
        .mark_rule(strokeWidth=1.5)
        .encode(x=x, y=alt.Y("lower:Q", title=y_title), y2="upper:Q", color=color)
    )
    points = (
        alt.Chart()
        .mark_point(filled=True, size=80, opacity=1, stroke=SURFACE, strokeWidth=2)
        .encode(x=x, y="error_rate_mean:Q", color=color, shape=shape, tooltip=tooltip)
    )
    panels = (
        alt.layer(lines, spread, points, data=data)
        .properties(width=230, height=250)
        .facet(
            column=alt.Column(
                "property:N", sort=list(CATEGORY_LABELS.values()), title=None
            )
        )
    )
    return _styled(
        panels.properties(
            title=alt.Title(
                f"Where the errors are: {EVALUATION_LABELS[evaluation_set].lower()}, "
                "reference setting",
                subtitle=GROUP_NOTE
                + "A property matters when the error rate rises from Q1 to Q4.",
            )
        )
    )


def bigram_effect_chart(
    groups: pd.DataFrame, evaluation_set: str
) -> alt.TopLevelMixin | None:
    """Change in error rate when bigrams are added, per group of each property."""
    data = _error_groups(groups, evaluation_set, UNIGRAM_BIGRAM_FEATURES.name)
    if data.empty:
        return None
    data = data.assign(change_points=100 * data["error_rate_change_from_reference"])
    color, _ = _model_encodings(data)
    models = [MODEL_LABELS[name] for name in _present_models(data)]
    x = alt.X("group:N", sort=GROUP_ORDER, title=None, axis=alt.Axis(labelAngle=0))
    tooltip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("property:N", title="Property"),
        alt.Tooltip("group:N", title="Group"),
        alt.Tooltip("reviews:Q", title="Reviews", format=","),
        alt.Tooltip("error_rate_mean:Q", title="Error rate with bigrams", format=".4f"),
        alt.Tooltip("change_points:Q", title="Change (points)", format="+.2f"),
    ]
    bars = (
        alt.Chart()
        .mark_bar(size=18, cornerRadiusEnd=4)
        .encode(
            x=x,
            xOffset=alt.XOffset("model:N", sort=models),
            y=alt.Y(
                "change_points:Q",
                title="Change in error rate (percentage points)",
                axis=alt.Axis(format="+.1f"),
            ),
            color=color,
            tooltip=tooltip,
        )
    )
    zero = alt.Chart().mark_rule(color=AXIS, strokeWidth=1).encode(y=alt.datum(0))
    panels = (
        alt.layer(bars, zero, data=data)
        .properties(width=230, height=250)
        .facet(
            column=alt.Column(
                "property:N", sort=list(CATEGORY_LABELS.values()), title=None
            )
        )
    )
    return _styled(
        panels.properties(
            title=alt.Title(
                "What bigrams change: error rate with bigrams minus error rate "
                f"without ({EVALUATION_LABELS[evaluation_set].lower()})",
                subtitle=GROUP_NOTE + "Below zero means fewer errors with bigrams.",
            )
        )
    )


def _save(chart: alt.TopLevelMixin, name: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{name}.html"]
    chart.save(paths[0])
    try:
        import vl_convert  # noqa: F401
    except ImportError:
        return paths
    for suffix, options in ((".png", {"scale_factor": 3}), (".pdf", {})):
        path = FIGURE_DIR / f"{name}{suffix}"
        chart.save(path, **options)
        paths.append(path)
    return paths


def save_analysis_figures() -> list[Path]:
    """Build every figure whose summary table exists and return the saved paths."""
    charts: dict[str, alt.TopLevelMixin | None] = {}
    tuning = _read("mlp-tuning-summary.csv")
    if tuning is not None:
        charts["mlp-tuning"] = tuning_chart(tuning)
    training_size = _read("mlp-training-size-summary.csv")
    if training_size is not None:
        charts["mlp-training-size"] = training_size_chart(training_size)
    stress = _read("vocabulary-stress-summary.csv")
    if stress is not None:
        charts["vocabulary-stress"] = vocabulary_stress_chart(stress)
    ablation = _read("ngram-ablation-summary.csv")
    if ablation is not None:
        charts["ngram-ablation"] = ngram_ablation_chart(ablation)
    groups = _read("error-analysis-groups.csv")
    if groups is not None:
        charts["error-analysis-groups"] = error_groups_chart(groups, "test")
        charts["error-analysis-bigram-effect"] = bigram_effect_chart(groups, "test")

    paths = []
    for name, chart in charts.items():
        if chart is not None:
            paths.extend(_save(chart, name))
    return paths


def main() -> None:
    paths = save_analysis_figures()
    for path in paths:
        print(f"Saved {path}")
    if paths and not any(path.suffix == ".png" for path in paths):
        print(
            "Only HTML figures were saved. For PNG and PDF, run with: "
            "uv run --with vl-convert-python python -m ml_semester_project.visualize_analysis"
        )


if __name__ == "__main__":
    main()
