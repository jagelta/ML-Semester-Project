from dataclasses import asdict, dataclass
import inspect
import json
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from .features import FeatureConfig
from .models.control import run_model as run_control_model
from .models.logistic_regression import run_model as run_logistic_regression
from .models.mlp import run_model as run_mlp
from .models.naive_bayes import run_model as run_naive_bayes


@dataclass(frozen=True)
class ExperimentPlan:
    validation_fraction: float = 0.1
    split_seed: int = 420
    training_fractions: tuple[float, ...] = tuple(step / 10 for step in range(1, 11))
    repeat_seeds: tuple[int, ...] = (420, 421, 422)
    model_names: tuple[str, ...] = (
        "control",
        "logistic_regression",
        "mlp",
        "naive_bayes",
    )


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    train_fraction: float
    seed: int


ModelRunner = Callable[..., np.ndarray]
MODEL_RUNNERS: dict[str, ModelRunner] = {
    "control": run_control_model,
    "logistic_regression": run_logistic_regression,
    "mlp": run_mlp,
    "naive_bayes": run_naive_bayes,
}
PREDICTION_COLUMNS = [
    "evaluation_set",
    "model_name",
    "train_fraction",
    "seed",
    "split_seed",
    "validation_fraction",
    "review_id",
    "text",
    "true_label",
    "predicted_label",
]

EXPERIMENT_PLAN = ExperimentPlan()


def build_experiment_configs(plan: ExperimentPlan) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            model_name=model_name,
            train_fraction=fraction,
            seed=seed,
        )
        for model_name in plan.model_names
        for fraction in plan.training_fractions
        for seed in plan.repeat_seeds
    ]


def _split_training_data(
    data: pd.DataFrame,
    plan: ExperimentPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        data,
        test_size=plan.validation_fraction,
        random_state=plan.split_seed,
        stratify=data["label"],
    )


def _stratified_training_subset(
    train_pool: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be greater than 0 and at most 1")

    if fraction == 1:
        return train_pool.sample(
            frac=1.0,
            replace=False,
            random_state=seed,
        )

    subset, _ = train_test_split(
        train_pool,
        train_size=fraction,
        random_state=seed,
        stratify=train_pool["label"],
    )
    return subset


def _call_model_runner(
    model_name: str,
    training_subset: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    seed: int,
    features: FeatureConfig | None,
) -> np.ndarray:
    # With features=None the runner is called exactly as before. An experiment
    # that changes the features (the n-gram ablation) passes a FeatureConfig, and
    # only runners that declare a `features` parameter can take part in it.
    runner = MODEL_RUNNERS[model_name]
    if features is None:
        return runner(training_subset, evaluation_data, seed)
    if "features" not in inspect.signature(runner).parameters:
        raise NotImplementedError(
            f"The {model_name} model runner does not accept feature settings yet"
        )
    return runner(training_subset, evaluation_data, seed, features=features)


def run_experiment(
    config: ExperimentConfig,
    plan: ExperimentPlan,
    train_pool: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    evaluation_set: str,
    split_seed: int | None,
    features: FeatureConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    if config.model_name not in MODEL_RUNNERS:
        raise ValueError(f"Unknown model: {config.model_name}")

    run_details = {
        **asdict(config),
        "evaluation_set": evaluation_set,
        "split_seed": split_seed,
    }
    if evaluation_set == "validation":
        run_details["validation_fraction"] = plan.validation_fraction

    try:
        training_subset = _stratified_training_subset(
            train_pool,
            config.train_fraction,
            config.seed,
        )
        predictions = _call_model_runner(
            config.model_name,
            training_subset,
            evaluation_data,
            config.seed,
            features,
        )
    except NotImplementedError as error:
        return (
            {
                **run_details,
                "status": "not_implemented",
                "reason": str(error),
            },
            None,
        )

    actual = evaluation_data["label"].to_numpy()
    prediction_rows = evaluation_data[["review_id", "text", "label"]].rename(
        columns={"label": "true_label"}
    ).copy()
    prediction_rows["evaluation_set"] = evaluation_set
    prediction_rows["model_name"] = config.model_name
    prediction_rows["train_fraction"] = config.train_fraction
    prediction_rows["seed"] = config.seed
    prediction_rows["split_seed"] = split_seed
    prediction_rows["validation_fraction"] = (
        plan.validation_fraction if evaluation_set == "validation" else None
    )
    prediction_rows["predicted_label"] = predictions
    prediction_rows = prediction_rows[PREDICTION_COLUMNS]

    return (
        {
            **run_details,
            "status": "completed",
            "training_rows": len(training_subset),
            "accuracy": accuracy_score(actual, predictions),
            "precision_macro": precision_score(
                actual, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                actual, predictions, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(actual, predictions, average="macro"),
            "confusion_matrix": json.dumps(
                confusion_matrix(actual, predictions).tolist()
            ),
        },
        prediction_rows,
    )


def run_sweep(
    configs: list[ExperimentConfig],
    plan: ExperimentPlan,
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    evaluation_set: str,
    split_seed: int | None,
    features: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = [
        run_experiment(
            config,
            plan,
            training_data,
            evaluation_data,
            evaluation_set,
            split_seed,
            features,
        )
        for config in configs
    ]
    metrics = pd.DataFrame([metrics for metrics, _ in runs])
    prediction_frames = [
        predictions
        for _, predictions in runs
        if predictions is not None
    ]
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    return metrics, predictions


def run_validation_experiments(
    training_data: pd.DataFrame,
    plan: ExperimentPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_pool, validation_data = _split_training_data(training_data, plan)
    return run_sweep(
        build_experiment_configs(plan),
        plan,
        train_pool,
        validation_data,
        evaluation_set="validation",
        split_seed=plan.split_seed,
    )


def run_final_test_evaluations(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    plan: ExperimentPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = [
        ExperimentConfig(model_name=model_name, train_fraction=1.0, seed=seed)
        for model_name in plan.model_names
        for seed in plan.repeat_seeds
    ]
    return run_sweep(
        configs,
        plan,
        training_data,
        test_data,
        evaluation_set="test",
        split_seed=None,
    )
