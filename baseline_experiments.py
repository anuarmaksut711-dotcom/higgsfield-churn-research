"""Leakage-safe baseline comparison on common walk-forward temporal folds."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from churn_pipeline import (
    ARTIFACT_DIR,
    ID_COL,
    RANDOM_SEED,
    TARGET,
    assemble_dataset,
    build_model,
    build_walk_forward_folds,
    load_tables,
    metric_record,
    prepare_features,
    validate_dataset_invariants,
)

LOG = logging.getLogger("baseline_experiments")


def simple_catboost_columns(columns: list[str]) -> list[str]:
    """Minimal, interpretable feature set without temporal/cross interactions."""
    exact = {
        "subscription_plan",
        "country_code",
        "attempt_count",
        "attempt_success_count",
        "attempt_failure_count",
        "attempt_failure_rate",
        "attempt_amount_sum",
        "attempt_amount_mean",
        "purchase_count",
        "purchase_amount_sum",
        "purchase_amount_mean",
    }
    return [column for column in columns if column in exact or column.startswith("quiz_")]


def logistic_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=5,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="saga",
                    tol=1e-3,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def tree_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def add_metric_row(
    rows: list[dict[str, object]],
    model_name: str,
    fold,
    y_true: pd.Series,
    prediction: np.ndarray,
) -> None:
    rows.append(
        {
            "model": model_name,
            "fold": fold.name,
            "train_users": len(fold.train_idx),
            "evaluation_users": len(fold.evaluation_idx),
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "evaluation_start": fold.evaluation_start,
            "evaluation_end": fold.evaluation_end,
            **metric_record(y_true, prediction),
        }
    )


def run(iterations: int, thread_count: int, only: str | None = None) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables("train")
    data = assemble_dataset(tables)
    validate_dataset_invariants(data, require_target=True)
    X, categorical = prepare_features(data)
    numeric = [column for column in X.columns if column not in categorical]
    y = data[TARGET].astype(str)
    folds = build_walk_forward_folds(tables)
    rows: list[dict[str, object]] = []

    for fold_number, fold in enumerate(folds, start=1):
        LOG.info("Evaluating baselines on %s", fold.name)
        y_train = y.iloc[fold.train_idx]
        y_eval = y.iloc[fold.evaluation_idx]

        if only in (None, "dummy"):
            for strategy in ("most_frequent", "stratified"):
                dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_SEED)
                dummy.fit(np.zeros((len(fold.train_idx), 1)), y_train)
                prediction = dummy.predict(np.zeros((len(fold.evaluation_idx), 1)))
                add_metric_row(rows, f"dummy_{strategy}", fold, y_eval, prediction)

        if only in (None, "logistic"):
            logistic = logistic_pipeline(categorical, numeric)
            logistic.fit(X.iloc[fold.train_idx], y_train)
            add_metric_row(
                rows,
                "multinomial_logistic_regression",
                fold,
                y_eval,
                logistic.predict(X.iloc[fold.evaluation_idx]),
            )

        if only in (None, "tree"):
            tree = tree_pipeline(categorical, numeric)
            tree.fit(X.iloc[fold.train_idx], y_train)
            add_metric_row(
                rows,
                "hist_gradient_boosting",
                fold,
                y_eval,
                tree.predict(X.iloc[fold.evaluation_idx]),
            )

        if only in (None, "simple_catboost"):
            simple_columns = simple_catboost_columns(list(X.columns))
            simple_categorical = [
                column for column in categorical if column in simple_columns
            ]
            simple_model: CatBoostClassifier = build_model(
                iterations, RANDOM_SEED + fold_number - 1, thread_count
            )
            simple_model.fit(
                X.iloc[fold.train_idx][simple_columns],
                y_train,
                cat_features=simple_categorical,
                eval_set=(
                    X.iloc[fold.tuning_idx][simple_columns],
                    y.iloc[fold.tuning_idx],
                ),
                early_stopping_rounds=80,
                use_best_model=True,
            )
            add_metric_row(
                rows,
                "simple_catboost",
                fold,
                y_eval,
                simple_model.predict(
                    X.iloc[fold.evaluation_idx][simple_columns]
                ).reshape(-1),
            )

    metrics = pd.DataFrame(rows)
    output_path = ARTIFACT_DIR / "baseline_metrics.csv"
    if only is not None and output_path.exists():
        existing = pd.read_csv(output_path)
        replacement_names = set(metrics["model"])
        existing = existing.loc[~existing["model"].isin(replacement_names)]
        metrics = pd.concat([existing, metrics], ignore_index=True)
    metrics = metrics.loc[~metrics["model"].str.startswith("engineered_catboost")]
    engineered_path = ARTIFACT_DIR / "walk_forward_metrics.csv"
    if engineered_path.exists():
        engineered = pd.read_csv(engineered_path)
        for variant, name in (
            ("raw_argmax", "engineered_catboost_raw"),
            ("tuning_selected_adjustment", "engineered_catboost_adjusted"),
        ):
            subset = engineered.loc[engineered["decision_variant"].eq(variant)].copy()
            subset["model"] = name
            keep = [column for column in metrics.columns if column in subset.columns]
            metrics = pd.concat([metrics, subset[keep]], ignore_index=True)
    metrics.to_csv(output_path, index=False)

    metric_columns = [
        "weighted_f1",
        "macro_f1",
        "not_churned_f1",
        "vol_churn_f1",
        "invol_churn_f1",
    ]
    summary = metrics.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("weighted_f1_mean", ascending=False)
    summary.to_csv(ARTIFACT_DIR / "baseline_summary.csv", index=False)
    LOG.info("Baseline summary:\n%s", summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=650)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument(
        "--only",
        choices=["dummy", "logistic", "tree", "simple_catboost"],
        help="Rerun only one baseline family and merge it into existing results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args.iterations, args.thread_count, args.only)
