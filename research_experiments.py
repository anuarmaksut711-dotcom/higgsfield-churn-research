"""Focused CatBoost studies on development temporal folds only.

The latest temporal fold is intentionally excluded from model/configuration
selection. It is evaluated only after a study has selected a final approach.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from churn_pipeline import (
    ARTIFACT_DIR,
    CLASSES,
    ID_COL,
    RANDOM_SEED,
    TARGET,
    assemble_dataset,
    build_model,
    build_walk_forward_folds,
    decision_argmax,
    load_tables,
    metric_record,
    optimize_decision_multipliers,
    prepare_features,
    validate_dataset_invariants,
)

LOG = logging.getLogger("research_experiments")
EXPERIMENT_DIR = ARTIFACT_DIR / "experiments"


def fit_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    fold,
    *,
    iterations: int,
    seed: int,
    thread_count: int,
    model_kwargs: dict[str, object] | None = None,
    adjust_decision: bool = False,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, float], int]:
    model = build_model(iterations, seed, thread_count)
    if model_kwargs:
        model.set_params(**model_kwargs)
    model.fit(
        X.iloc[fold.train_idx],
        y.iloc[fold.train_idx],
        cat_features=categorical,
        eval_set=(X.iloc[fold.tuning_idx], y.iloc[fold.tuning_idx]),
        early_stopping_rounds=60,
        use_best_model=True,
    )
    classes = [str(item) for item in model.classes_]
    tuning_probability = model.predict_proba(X.iloc[fold.tuning_idx])
    multipliers, _, _ = optimize_decision_multipliers(
        y.iloc[fold.tuning_idx], tuning_probability, classes
    )
    evaluation_probability = model.predict_proba(X.iloc[fold.evaluation_idx])
    raw_prediction = np.asarray(classes, dtype=object)[
        np.argmax(evaluation_probability, axis=1)
    ]
    prediction = (
        decision_argmax(evaluation_probability, classes, multipliers)
        if adjust_decision
        else raw_prediction
    )
    return (
        metric_record(y.iloc[fold.evaluation_idx], prediction),
        prediction,
        evaluation_probability,
        multipliers,
        int(model.get_best_iteration() + 1),
    )


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    basic = [column for column in columns if column in {"subscription_plan", "country_code"}]
    quiz = [column for column in columns if column.startswith("quiz_")]
    signup = [column for column in columns if column.startswith("signup_")]
    purchase = [column for column in columns if column.startswith("purchase_")]
    temporal = [
        column
        for column in columns
        if any(token in column for token in ("_d1", "_d3", "_d7", "_d14"))
        or column.endswith(("_first_day", "_last_day", "_span_days", "_active_days"))
    ]
    payment = [
        column
        for column in columns
        if any(
            token in column.lower()
            for token in (
                "failure",
                "cvc",
                "3d_secure",
                "prepaid",
                "card_funding",
                "card_declined",
                "insufficient_funds",
            )
        )
    ]
    attempts = [
        column
        for column in columns
        if column.startswith("attempt_") or column.startswith("is_")
    ]
    cross = [
        column
        for column in columns
        if column
        in {
            "attempts_per_purchase",
            "failures_per_purchase",
            "spend_per_attempt",
            "purchase_attempt_gap",
            "activity_days_total",
        }
    ]
    return {
        "basic": basic,
        "quiz": quiz,
        "signup": signup,
        "purchase": purchase,
        "temporal": temporal,
        "payment": payment,
        "attempts": attempts,
        "cross": cross,
    }


def ordered_union(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(column for group in groups for column in group))


def summarize(metrics: pd.DataFrame, group: str) -> pd.DataFrame:
    metric_columns = [
        "weighted_f1",
        "macro_f1",
        "not_churned_f1",
        "vol_churn_f1",
        "invol_churn_f1",
    ]
    summary = metrics.groupby(group)[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("weighted_f1_mean", ascending=False)


def common_row(name: str, fold, metrics: dict[str, float], **extra) -> dict[str, object]:
    return {
        "experiment": name,
        "fold": fold.name,
        "train_users": len(fold.train_idx),
        "tuning_users": len(fold.tuning_idx),
        "evaluation_users": len(fold.evaluation_idx),
        "evaluation_start": fold.evaluation_start,
        "evaluation_end": fold.evaluation_end,
        **extra,
        **metrics,
    }


def save_study(name: str, rows: list[dict[str, object]], config: dict[str, object]) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ARTIFACT_DIR / f"{name}_metrics.csv", index=False)
    summary = summarize(metrics, "experiment")
    summary.to_csv(ARTIFACT_DIR / f"{name}_summary.csv", index=False)
    payload = {
        "experiment_name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selection_folds": ["fold_1", "fold_2"],
        "final_fold_used_for_selection": False,
        **config,
        "results": rows,
    }
    (EXPERIMENT_DIR / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("%s summary:\n%s", name, summary.to_string(index=False))


def run_ablation(iterations: int, thread_count: int) -> None:
    tables = load_tables("train")
    data = assemble_dataset(tables, observation_days=14)
    validate_dataset_invariants(data, require_target=True)
    X, categorical = prepare_features(data)
    y = data[TARGET].astype(str)
    groups = feature_groups(list(X.columns))
    base = groups["basic"]
    configurations = {
        "base": base,
        "base_plus_transactions": ordered_union(base, groups["attempts"]),
        "base_plus_quiz": ordered_union(base, groups["quiz"]),
        "base_plus_purchases": ordered_union(base, groups["purchase"]),
        "base_plus_temporal": ordered_union(base, groups["temporal"]),
        "base_plus_cross_ratios": ordered_union(base, groups["cross"]),
        "full_without_quiz": [c for c in X.columns if c not in groups["quiz"]],
        "full_without_payment": [c for c in X.columns if c not in groups["payment"]],
        "full_without_signup_time": [c for c in X.columns if c not in groups["signup"]],
        "full_raw": list(X.columns),
        "full_with_decision_adjustment": list(X.columns),
    }
    rows: list[dict[str, object]] = []
    for name, columns in configurations.items():
        cat = [column for column in categorical if column in columns]
        adjusted = name == "full_with_decision_adjustment"
        for fold in build_walk_forward_folds(tables)[:2]:
            LOG.info("Ablation %s on %s (%d features)", name, fold.name, len(columns))
            metrics, _, _, _, best_iteration = fit_evaluate(
                X[columns],
                y,
                cat,
                fold,
                iterations=iterations,
                seed=RANDOM_SEED,
                thread_count=thread_count,
                adjust_decision=adjusted,
            )
            rows.append(
                common_row(
                    name,
                    fold,
                    metrics,
                    feature_count=len(columns),
                    best_iteration=best_iteration,
                    decision_adjusted=adjusted,
                )
            )
    save_study(
        "ablation",
        rows,
        {"iterations_cap": iterations, "observation_days": 14},
    )


def run_observation_windows(iterations: int, thread_count: int) -> None:
    tables = load_tables("train")
    folds = build_walk_forward_folds(tables)[:2]
    rows: list[dict[str, object]] = []
    for window in (1, 3, 7, 14):
        data = assemble_dataset(tables, observation_days=window)
        X, categorical = prepare_features(data)
        y = data[TARGET].astype(str)
        for fold in folds:
            LOG.info("Observation window d%d on %s", window, fold.name)
            metrics, _, _, _, best_iteration = fit_evaluate(
                X,
                y,
                categorical,
                fold,
                iterations=iterations,
                seed=RANDOM_SEED,
                thread_count=thread_count,
                adjust_decision=True,
            )
            rows.append(
                common_row(
                    f"{window}_days",
                    fold,
                    metrics,
                    observation_days=window,
                    feature_count=X.shape[1],
                    best_iteration=best_iteration,
                    decision_adjusted=True,
                )
            )
    save_study(
        "observation_window",
        rows,
        {"iterations_cap": iterations, "windows": [1, 3, 7, 14]},
    )


def run_seed_ensemble(iterations: int, thread_count: int) -> None:
    tables = load_tables("train")
    data = assemble_dataset(tables, observation_days=14)
    X, categorical = prepare_features(data)
    y = data[TARGET].astype(str)
    rows: list[dict[str, object]] = []
    seeds = [RANDOM_SEED, RANDOM_SEED + 97, RANDOM_SEED + 194]
    for fold in build_walk_forward_folds(tables)[:2]:
        tuning_probabilities: list[np.ndarray] = []
        evaluation_probabilities: list[np.ndarray] = []
        classes: list[str] | None = None
        for seed in seeds:
            LOG.info("Seed %d on %s", seed, fold.name)
            model = build_model(iterations, seed, thread_count)
            model.fit(
                X.iloc[fold.train_idx],
                y.iloc[fold.train_idx],
                cat_features=categorical,
                eval_set=(X.iloc[fold.tuning_idx], y.iloc[fold.tuning_idx]),
                early_stopping_rounds=60,
                use_best_model=True,
            )
            classes = [str(item) for item in model.classes_]
            tuning_probabilities.append(model.predict_proba(X.iloc[fold.tuning_idx]))
            evaluation_probabilities.append(
                model.predict_proba(X.iloc[fold.evaluation_idx])
            )
        assert classes is not None
        for count, name in ((1, "single_seed"), (3, "three_seed_ensemble")):
            tuning_probability = np.mean(tuning_probabilities[:count], axis=0)
            evaluation_probability = np.mean(evaluation_probabilities[:count], axis=0)
            multipliers, _, _ = optimize_decision_multipliers(
                y.iloc[fold.tuning_idx], tuning_probability, classes
            )
            prediction = decision_argmax(
                evaluation_probability, classes, multipliers
            )
            rows.append(
                common_row(
                    name,
                    fold,
                    metric_record(y.iloc[fold.evaluation_idx], prediction),
                    seed_count=count,
                    seeds=",".join(str(seed) for seed in seeds[:count]),
                    decision_adjusted=True,
                )
            )
    save_study(
        "seed_ensemble",
        rows,
        {"iterations_cap": iterations, "seeds": seeds, "observation_days": 14},
    )


def run_hyperparameter_screen(thread_count: int) -> None:
    tables = load_tables("train")
    data = assemble_dataset(tables, observation_days=14)
    X, categorical = prepare_features(data)
    y = data[TARGET].astype(str)
    configurations = {
        "current": {
            "iterations": 300,
            "depth": 7,
            "learning_rate": 0.06,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
        },
        "shallower_depth6": {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.06,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
        },
        "deeper_depth8": {
            "iterations": 300,
            "depth": 8,
            "learning_rate": 0.06,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
        },
        "slower_regularized": {
            "iterations": 450,
            "depth": 7,
            "learning_rate": 0.04,
            "l2_leaf_reg": 10.0,
            "random_strength": 1.0,
        },
    }
    rows: list[dict[str, object]] = []
    for name, config in configurations.items():
        for fold in build_walk_forward_folds(tables)[:2]:
            LOG.info("Hyperparameters %s on %s", name, fold.name)
            iterations = int(config["iterations"])
            kwargs = {key: value for key, value in config.items() if key != "iterations"}
            metrics, _, _, multipliers, best_iteration = fit_evaluate(
                X,
                y,
                categorical,
                fold,
                iterations=iterations,
                seed=RANDOM_SEED,
                thread_count=thread_count,
                model_kwargs=kwargs,
                adjust_decision=True,
            )
            rows.append(
                common_row(
                    name,
                    fold,
                    metrics,
                    best_iteration=best_iteration,
                    decision_multipliers=json.dumps(multipliers, sort_keys=True),
                    **config,
                )
            )
    save_study(
        "hyperparameter_screen",
        rows,
        {"configurations": configurations, "observation_days": 14},
    )


def run_final_window_evaluation(iterations: int, thread_count: int) -> None:
    """Evaluate the preselected 3-day challenger against the 14-day reference once."""
    tables = load_tables("train")
    final_fold = build_walk_forward_folds(tables)[-1]
    rows: list[dict[str, object]] = []
    for window in (3, 14):
        data = assemble_dataset(tables, observation_days=window)
        X, categorical = prepare_features(data)
        y = data[TARGET].astype(str)
        LOG.info("Final locked evaluation: d%d", window)
        metrics, _, _, multipliers, best_iteration = fit_evaluate(
            X,
            y,
            categorical,
            final_fold,
            iterations=iterations,
            seed=RANDOM_SEED,
            thread_count=thread_count,
            adjust_decision=True,
        )
        rows.append(
            common_row(
                f"{window}_days",
                final_fold,
                metrics,
                observation_days=window,
                feature_count=X.shape[1],
                best_iteration=best_iteration,
                decision_multipliers=json.dumps(multipliers, sort_keys=True),
                locked_final_evaluation=True,
            )
        )
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        ARTIFACT_DIR / "observation_window_final_evaluation.csv", index=False
    )
    (EXPERIMENT_DIR / "observation_window_final_evaluation.json").write_text(
        json.dumps(
            {
                "experiment_name": "observation_window_final_evaluation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "selection_basis": "3-day challenger selected on folds 1-2",
                "final_fold_used_once": True,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info("Final window evaluation:\n%s", pd.DataFrame(rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        choices=[
            "ablation",
            "observation-window",
            "seed-ensemble",
            "hyperparameters",
            "final-window-evaluation",
        ],
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--thread-count", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.study == "ablation":
        run_ablation(args.iterations, args.thread_count)
    elif args.study == "observation-window":
        run_observation_windows(args.iterations, args.thread_count)
    elif args.study == "seed-ensemble":
        run_seed_ensemble(args.iterations, args.thread_count)
    elif args.study == "hyperparameters":
        run_hyperparameter_screen(args.thread_count)
    else:
        run_final_window_evaluation(args.iterations, args.thread_count)
