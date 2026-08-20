"""Reproducible Higgsfield churn pipeline.

Predicts ``not_churned``, ``vol_churn`` or ``invol_churn`` from signals
observed during the first 14 days after subscription start. Validation is
out-of-time and the optimized metric is weighted F1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_SCRIPT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_SCRIPT_ROOT / ".venv" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.churn.config import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "churn_status"
ID_COL = "user_id"
CLASSES = ["not_churned", "vol_churn", "invol_churn"]
OBSERVATION_DAYS = float(DEFAULT_CONFIG.observation_days)
RANDOM_SEED = DEFAULT_CONFIG.random_seed
TEMPORAL_BLOCK_USERS = DEFAULT_CONFIG.temporal_block_users

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("churn_pipeline")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a source CSV and drop serialization-only index columns."""
    if not path.exists():
        raise FileNotFoundError(f"Required dataset is missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")].copy()


def load_tables(split: str) -> dict[str, pd.DataFrame]:
    base = TRAIN_DIR if split == "train" else TEST_DIR
    prefix = "train_users" if split == "train" else "test_users"
    tables = {
        "users": read_csv(base / f"{prefix}.csv"),
        "properties": read_csv(base / f"{prefix}_properties.csv"),
        "quizzes": read_csv(base / f"{prefix}_quizzes.csv"),
        "purchases": read_csv(base / f"{prefix}_purchases.csv"),
        "attempts": read_csv(base / f"{prefix}_transaction_attempts_v1.csv"),
    }
    LOG.info(
        "%s tables: %s",
        split,
        ", ".join(f"{name}={len(df):,}" for name, df in tables.items()),
    )
    return tables


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def safe_mode(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return "unknown"
    modes = values.mode()
    return str(modes.iloc[0] if not modes.empty else values.iloc[0])


def true_rate(series: pd.Series) -> float:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "1": 1.0, "yes": 1.0, "false": 0.0, "0": 0.0, "no": 0.0})
    )
    return float(mapped.mean()) if mapped.notna().any() else 0.0


def bool_as_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "1": 1.0, "yes": 1.0, "false": 0.0, "0": 0.0, "no": 0.0})
        .fillna(0.0)
    )


def add_event_clock(
    events: pd.DataFrame,
    properties: pd.DataFrame,
    time_col: str,
    observation_days: float = OBSERVATION_DAYS,
) -> pd.DataFrame:
    """Attach event time relative to each user's subscription start."""
    base = properties[[ID_COL, "subscription_start_date"]].drop_duplicates(ID_COL)
    frame = events.merge(base, on=ID_COL, how="inner", validate="many_to_one")
    frame[time_col] = parse_time(frame[time_col])
    frame["subscription_start_date"] = parse_time(frame["subscription_start_date"])
    frame["event_day"] = (
        frame[time_col] - frame["subscription_start_date"]
    ).dt.total_seconds() / 86_400.0
    frame = frame.loc[
        frame["event_day"].ge(0) & frame["event_day"].lt(observation_days)
    ].copy()
    frame["event_date"] = frame[time_col].dt.floor("D")
    frame["event_hour"] = frame[time_col].dt.hour
    frame["event_weekday"] = frame[time_col].dt.dayofweek
    return frame


def aggregate_attempts(
    attempts: pd.DataFrame,
    properties: pd.DataFrame,
    observation_days: float = OBSERVATION_DAYS,
) -> pd.DataFrame:
    frame = add_event_clock(
        attempts, properties, "transaction_time", observation_days=observation_days
    )
    frame["amount_in_usd"] = pd.to_numeric(frame["amount_in_usd"], errors="coerce")
    frame["is_failure"] = frame["failure_code"].notna().astype(int)
    frame["is_success"] = 1 - frame["is_failure"]
    frame["is_night"] = frame["event_hour"].between(0, 5).astype(int)
    frame["is_weekend"] = frame["event_weekday"].ge(5).astype(int)
    grouped = frame.groupby(ID_COL, observed=True)
    result = grouped.agg(
        attempt_count=("transaction_id", "size"),
        attempt_unique_transactions=("transaction_id", "nunique"),
        attempt_active_days=("event_date", "nunique"),
        attempt_success_count=("is_success", "sum"),
        attempt_failure_count=("is_failure", "sum"),
        attempt_failure_rate=("is_failure", "mean"),
        attempt_amount_sum=("amount_in_usd", "sum"),
        attempt_amount_mean=("amount_in_usd", "mean"),
        attempt_amount_std=("amount_in_usd", "std"),
        attempt_amount_max=("amount_in_usd", "max"),
        attempt_first_day=("event_day", "min"),
        attempt_last_day=("event_day", "max"),
        attempt_night_rate=("is_night", "mean"),
        attempt_weekend_rate=("is_weekend", "mean"),
        attempt_distinct_failures=("failure_code", "nunique"),
        attempt_distinct_cards=("card_brand", "nunique"),
        attempt_distinct_banks=("bank_name", "nunique"),
        attempt_distinct_card_countries=("card_country", "nunique"),
    ).reset_index()
    for days in (day for day in (1, 3, 7, 14) if day <= observation_days):
        subset = frame.loc[frame["event_day"].lt(days)]
        counts = subset.groupby(ID_COL).size().rename(f"attempt_count_d{days}")
        failures = subset.groupby(ID_COL)["is_failure"].sum().rename(
            f"attempt_failures_d{days}"
        )
        result = result.merge(counts, on=ID_COL, how="left").merge(
            failures, on=ID_COL, how="left"
        )
    failure_names = [
        "card_declined",
        "insufficient_funds",
        "incorrect_cvc",
        "expired_card",
        "authentication_required",
        "do_not_honor",
        "processing_error",
        "generic_decline",
    ]
    failure_text = frame["failure_code"].fillna("success").astype(str).str.lower()
    for code in failure_names:
        values = (
            failure_text.eq(code)
            .groupby(frame[ID_COL])
            .sum()
            .rename(f"failure_{code}_count")
        )
        result = result.merge(values, on=ID_COL, how="left")
    for col in (
        "is_3d_secure",
        "is_3d_secure_authenticated",
        "is_prepaid",
        "is_virtual",
        "is_business",
    ):
        if col in frame.columns:
            values = bool_as_float(frame[col]).groupby(frame[ID_COL]).mean().rename(
                f"{col}_rate"
            )
            result = result.merge(values, on=ID_COL, how="left")
    category_cols = [
        "billing_address_country",
        "card_3d_secure_support",
        "card_brand",
        "card_country",
        "card_funding",
        "cvc_check",
        "digital_wallet",
        "payment_method_type",
        "bank_name",
        "bank_country",
        "failure_code",
    ]
    for col in category_cols:
        if col in frame.columns:
            values = (
                frame[[ID_COL, "transaction_time", col]]
                .dropna(subset=[col])
                .sort_values("transaction_time")
                .drop_duplicates(ID_COL, keep="last")
                .set_index(ID_COL)[col]
                .astype(str)
                .rename(f"attempt_last_{col}")
            )
            result = result.merge(values, on=ID_COL, how="left")
    result["attempt_span_days"] = result["attempt_last_day"] - result["attempt_first_day"]
    result["attempt_success_rate"] = (
        result["attempt_success_count"] / result["attempt_count"].clip(lower=1)
    )
    return result


def aggregate_purchases(
    purchases: pd.DataFrame,
    properties: pd.DataFrame,
    observation_days: float = OBSERVATION_DAYS,
) -> pd.DataFrame:
    frame = add_event_clock(
        purchases, properties, "purchase_time", observation_days=observation_days
    )
    frame["purchase_amount_dollars"] = pd.to_numeric(
        frame["purchase_amount_dollars"], errors="coerce"
    )
    grouped = frame.groupby(ID_COL, observed=True)
    result = grouped.agg(
        purchase_count=("transaction_id", "size"),
        purchase_unique_transactions=("transaction_id", "nunique"),
        purchase_active_days=("event_date", "nunique"),
        purchase_amount_sum=("purchase_amount_dollars", "sum"),
        purchase_amount_mean=("purchase_amount_dollars", "mean"),
        purchase_amount_std=("purchase_amount_dollars", "std"),
        purchase_amount_max=("purchase_amount_dollars", "max"),
        purchase_first_day=("event_day", "min"),
        purchase_last_day=("event_day", "max"),
        purchase_type_nunique=("purchase_type", "nunique"),
    ).reset_index()
    last_type = (
        frame[[ID_COL, "purchase_time", "purchase_type"]]
        .dropna(subset=["purchase_type"])
        .sort_values("purchase_time")
        .drop_duplicates(ID_COL, keep="last")
        .set_index(ID_COL)["purchase_type"]
        .astype(str)
        .rename("purchase_last_type")
    )
    result = result.merge(last_type, on=ID_COL, how="left")
    for days in (day for day in (1, 3, 7, 14) if day <= observation_days):
        subset = frame.loc[frame["event_day"].lt(days)]
        counts = subset.groupby(ID_COL).size().rename(f"purchase_count_d{days}")
        spend = subset.groupby(ID_COL)["purchase_amount_dollars"].sum().rename(
            f"purchase_spend_d{days}"
        )
        result = result.merge(counts, on=ID_COL, how="left").merge(
            spend, on=ID_COL, how="left"
        )
    result["purchase_span_days"] = result["purchase_last_day"] - result["purchase_first_day"]
    return result


def aggregate_quizzes(quizzes: pd.DataFrame) -> pd.DataFrame:
    frame = quizzes.copy()
    answer_cols = [col for col in frame.columns if col != ID_COL]
    frame["quiz_missing_answers"] = frame[answer_cols].isna().sum(axis=1)
    frame["quiz_answered"] = len(answer_cols) - frame["quiz_missing_answers"]
    grouped = frame.groupby(ID_COL, observed=True)
    result = grouped.agg(
        quiz_rows=(ID_COL, "size"),
        quiz_missing_answers=("quiz_missing_answers", "mean"),
        quiz_answered=("quiz_answered", "mean"),
    ).reset_index()
    answers = grouped[answer_cols].first().reset_index()
    answers = answers.rename(columns={col: f"quiz_{col}" for col in answer_cols})
    result = result.merge(answers, on=ID_COL, how="left")
    if "quiz_frustration" in result.columns:
        frustration = result["quiz_frustration"].astype(str).str.lower()
        result["quiz_cost_concern"] = frustration.str.contains(
            "cost|price|expensive", regex=True
        ).astype(int)
    return result


def assemble_dataset(
    tables: dict[str, pd.DataFrame],
    observation_days: float = OBSERVATION_DAYS,
) -> pd.DataFrame:
    properties = tables["properties"].drop_duplicates(ID_COL).copy()
    signup = parse_time(properties["subscription_start_date"])
    properties["signup_weekday"] = signup.dt.dayofweek
    properties["signup_hour"] = signup.dt.hour
    properties["signup_is_weekend"] = signup.dt.dayofweek.ge(5).astype(int)
    properties = properties.drop(columns=["subscription_start_date"])
    attempts = aggregate_attempts(
        tables["attempts"], tables["properties"], observation_days=observation_days
    )
    purchases = aggregate_purchases(
        tables["purchases"], tables["properties"], observation_days=observation_days
    )
    quizzes = aggregate_quizzes(tables["quizzes"])
    dataset = (
        tables["users"]
        .merge(properties, on=ID_COL, how="left", validate="one_to_one")
        .merge(quizzes, on=ID_COL, how="left", validate="one_to_one")
        .merge(attempts, on=ID_COL, how="left", validate="one_to_one")
        .merge(purchases, on=ID_COL, how="left", validate="one_to_one")
    )
    dataset["attempts_per_purchase"] = dataset.get("attempt_count", 0) / (
        dataset.get("purchase_count", pd.Series(0, index=dataset.index)).fillna(0) + 1
    )
    dataset["failures_per_purchase"] = dataset.get("attempt_failure_count", 0) / (
        dataset.get("purchase_count", pd.Series(0, index=dataset.index)).fillna(0) + 1
    )
    dataset["spend_per_attempt"] = dataset.get("purchase_amount_sum", 0) / (
        dataset.get("attempt_count", pd.Series(0, index=dataset.index)).fillna(0) + 1
    )
    dataset["purchase_attempt_gap"] = dataset.get(
        "purchase_last_day", pd.Series(0, index=dataset.index)
    ).fillna(0) - dataset.get("attempt_last_day", pd.Series(0, index=dataset.index)).fillna(0)
    dataset["activity_days_total"] = dataset.get(
        "attempt_active_days", pd.Series(0, index=dataset.index)
    ).fillna(0) + dataset.get("purchase_active_days", pd.Series(0, index=dataset.index)).fillna(0)
    LOG.info("assembled dataset: %s", dataset.shape)
    return dataset


def prepare_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = dataset.drop(
        columns=[col for col in (ID_COL, TARGET) if col in dataset.columns]
    ).copy()
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [col for col in features.columns if col not in categorical]
    features[categorical] = features[categorical].fillna("unknown").astype(str)
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features, categorical


def align_feature_schemas(
    train_features: pd.DataFrame,
    other_features: pd.DataFrame,
    categorical: Iterable[str],
) -> pd.DataFrame:
    """Align inference/evaluation features to the fitted training schema."""
    aligned = other_features.reindex(columns=train_features.columns, fill_value=0).copy()
    for col in categorical:
        aligned[col] = aligned[col].fillna("unknown").astype(str)
    if list(aligned.columns) != list(train_features.columns):
        raise AssertionError("Feature schemas are not identical after alignment")
    return aligned


def validate_dataset_invariants(dataset: pd.DataFrame, *, require_target: bool) -> None:
    """Fail fast on invariants required for leakage-safe model training."""
    if ID_COL not in dataset.columns:
        raise AssertionError(f"Missing identifier column: {ID_COL}")
    if dataset[ID_COL].isna().any() or not dataset[ID_COL].is_unique:
        raise AssertionError("Each user_id must map to exactly one feature row")
    if require_target and TARGET not in dataset.columns:
        raise AssertionError(f"Training dataset is missing target: {TARGET}")
    if not require_target and TARGET in dataset.columns:
        raise AssertionError("Target leaked into an inference dataset")


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_idx: np.ndarray
    tuning_idx: np.ndarray
    evaluation_idx: np.ndarray
    train_start: str
    train_end: str
    tuning_start: str
    tuning_end: str
    evaluation_start: str
    evaluation_end: str


def _time_range(signup: pd.Series, indices: np.ndarray) -> tuple[str, str]:
    values = signup.iloc[indices]
    return str(values.min()), str(values.max())


def build_walk_forward_folds(
    tables: dict[str, pd.DataFrame],
    block_users: int = TEMPORAL_BLOCK_USERS,
    n_folds: int = 3,
) -> list[TemporalFold]:
    """Create expanding train/tuning/evaluation folds ordered by signup time.

    With 90k users and 15k-user blocks the folds are:
    30k/15k/15k, 45k/15k/15k and 60k/15k/15k.  Thus every model-selection
    period strictly precedes its evaluation period, and the latest 15k users
    form the final untouched temporal evaluation cohort.
    """
    users = tables["users"][[ID_COL]].merge(
        tables["properties"][[ID_COL, "subscription_start_date"]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    signup = parse_time(users["subscription_start_date"])
    if signup.isna().any():
        raise ValueError("Missing or invalid subscription_start_date")
    order = np.argsort(signup.to_numpy(), kind="stable")
    required = (n_folds + 3) * block_users
    if len(order) < required:
        raise ValueError(
            f"Need at least {required:,} users for {n_folds} folds; got {len(order):,}"
        )
    initial_train = len(order) - (n_folds + 1) * block_users
    if initial_train < block_users:
        raise ValueError("Initial temporal training period is too small")

    folds: list[TemporalFold] = []
    for fold_number in range(n_folds):
        train_end = initial_train + fold_number * block_users
        tuning_end = train_end + block_users
        evaluation_end = tuning_end + block_users
        train_idx = order[:train_end]
        tuning_idx = order[train_end:tuning_end]
        evaluation_idx = order[tuning_end:evaluation_end]
        train_range = _time_range(signup, train_idx)
        tuning_range = _time_range(signup, tuning_idx)
        evaluation_range = _time_range(signup, evaluation_idx)
        if not (
            signup.iloc[train_idx].max() < signup.iloc[tuning_idx].min()
            and signup.iloc[tuning_idx].max() < signup.iloc[evaluation_idx].min()
        ):
            raise AssertionError("Temporal fold ordering is invalid")
        folds.append(
            TemporalFold(
                name=f"fold_{fold_number + 1}",
                train_idx=train_idx,
                tuning_idx=tuning_idx,
                evaluation_idx=evaluation_idx,
                train_start=train_range[0],
                train_end=train_range[1],
                tuning_start=tuning_range[0],
                tuning_end=tuning_range[1],
                evaluation_start=evaluation_range[0],
                evaluation_end=evaluation_range[1],
            )
        )
    return folds


def temporal_split(
    tables: dict[str, pd.DataFrame], validation_days: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """Legacy two-way split retained for backward compatibility and tests."""
    users = tables["users"][[ID_COL]].merge(
        tables["properties"][[ID_COL, "subscription_start_date"]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    signup = parse_time(users["subscription_start_date"])
    cutoff = signup.max() - pd.Timedelta(days=validation_days)
    train_idx = np.flatnonzero(signup.lt(cutoff).to_numpy())
    valid_idx = np.flatnonzero(signup.ge(cutoff).to_numpy())
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise ValueError("Temporal split is empty; check subscription dates")
    return train_idx, valid_idx, str(cutoff)


def optimize_decision_multipliers(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: list[str],
) -> tuple[dict[str, float], np.ndarray, float]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    # A broad range is necessary: voluntary churn is systematically less
    # separable and the optimum is often well above 1.5 on temporal folds.
    candidates = np.arange(0.50, 3.01, 0.10)
    best_score = -1.0
    best_weights = {name: 1.0 for name in classes}
    best_pred: np.ndarray | None = None
    for vol_weight in candidates:
        for invol_weight in candidates:
            weights = np.ones(len(classes), dtype=float)
            weights[class_to_idx["vol_churn"]] = vol_weight
            weights[class_to_idx["invol_churn"]] = invol_weight
            pred = np.asarray(classes, dtype=object)[np.argmax(probabilities * weights, axis=1)]
            score = f1_score(y_true, pred, average="weighted")
            if score > best_score:
                best_score = score
                best_pred = pred
                best_weights = {
                    "not_churned": 1.0,
                    "vol_churn": float(vol_weight),
                    "invol_churn": float(invol_weight),
                }
    assert best_pred is not None
    return best_weights, best_pred, float(best_score)


def decision_argmax(
    probabilities: np.ndarray,
    classes: Iterable[str],
    multipliers: dict[str, float],
) -> np.ndarray:
    class_list = list(classes)
    weights = np.asarray([multipliers.get(name, 1.0) for name in class_list])
    return np.asarray(class_list, dtype=object)[np.argmax(probabilities * weights, axis=1)]


def weighted_argmax(
    probabilities: np.ndarray,
    classes: Iterable[str],
    multipliers: dict[str, float],
) -> np.ndarray:
    """Backward-compatible alias; use :func:`decision_argmax` in new code."""
    return decision_argmax(probabilities, classes, multipliers)


def metric_record(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    report = classification_report(
        y_true, pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    record: dict[str, float] = {
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
    }
    for class_name in CLASSES:
        for metric in ("precision", "recall", "f1-score", "support"):
            key = "f1" if metric == "f1-score" else metric
            record[f"{class_name}_{key}"] = float(report[class_name][metric])
    return record


def build_model(iterations: int, seed: int, thread_count: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=DEFAULT_CONFIG.learning_rate,
        depth=DEFAULT_CONFIG.depth,
        l2_leaf_reg=DEFAULT_CONFIG.l2_leaf_reg,
        random_strength=DEFAULT_CONFIG.random_strength,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Weighted",
        random_seed=seed,
        thread_count=thread_count,
        allow_writing_files=False,
        verbose=100,
    )


def run_walk_forward_evaluation(
    iterations: int,
    thread_count: int,
    block_users: int = TEMPORAL_BLOCK_USERS,
    n_folds: int = 3,
) -> None:
    """Evaluate raw and tuning-selected decisions on expanding temporal folds."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables("train")
    data = assemble_dataset(tables)
    validate_dataset_invariants(data, require_target=True)
    X, categorical = prepare_features(data)
    if TARGET in X.columns or ID_COL in X.columns:
        raise AssertionError("Target or identifier leaked into model features")
    y = data[TARGET].astype(str)
    folds = build_walk_forward_folds(tables, block_users=block_users, n_folds=n_folds)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    confusion_payload: dict[str, dict[str, list[list[int]]]] = {}
    multiplier_rows: list[dict[str, object]] = []
    for fold_number, fold in enumerate(folds, start=1):
        LOG.info(
            "%s | train=%s tuning=%s evaluation=%s",
            fold.name,
            f"{len(fold.train_idx):,}",
            f"{len(fold.tuning_idx):,}",
            f"{len(fold.evaluation_idx):,}",
        )
        model = build_model(iterations, RANDOM_SEED + fold_number - 1, thread_count)
        model.fit(
            X.iloc[fold.train_idx],
            y.iloc[fold.train_idx],
            cat_features=categorical,
            eval_set=(X.iloc[fold.tuning_idx], y.iloc[fold.tuning_idx]),
            early_stopping_rounds=80,
            use_best_model=True,
        )
        classes = [str(item) for item in model.classes_]
        tuning_probabilities = model.predict_proba(X.iloc[fold.tuning_idx])
        multipliers, _, tuning_adjusted_f1 = optimize_decision_multipliers(
            y.iloc[fold.tuning_idx], tuning_probabilities, classes
        )
        tuning_raw_pred = np.asarray(classes, dtype=object)[
            np.argmax(tuning_probabilities, axis=1)
        ]
        tuning_raw_f1 = f1_score(
            y.iloc[fold.tuning_idx], tuning_raw_pred, average="weighted"
        )
        evaluation_probabilities = model.predict_proba(X.iloc[fold.evaluation_idx])
        raw_pred = np.asarray(classes, dtype=object)[
            np.argmax(evaluation_probabilities, axis=1)
        ]
        adjusted_pred = decision_argmax(
            evaluation_probabilities, classes, multipliers
        )
        base_metadata: dict[str, object] = {
            "fold": fold.name,
            "train_users": len(fold.train_idx),
            "tuning_users": len(fold.tuning_idx),
            "evaluation_users": len(fold.evaluation_idx),
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "tuning_start": fold.tuning_start,
            "tuning_end": fold.tuning_end,
            "evaluation_start": fold.evaluation_start,
            "evaluation_end": fold.evaluation_end,
            "best_iteration": int(model.get_best_iteration() + 1),
            "tuning_raw_weighted_f1": float(tuning_raw_f1),
            "tuning_adjusted_weighted_f1": float(tuning_adjusted_f1),
        }
        for decision_variant, pred in (
            ("raw_argmax", raw_pred),
            ("tuning_selected_adjustment", adjusted_pred),
        ):
            metric_rows.append(
                {
                    **base_metadata,
                    "decision_variant": decision_variant,
                    **metric_record(y.iloc[fold.evaluation_idx], pred),
                }
            )
        for class_name, multiplier in multipliers.items():
            multiplier_rows.append(
                {
                    "fold": fold.name,
                    "class": class_name,
                    "decision_multiplier": multiplier,
                }
            )
        confusion_payload[fold.name] = {
            "raw_argmax": confusion_matrix(
                y.iloc[fold.evaluation_idx], raw_pred, labels=CLASSES
            ).tolist(),
            "tuning_selected_adjustment": confusion_matrix(
                y.iloc[fold.evaluation_idx], adjusted_pred, labels=CLASSES
            ).tolist(),
        }
        frame = pd.DataFrame(
            {
                "fold": fold.name,
                ID_COL: data.iloc[fold.evaluation_idx][ID_COL].values,
                "true": y.iloc[fold.evaluation_idx].values,
                "raw_pred": raw_pred,
                "adjusted_pred": adjusted_pred,
            }
        )
        for class_index, class_name in enumerate(classes):
            frame[f"model_probability_{class_name}"] = evaluation_probabilities[
                :, class_index
            ]
            frame[f"decision_score_{class_name}"] = (
                evaluation_probabilities[:, class_index] * multipliers[class_name]
            )
        prediction_frames.append(frame)
        LOG.info(
            "%s evaluation weighted F1 raw=%.6f adjusted=%.6f",
            fold.name,
            metric_rows[-2]["weighted_f1"],
            metric_rows[-1]["weighted_f1"],
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ARTIFACT_DIR / "walk_forward_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        ARTIFACT_DIR / "walk_forward_predictions.csv", index=False
    )
    pd.DataFrame(multiplier_rows).to_csv(
        ARTIFACT_DIR / "walk_forward_decision_multipliers.csv", index=False
    )
    (ARTIFACT_DIR / "walk_forward_confusion_matrices.json").write_text(
        json.dumps(
            {"labels": CLASSES, "folds": confusion_payload},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_metrics = [
        "weighted_f1",
        "macro_f1",
        *[f"{class_name}_{metric}" for class_name in CLASSES for metric in ("precision", "recall", "f1")],
    ]
    summary = (
        metrics.groupby("decision_variant")[summary_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "decision_variant"
        if column[0] == "decision_variant"
        else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    summary.to_csv(ARTIFACT_DIR / "walk_forward_summary.csv", index=False)
    LOG.info("Walk-forward summary:\n%s", summary.to_string(index=False))


def save_validation_artifacts(
    y_true: pd.Series,
    pred: np.ndarray,
    classes: list[str],
    fold: TemporalFold,
    tuning_raw_score: float,
    tuning_adjusted_score: float,
    final_raw_metrics: dict[str, float],
    final_adjusted_metrics: dict[str, float],
    multipliers: dict[str, float],
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true, pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(ARTIFACT_DIR / "classification_report.csv")
    matrix = confusion_matrix(y_true, pred, labels=CLASSES)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
        ARTIFACT_DIR / "confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Out-of-time validation confusion matrix")
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    metrics = {
        "metric": "weighted_f1",
        "methodology": "train -> temporal tuning -> untouched temporal evaluation",
        "final_fold": {
            "name": fold.name,
            "train_range": [fold.train_start, fold.train_end],
            "tuning_range": [fold.tuning_start, fold.tuning_end],
            "evaluation_range": [fold.evaluation_start, fold.evaluation_end],
            "train_rows": int(len(fold.train_idx)),
            "tuning_rows": int(len(fold.tuning_idx)),
            "evaluation_rows": int(len(fold.evaluation_idx)),
        },
        "tuning_raw_weighted_f1": tuning_raw_score,
        "tuning_adjusted_weighted_f1": tuning_adjusted_score,
        "final_raw_metrics": final_raw_metrics,
        "final_adjusted_metrics": final_adjusted_metrics,
        "decision_multipliers": multipliers,
        "evaluation_distribution": y_true.value_counts().to_dict(),
        "adjusted_prediction_distribution": pd.Series(pred).value_counts().to_dict(),
        "probability_calibrated": False,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {"class": list(multipliers), "multiplier": list(multipliers.values())}
    ).to_csv(ARTIFACT_DIR / "decision_multipliers.csv", index=False)


def train_pipeline(
    iterations: int,
    thread_count: int,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_tables = load_tables("train")
    test_tables = load_tables("test")
    train_data = assemble_dataset(train_tables)
    test_data = assemble_dataset(test_tables)
    validate_dataset_invariants(train_data, require_target=True)
    validate_dataset_invariants(test_data, require_target=False)
    X, categorical = prepare_features(train_data)
    X_test, _ = prepare_features(test_data)
    X_test = align_feature_schemas(X, X_test, categorical)
    if TARGET in X.columns or ID_COL in X.columns:
        raise AssertionError("Target or identifier leaked into model features")
    y = train_data[TARGET].astype(str)
    if set(y.unique()) != set(CLASSES):
        raise ValueError(f"Unexpected target classes: {sorted(y.unique())}")

    folds = build_walk_forward_folds(train_tables)
    final_fold = folds[-1]
    train_idx = final_fold.train_idx
    tuning_idx = final_fold.tuning_idx
    evaluation_idx = final_fold.evaluation_idx
    LOG.info(
        "Strict temporal split | train=%s | tuning=%s | final evaluation=%s",
        f"{len(train_idx):,}",
        f"{len(tuning_idx):,}",
        f"{len(evaluation_idx):,}",
    )
    validation_model = build_model(iterations, RANDOM_SEED, thread_count)
    validation_model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
        cat_features=categorical,
        eval_set=(X.iloc[tuning_idx], y.iloc[tuning_idx]),
        early_stopping_rounds=80,
        use_best_model=True,
    )
    model_classes = [str(item) for item in validation_model.classes_]
    tuning_probabilities = validation_model.predict_proba(X.iloc[tuning_idx])
    tuning_raw_pred = np.asarray(model_classes, dtype=object)[
        np.argmax(tuning_probabilities, axis=1)
    ]
    tuning_raw_score = f1_score(
        y.iloc[tuning_idx], tuning_raw_pred, average="weighted"
    )
    multipliers, tuning_adjusted_pred, tuning_adjusted_score = (
        optimize_decision_multipliers(
            y.iloc[tuning_idx], tuning_probabilities, model_classes
        )
    )
    evaluation_probabilities = validation_model.predict_proba(X.iloc[evaluation_idx])
    evaluation_raw_pred = np.asarray(model_classes, dtype=object)[
        np.argmax(evaluation_probabilities, axis=1)
    ]
    evaluation_adjusted_pred = decision_argmax(
        evaluation_probabilities, model_classes, multipliers
    )
    final_raw_metrics = metric_record(y.iloc[evaluation_idx], evaluation_raw_pred)
    final_adjusted_metrics = metric_record(
        y.iloc[evaluation_idx], evaluation_adjusted_pred
    )
    LOG.info(
        "Tuning weighted F1 raw=%.6f adjusted=%.6f",
        tuning_raw_score,
        tuning_adjusted_score,
    )
    LOG.info(
        "Untouched final weighted F1 raw=%.6f adjusted=%.6f",
        final_raw_metrics["weighted_f1"],
        final_adjusted_metrics["weighted_f1"],
    )
    LOG.info("Decision multipliers selected on tuning only: %s", multipliers)
    save_validation_artifacts(
        y.iloc[evaluation_idx],
        evaluation_adjusted_pred,
        model_classes,
        final_fold,
        tuning_raw_score,
        tuning_adjusted_score,
        final_raw_metrics,
        final_adjusted_metrics,
        multipliers,
    )
    validation_frame = pd.DataFrame(
        {
            ID_COL: train_data.iloc[evaluation_idx][ID_COL].values,
            "true": y.iloc[evaluation_idx].values,
            "raw_pred": evaluation_raw_pred,
            "adjusted_pred": evaluation_adjusted_pred,
        }
    )
    for idx, class_name in enumerate(model_classes):
        validation_frame[f"model_probability_{class_name}"] = (
            evaluation_probabilities[:, idx]
        )
        validation_frame[f"decision_score_{class_name}"] = (
            evaluation_probabilities[:, idx] * multipliers[class_name]
        )
    validation_frame.to_csv(ARTIFACT_DIR / "validation_predictions.csv", index=False)

    best_iterations = max(120, int(validation_model.get_best_iteration() + 1))
    if validation_model.get_best_iteration() < 0:
        best_iterations = iterations
    LOG.info("Training single final model, %d iterations", best_iterations)
    model = build_model(best_iterations, RANDOM_SEED, thread_count)
    model.fit(X, y, cat_features=categorical)
    test_probabilities = model.predict_proba(X_test)
    model.save_model(ARTIFACT_DIR / "catboost_final.cbm")
    test_pred = decision_argmax(test_probabilities, model_classes, multipliers)
    submission = pd.DataFrame({ID_COL: test_data[ID_COL].values, TARGET: test_pred})
    submission.to_csv(ARTIFACT_DIR / "submission.csv", index=False)
    detailed = submission.copy()
    for idx, class_name in enumerate(model_classes):
        detailed[f"model_probability_{class_name}"] = test_probabilities[:, idx]
        detailed[f"decision_score_{class_name}"] = (
            test_probabilities[:, idx] * multipliers[class_name]
        )
    detailed.to_csv(ARTIFACT_DIR / "submission_with_probabilities.csv", index=False)
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACT_DIR / "feature_importance_cb.csv", index=False)
    pd.Series(X.columns, name="feature").to_csv(
        ARTIFACT_DIR / "feature_list.csv", index=False
    )
    LOG.info("Test prediction distribution:\n%s", submission[TARGET].value_counts().to_string())
    LOG.info("Artifacts saved to %s", ARTIFACT_DIR)


def predict_from_saved_models() -> None:
    """Build test artifacts from the selected single model without retraining."""
    model_path = ARTIFACT_DIR / "catboost_final.cbm"
    if not model_path.exists():
        raise FileNotFoundError("No artifacts/catboost_final.cbm model found")
    train_data = assemble_dataset(load_tables("train"))
    test_data = assemble_dataset(load_tables("test"))
    validate_dataset_invariants(train_data, require_target=True)
    validate_dataset_invariants(test_data, require_target=False)
    X, categorical = prepare_features(train_data)
    X_test, _ = prepare_features(test_data)
    X_test = align_feature_schemas(X, X_test, categorical)

    model = CatBoostClassifier()
    model.load_model(model_path)
    if list(model.feature_names_) != list(X.columns):
        raise RuntimeError("Saved model is incompatible with current feature schema")
    model_classes = [str(item) for item in model.classes_]
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    multipliers = metrics["decision_multipliers"]
    test_probabilities = model.predict_proba(X_test)
    test_pred = decision_argmax(test_probabilities, model_classes, multipliers)
    submission = pd.DataFrame({ID_COL: test_data[ID_COL].values, TARGET: test_pred})
    submission.to_csv(ARTIFACT_DIR / "submission.csv", index=False)
    detailed = submission.copy()
    for idx, class_name in enumerate(model_classes):
        detailed[f"model_probability_{class_name}"] = test_probabilities[:, idx]
        detailed[f"decision_score_{class_name}"] = (
            test_probabilities[:, idx] * multipliers[class_name]
        )
    detailed.to_csv(ARTIFACT_DIR / "submission_with_probabilities.csv", index=False)
    pd.DataFrame(
        {"feature": X.columns, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False).to_csv(
        ARTIFACT_DIR / "feature_importance_cb.csv", index=False
    )
    pd.Series(X.columns, name="feature").to_csv(
        ARTIFACT_DIR / "feature_list.csv", index=False
    )
    LOG.info("Loaded selected single model: %s", model_path.name)
    LOG.info("Test prediction distribution:\n%s", submission[TARGET].value_counts().to_string())
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_CONFIG.iterations)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: 120 iterations and one final model.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Load the selected single model and rebuild test artifacts.",
    )
    parser.add_argument(
        "--walk-forward-only",
        action="store_true",
        help="Run strict expanding temporal evaluation without final test training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.walk_forward_only:
        run_walk_forward_evaluation(args.iterations, args.thread_count)
        return
    if args.predict_only:
        predict_from_saved_models()
        return
    if args.quick:
        args.iterations = min(args.iterations, 120)
    train_pipeline(
        iterations=args.iterations,
        thread_count=args.thread_count,
    )


if __name__ == "__main__":
    main()
