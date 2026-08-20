"""Leakage-safe model explanations, permutation importance, and stability."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from churn_pipeline import (
    ARTIFACT_DIR,
    CLASSES,
    ID_COL,
    RANDOM_SEED,
    TARGET,
    align_feature_schemas,
    assemble_dataset,
    build_model,
    build_walk_forward_folds,
    decision_argmax,
    load_tables,
    metric_record,
    optimize_decision_multipliers,
    prepare_features,
)

LOG = logging.getLogger("model_analysis")


def normalize_shap(values: np.ndarray, feature_count: int) -> np.ndarray:
    """Return multiclass SHAP in [rows, classes, features + bias] order."""
    if values.ndim != 3:
        raise ValueError(f"Expected 3D multiclass SHAP output, got {values.shape}")
    if values.shape[2] == feature_count + 1:
        return values
    if values.shape[1] == feature_count + 1:
        return np.transpose(values, (0, 2, 1))
    raise ValueError(f"Unexpected SHAP shape: {values.shape}")


def format_feature_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def train_temporal_models(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    tables,
    iterations: int,
    thread_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], CatBoostClassifier]:
    shap_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    final_model: CatBoostClassifier | None = None
    rng = np.random.default_rng(RANDOM_SEED)
    for fold_number, fold in enumerate(build_walk_forward_folds(tables), start=1):
        LOG.info("Training explanation model for %s", fold.name)
        model = build_model(iterations, RANDOM_SEED + fold_number - 1, thread_count)
        model.fit(
            X.iloc[fold.train_idx],
            y.iloc[fold.train_idx],
            cat_features=categorical,
            eval_set=(X.iloc[fold.tuning_idx], y.iloc[fold.tuning_idx]),
            early_stopping_rounds=80,
            use_best_model=True,
            verbose=False,
        )
        classes = [str(item) for item in model.classes_]
        tuning_probability = model.predict_proba(X.iloc[fold.tuning_idx])
        multipliers, _, _ = optimize_decision_multipliers(
            y.iloc[fold.tuning_idx], tuning_probability, classes
        )
        eval_X = X.iloc[fold.evaluation_idx].copy()
        eval_y = y.iloc[fold.evaluation_idx]
        eval_probability = model.predict_proba(eval_X)
        base_prediction = decision_argmax(eval_probability, classes, multipliers)
        base_metrics = metric_record(eval_y, base_prediction)

        sample_size = min(1_500, len(eval_X))
        sample_positions = rng.choice(len(eval_X), size=sample_size, replace=False)
        sample = eval_X.iloc[sample_positions]
        shap = normalize_shap(
            np.asarray(
                model.get_feature_importance(
                    Pool(sample, cat_features=categorical), type="ShapValues"
                )
            ),
            X.shape[1],
        )
        for class_index, class_name in enumerate(classes):
            mean_abs = np.abs(shap[:, class_index, :-1]).mean(axis=0)
            for feature, importance in zip(X.columns, mean_abs):
                shap_rows.append(
                    {
                        "fold": fold.name,
                        "class": class_name,
                        "feature": feature,
                        "mean_abs_shap": float(importance),
                        "sample_users": sample_size,
                    }
                )

        for feature in X.columns:
            permuted = eval_X.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            probability = model.predict_proba(permuted)
            prediction = decision_argmax(probability, classes, multipliers)
            shuffled_metrics = metric_record(eval_y, prediction)
            permutation_rows.append(
                {
                    "fold": fold.name,
                    "feature": feature,
                    "weighted_f1_drop": base_metrics["weighted_f1"]
                    - shuffled_metrics["weighted_f1"],
                    "macro_f1_drop": base_metrics["macro_f1"]
                    - shuffled_metrics["macro_f1"],
                    **{
                        f"{class_name}_f1_drop": base_metrics[f"{class_name}_f1"]
                        - shuffled_metrics[f"{class_name}_f1"]
                        for class_name in CLASSES
                    },
                    "evaluation_users": len(eval_X),
                }
            )
        if fold.name == "fold_3":
            final_model = model
    assert final_model is not None
    return shap_rows, permutation_rows, final_model


def build_test_explanations(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical: list[str],
    test_ids: pd.Series,
) -> None:
    model = CatBoostClassifier()
    model.load_model(ARTIFACT_DIR / "catboost_final.cbm")
    if list(model.feature_names_) != list(X_train.columns):
        raise AssertionError("Final model feature schema does not match current features")
    classes = [str(item) for item in model.classes_]
    shap = normalize_shap(
        np.asarray(
            model.get_feature_importance(
                Pool(X_test, cat_features=categorical), type="ShapValues"
            )
        ),
        X_test.shape[1],
    )
    submission = pd.read_csv(ARTIFACT_DIR / "submission.csv")
    if not submission[ID_COL].equals(test_ids.reset_index(drop=True)):
        raise AssertionError("Submission and test feature order differ")
    class_to_index = {name: index for index, name in enumerate(classes)}
    rows: list[dict[str, object]] = []
    feature_names = np.asarray(X_test.columns, dtype=object)
    for row_index, predicted_class in enumerate(submission[TARGET].astype(str)):
        contributions = shap[row_index, class_to_index[predicted_class], :-1]
        positive = np.argsort(contributions)[::-1][:3]
        negative = np.argsort(contributions)[:3]
        row: dict[str, object] = {
            ID_COL: test_ids.iloc[row_index],
            "prediction": predicted_class,
            "explanation_type": "CatBoost SHAP contribution to predicted-class raw score",
        }
        for rank, feature_index in enumerate(positive, start=1):
            feature = str(feature_names[feature_index])
            row[f"positive_{rank}_feature"] = feature
            row[f"positive_{rank}_value"] = format_feature_value(
                X_test.iloc[row_index, feature_index]
            )
            row[f"positive_{rank}_shap"] = float(contributions[feature_index])
        for rank, feature_index in enumerate(negative, start=1):
            feature = str(feature_names[feature_index])
            row[f"negative_{rank}_feature"] = feature
            row[f"negative_{rank}_value"] = format_feature_value(
                X_test.iloc[row_index, feature_index]
            )
            row[f"negative_{rank}_shap"] = float(contributions[feature_index])
        rows.append(row)
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "test_model_explanations.csv", index=False)
    LOG.info("Saved SHAP explanations for %d test users", len(rows))


def main(iterations: int = 650, thread_count: int = -1) -> None:
    train_tables = load_tables("train")
    test_tables = load_tables("test")
    train_data = assemble_dataset(train_tables)
    test_data = assemble_dataset(test_tables)
    X, categorical = prepare_features(train_data)
    X_test, _ = prepare_features(test_data)
    X_test = align_feature_schemas(X, X_test, categorical)
    y = train_data[TARGET].astype(str)
    shap_rows, permutation_rows, _ = train_temporal_models(
        X, y, categorical, train_tables, iterations, thread_count
    )
    shap_frame = pd.DataFrame(shap_rows)
    permutation_frame = pd.DataFrame(permutation_rows)
    shap_frame.to_csv(ARTIFACT_DIR / "shap_importance_by_fold.csv", index=False)
    permutation_frame.to_csv(
        ARTIFACT_DIR / "permutation_importance_by_fold.csv", index=False
    )
    stability = (
        shap_frame.groupby(["class", "feature"])["mean_abs_shap"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_abs_shap_mean",
                "std": "mean_abs_shap_std",
                "min": "mean_abs_shap_min",
                "max": "mean_abs_shap_max",
            }
        )
    )
    stability["stability_cv"] = stability["mean_abs_shap_std"] / stability[
        "mean_abs_shap_mean"
    ].replace(0, np.nan)
    stability.to_csv(ARTIFACT_DIR / "shap_stability.csv", index=False)
    build_test_explanations(
        X, X_test, categorical, test_data[ID_COL].reset_index(drop=True)
    )
    LOG.info(
        "Top permutation features:\n%s",
        permutation_frame.groupby("feature")["weighted_f1_drop"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .to_string(),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
