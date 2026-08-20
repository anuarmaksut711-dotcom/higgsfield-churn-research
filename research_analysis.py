"""Error analysis and categorical drift diagnostics for temporal evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from churn_pipeline import (
    ARTIFACT_DIR,
    CLASSES,
    ID_COL,
    assemble_dataset,
    build_walk_forward_folds,
    load_tables,
)

LOG = logging.getLogger("research_analysis")

NUMERIC_ERROR_FEATURES = [
    "attempt_count",
    "purchase_count",
    "attempt_failure_rate",
    "attempt_failure_count",
    "attempt_success_count",
    "attempt_active_days",
    "purchase_active_days",
    "activity_days_total",
    "attempt_last_day",
    "purchase_last_day",
]

CATEGORICAL_ERROR_FEATURES = [
    "country_code",
    "subscription_plan",
    "quiz_role",
    "quiz_frustration",
    "attempt_last_card_country",
    "attempt_last_bank_name",
    "attempt_last_card_funding",
    "attempt_last_card_3d_secure_support",
    "attempt_last_cvc_check",
    "attempt_last_failure_code",
]

DRIFT_FEATURES = [
    "country_code",
    "attempt_last_card_country",
    "attempt_last_bank_name",
    "attempt_last_billing_address_country",
    "attempt_last_card_3d_secure_support",
    "attempt_last_card_funding",
]


def js_divergence(left: pd.Series, right: pd.Series) -> float:
    left = left.fillna("unknown").astype(str).value_counts(normalize=True)
    right = right.fillna("unknown").astype(str).value_counts(normalize=True)
    values = left.index.union(right.index)
    p = left.reindex(values, fill_value=0.0).to_numpy(dtype=float)
    q = right.reindex(values, fill_value=0.0).to_numpy(dtype=float)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def run_error_analysis(train_data: pd.DataFrame) -> None:
    predictions = pd.read_csv(ARTIFACT_DIR / "walk_forward_predictions.csv")
    final_predictions = predictions.loc[predictions["fold"].eq("fold_3")].copy()
    merged = final_predictions.merge(train_data, on=ID_COL, how="left", validate="one_to_one")
    merged["transition"] = merged["true"] + " -> " + merged["adjusted_pred"]
    merged["is_correct"] = merged["true"].eq(merged["adjusted_pred"])
    probability_columns = [f"model_probability_{class_name}" for class_name in CLASSES]
    merged["max_model_probability"] = merged[probability_columns].max(axis=1)
    merged["high_model_probability_error"] = (
        ~merged["is_correct"] & merged["max_model_probability"].ge(0.70)
    )

    transition_counts = (
        merged.groupby(["true", "adjusted_pred", "transition"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    transition_counts["share_of_final_evaluation"] = transition_counts["count"] / len(merged)
    transition_counts.to_csv(ARTIFACT_DIR / "error_transition_counts.csv", index=False)

    numeric_rows: list[dict[str, object]] = []
    for transition, subset in merged.groupby("transition", observed=True):
        for feature in NUMERIC_ERROR_FEATURES:
            if feature not in subset.columns:
                continue
            values = pd.to_numeric(subset[feature], errors="coerce")
            numeric_rows.append(
                {
                    "transition": transition,
                    "feature": feature,
                    "count": int(values.notna().sum()),
                    "mean": float(values.mean()) if values.notna().any() else np.nan,
                    "median": float(values.median()) if values.notna().any() else np.nan,
                    "std": float(values.std()) if values.notna().sum() > 1 else np.nan,
                }
            )
    pd.DataFrame(numeric_rows).to_csv(
        ARTIFACT_DIR / "error_numeric_patterns.csv", index=False
    )

    categorical_rows: list[dict[str, object]] = []
    for feature in CATEGORICAL_ERROR_FEATURES:
        if feature not in merged.columns:
            continue
        base = merged[feature].fillna("unknown").astype(str).value_counts(normalize=True)
        for transition, subset in merged.groupby("transition", observed=True):
            distribution = (
                subset[feature].fillna("unknown").astype(str).value_counts(normalize=True)
            )
            counts = subset[feature].fillna("unknown").astype(str).value_counts()
            for value, proportion in distribution.head(10).items():
                base_proportion = float(base.get(value, 0.0))
                categorical_rows.append(
                    {
                        "transition": transition,
                        "feature": feature,
                        "value": value,
                        "count": int(counts[value]),
                        "proportion": float(proportion),
                        "base_proportion": base_proportion,
                        "lift_vs_final_cohort": (
                            float(proportion / base_proportion)
                            if base_proportion > 0
                            else np.nan
                        ),
                    }
                )
    categorical = pd.DataFrame(categorical_rows)
    categorical.to_csv(ARTIFACT_DIR / "error_categorical_patterns.csv", index=False)

    selected_columns = [
        ID_COL,
        "true",
        "raw_pred",
        "adjusted_pred",
        "transition",
        "max_model_probability",
        *probability_columns,
        *[column for column in NUMERIC_ERROR_FEATURES if column in merged.columns],
        *[column for column in CATEGORICAL_ERROR_FEATURES if column in merged.columns],
    ]
    merged.loc[merged["high_model_probability_error"], selected_columns].sort_values(
        "max_model_probability", ascending=False
    ).to_csv(ARTIFACT_DIR / "high_model_probability_errors.csv", index=False)

    class_rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        true_class = merged["true"].eq(class_name)
        pred_class = merged["adjusted_pred"].eq(class_name)
        class_rows.extend(
            [
                {"class": class_name, "error_role": "TP", "count": int((true_class & pred_class).sum())},
                {"class": class_name, "error_role": "FP", "count": int((~true_class & pred_class).sum())},
                {"class": class_name, "error_role": "FN", "count": int((true_class & ~pred_class).sum())},
            ]
        )
    pd.DataFrame(class_rows).to_csv(ARTIFACT_DIR / "class_error_counts.csv", index=False)

    key_transitions = [
        "vol_churn -> not_churned",
        "vol_churn -> invol_churn",
        "invol_churn -> not_churned",
    ]
    report_lines = [
        "# Error analysis — locked final temporal fold",
        "",
        f"Evaluation users: {len(merged):,}.",
        "Model probabilities below are uncalibrated model outputs, not absolute risk estimates.",
        "",
        "## Key transitions",
        "",
    ]
    count_map = transition_counts.set_index("transition")["count"].to_dict()
    for transition in key_transitions:
        report_lines.append(f"- `{transition}`: {int(count_map.get(transition, 0)):,} users.")
    report_lines.extend(
        [
            "",
            "## High-model-probability errors",
            "",
            f"- Errors with max raw model probability ≥ 0.70: {int(merged['high_model_probability_error'].sum()):,}.",
            "- This threshold identifies confident-looking model outputs; it does not imply calibrated 70% risk.",
            "",
            "## Most overrepresented segments in key errors",
            "",
        ]
    )
    for transition in key_transitions:
        candidates = categorical.loc[
            categorical["transition"].eq(transition)
            & categorical["count"].ge(30)
            & categorical["base_proportion"].ge(0.01)
        ].sort_values("lift_vs_final_cohort", ascending=False)
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        report_lines.append(
            f"- `{transition}`: `{row['feature']}={row['value']}` is {row['lift_vs_final_cohort']:.2f}× as common as in the full final cohort (n={int(row['count']):,})."
        )
    (ARTIFACT_DIR / "error_analysis_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    LOG.info("Final-fold errors analyzed: %d", len(merged))


def run_drift_analysis(train_data: pd.DataFrame, test_data: pd.DataFrame, tables) -> None:
    final_fold = build_walk_forward_folds(tables)[-1]
    cohort = pd.Series("unused", index=train_data.index, dtype="object")
    cohort.iloc[final_fold.train_idx] = "historical_train"
    cohort.iloc[final_fold.tuning_idx] = "tuning"
    cohort.iloc[final_fold.evaluation_idx] = "final_evaluation"
    train = train_data.copy()
    train["cohort"] = cohort
    test = test_data.copy()
    test["cohort"] = "test"
    combined = pd.concat([train, test], ignore_index=True, sort=False)

    distribution_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    reference = combined.loc[combined["cohort"].eq("historical_train")]
    for feature in DRIFT_FEATURES:
        if feature not in combined.columns:
            continue
        for cohort_name, subset in combined.groupby("cohort", observed=True):
            values = subset[feature].fillna("unknown").astype(str)
            counts = values.value_counts()
            for value, count in counts.items():
                distribution_rows.append(
                    {
                        "feature": feature,
                        "cohort": cohort_name,
                        "value": value,
                        "count": int(count),
                        "proportion": float(count / len(subset)),
                    }
                )
            if cohort_name != "historical_train":
                drift_rows.append(
                    {
                        "feature": feature,
                        "reference_cohort": "historical_train",
                        "comparison_cohort": cohort_name,
                        "jensen_shannon_divergence": js_divergence(
                            reference[feature], subset[feature]
                        ),
                        "reference_users": len(reference),
                        "comparison_users": len(subset),
                    }
                )
    pd.DataFrame(distribution_rows).to_csv(
        ARTIFACT_DIR / "categorical_distribution_by_cohort.csv", index=False
    )
    drift = pd.DataFrame(drift_rows).sort_values(
        "jensen_shannon_divergence", ascending=False
    )
    drift.to_csv(ARTIFACT_DIR / "categorical_drift.csv", index=False)
    LOG.info("Largest categorical drift:\n%s", drift.head(10).to_string(index=False))


def main() -> None:
    train_tables = load_tables("train")
    test_tables = load_tables("test")
    train_data = assemble_dataset(train_tables)
    test_data = assemble_dataset(test_tables)
    run_error_analysis(train_data)
    run_drift_analysis(train_data, test_data, train_tables)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
