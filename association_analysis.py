"""Discovery/validation analysis of churn associations (not causal effects)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from churn_explainer import CATEGORICAL_DRIVERS, NUMERIC_DRIVERS
from churn_pipeline import (
    ARTIFACT_DIR,
    TARGET,
    assemble_dataset,
    build_walk_forward_folds,
    load_tables,
)

LOG = logging.getLogger("association_analysis")


def numeric_validation(discovery: pd.DataFrame, validation: pd.DataFrame) -> list[dict]:
    rows: list[dict[str, object]] = []
    for churn_type, features in NUMERIC_DRIVERS.items():
        for feature in features:
            if feature not in validation.columns:
                continue
            discovery_target = pd.to_numeric(
                discovery.loc[discovery[TARGET].eq(churn_type), feature], errors="coerce"
            ).fillna(0)
            discovery_base = pd.to_numeric(
                discovery.loc[discovery[TARGET].eq("not_churned"), feature], errors="coerce"
            ).fillna(0)
            target = pd.to_numeric(
                validation.loc[validation[TARGET].eq(churn_type), feature], errors="coerce"
            ).fillna(0)
            base = pd.to_numeric(
                validation.loc[validation[TARGET].eq("not_churned"), feature], errors="coerce"
            ).fillna(0)
            _, p_value = mannwhitneyu(
                target, base, alternative="two-sided", method="asymptotic"
            )
            discovery_diff = float(discovery_target.mean() - discovery_base.mean())
            validation_diff = float(target.mean() - base.mean())
            pooled_std = float(pd.concat([target, base]).std(ddof=0))
            rows.append(
                {
                    "churn_type": churn_type,
                    "feature": feature,
                    "evidence_type": "numeric_predefined",
                    "segment": "all",
                    "discovery_effect": discovery_diff,
                    "validation_effect": validation_diff,
                    "validation_standardized_effect": (
                        validation_diff / pooled_std if pooled_std else 0.0
                    ),
                    "validation_churn_value": float(target.mean()),
                    "validation_reference_value": float(base.mean()),
                    "validation_support": int(len(target)),
                    "p_value": float(p_value),
                    "direction_replicated": bool(
                        np.sign(discovery_diff) == np.sign(validation_diff)
                    ),
                }
            )
    return rows


def discover_categorical_candidates(discovery: pd.DataFrame) -> list[dict]:
    candidates: list[dict[str, object]] = []
    for churn_type, features in CATEGORICAL_DRIVERS.items():
        class_rate = float(discovery[TARGET].eq(churn_type).mean())
        for feature in features:
            if feature not in discovery.columns:
                continue
            table = pd.DataFrame(
                {
                    "value": discovery[feature].fillna("unknown").astype(str),
                    "is_target": discovery[TARGET].eq(churn_type),
                }
            )
            grouped = table.groupby("value", observed=True)["is_target"].agg(["mean", "size"])
            grouped = grouped.loc[grouped["size"].ge(100)].copy()
            grouped["lift"] = grouped["mean"] / class_rate
            grouped["distance_from_one"] = (grouped["lift"] - 1).abs()
            for value, row in grouped.sort_values(
                ["distance_from_one", "size"], ascending=False
            ).head(5).iterrows():
                candidates.append(
                    {
                        "churn_type": churn_type,
                        "feature": feature,
                        "segment": str(value),
                        "discovery_rate": float(row["mean"]),
                        "discovery_lift": float(row["lift"]),
                        "discovery_support": int(row["size"]),
                    }
                )
    return candidates


def validate_categorical_candidates(
    candidates: list[dict], validation: pd.DataFrame
) -> list[dict]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        churn_type = str(candidate["churn_type"])
        feature = str(candidate["feature"])
        segment = str(candidate["segment"])
        values = validation[feature].fillna("unknown").astype(str)
        in_segment = values.eq(segment)
        is_target = validation[TARGET].eq(churn_type)
        target_in = int((in_segment & is_target).sum())
        other_in = int((in_segment & ~is_target).sum())
        target_out = int((~in_segment & is_target).sum())
        other_out = int((~in_segment & ~is_target).sum())
        observed = [[target_in, other_in], [target_out, other_out]]
        if min(map(sum, observed)) == 0 or min(np.asarray(observed).sum(axis=0)) == 0:
            p_value = 1.0
        else:
            _, p_value, _, _ = chi2_contingency(observed, correction=False)
        segment_rate = target_in / max(1, target_in + other_in)
        class_rate = float(is_target.mean())
        validation_lift = segment_rate / class_rate if class_rate else np.nan
        discovery_lift = float(candidate["discovery_lift"])
        rows.append(
            {
                **candidate,
                "evidence_type": "categorical_discovered_then_validated",
                "validation_effect": float(validation_lift - 1),
                "validation_standardized_effect": float(validation_lift),
                "validation_churn_value": float(segment_rate),
                "validation_reference_value": class_rate,
                "validation_support": int(in_segment.sum()),
                "p_value": float(p_value),
                "direction_replicated": bool(
                    np.sign(discovery_lift - 1) == np.sign(validation_lift - 1)
                ),
            }
        )
    return rows


def apply_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["p_value"] = pd.to_numeric(frame["p_value"], errors="coerce").fillna(1.0)
    result = frame.sort_values("p_value").reset_index(drop=True).copy()
    ranks = np.arange(1, len(result) + 1)
    result["q_value"] = np.minimum.accumulate(
        (result["p_value"].to_numpy() * len(result) / ranks)[::-1]
    )[::-1].clip(max=1.0)
    result["validated"] = result["q_value"].lt(0.05) & result[
        "direction_replicated"
    ]
    return result


def add_predictive_context(evidence: pd.DataFrame) -> pd.DataFrame:
    shap_path = ARTIFACT_DIR / "shap_importance_by_fold.csv"
    permutation_path = ARTIFACT_DIR / "permutation_importance_by_fold.csv"
    if shap_path.exists():
        shap = pd.read_csv(shap_path)
        shap = (
            shap.groupby(["class", "feature"])["mean_abs_shap"]
            .mean()
            .rename("mean_abs_shap")
            .reset_index()
            .rename(columns={"class": "churn_type"})
        )
        evidence = evidence.merge(shap, on=["churn_type", "feature"], how="left")
    if permutation_path.exists():
        permutation = (
            pd.read_csv(permutation_path)
            .groupby("feature")["weighted_f1_drop"]
            .mean()
            .rename("mean_weighted_f1_permutation_drop")
            .reset_index()
        )
        evidence = evidence.merge(permutation, on="feature", how="left")
    return evidence


def main() -> None:
    tables = load_tables("train")
    data = assemble_dataset(tables)
    final_fold = build_walk_forward_folds(tables)[-1]
    discovery = data.iloc[final_fold.train_idx].copy()
    validation = data.iloc[final_fold.evaluation_idx].copy()
    rows = numeric_validation(discovery, validation)
    candidates = discover_categorical_candidates(discovery)
    rows.extend(validate_categorical_candidates(candidates, validation))
    evidence = add_predictive_context(apply_fdr(pd.DataFrame(rows)))
    evidence.to_csv(ARTIFACT_DIR / "churn_driver_evidence.csv", index=False)

    validated = evidence.loc[evidence["validated"]].copy()
    lines = [
        "# Churn association analysis",
        "",
        "Candidate categorical segments were discovered on the historical training period",
        "and tested once on the later locked temporal cohort. Numeric hypotheses were predefined.",
        "FDR controls multiple testing. Associations are predictive/statistical, not causal.",
        "",
        f"Validated associations: {len(validated)} of {len(evidence)} tested hypotheses.",
        "",
    ]
    for churn_type in ("invol_churn", "vol_churn"):
        lines.extend([f"## {churn_type}", ""])
        subset = validated.loc[validated["churn_type"].eq(churn_type)].copy()
        subset = subset.reindex(
            subset["validation_standardized_effect"].abs().sort_values(ascending=False).index
        ).head(8)
        for row in subset.itertuples():
            lines.append(
                f"- `{row.feature}` / `{row.segment}`: validation effect "
                f"{row.validation_standardized_effect:.3f}, n={row.validation_support:,}, "
                f"q={row.q_value:.2e}."
            )
        lines.append("")
    (ARTIFACT_DIR / "churn_association_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    LOG.info("Validated associations: %d/%d", len(validated), len(evidence))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
