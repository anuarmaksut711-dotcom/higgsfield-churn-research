"""Separation between model attribution and business intervention outputs."""

MODEL_EXPLANATION_ARTIFACT = "artifacts/test_model_explanations.csv"
BUSINESS_INTERVENTION_ARTIFACT = "artifacts/user_insights.csv"
ASSOCIATION_ARTIFACT = "artifacts/churn_driver_evidence.csv"

__all__ = [
    "MODEL_EXPLANATION_ARTIFACT",
    "BUSINESS_INTERVENTION_ARTIFACT",
    "ASSOCIATION_ARTIFACT",
]
