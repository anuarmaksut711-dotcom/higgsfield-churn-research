"""Rebuild submission artifacts from the saved selected model."""

import _bootstrap  # noqa: F401

from churn_pipeline import predict_from_saved_models


if __name__ == "__main__":
    predict_from_saved_models()
