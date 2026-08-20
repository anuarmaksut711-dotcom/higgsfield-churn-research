import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from churn_pipeline import (
    ARTIFACT_DIR,
    ID_COL,
    TARGET,
    add_event_clock,
    aggregate_attempts,
    aggregate_purchases,
    align_feature_schemas,
    assemble_dataset,
    build_walk_forward_folds,
    decision_argmax,
    prepare_features,
    validate_dataset_invariants,
)


def properties(user_ids: list[str], start: str = "1067-01-01 00:00:00+00:00"):
    return pd.DataFrame(
        {
            ID_COL: user_ids,
            "subscription_start_date": [start] * len(user_ids),
            "subscription_plan": ["Pro"] * len(user_ids),
            "country_code": ["US"] * len(user_ids),
        }
    )


def minimal_tables(include_target: bool = True) -> dict[str, pd.DataFrame]:
    user_ids = ["u1", "u2"]
    users = pd.DataFrame({ID_COL: user_ids})
    if include_target:
        users[TARGET] = ["not_churned", "vol_churn"]
    return {
        "users": users,
        "properties": properties(user_ids),
        "quizzes": pd.DataFrame({ID_COL: user_ids, "role": ["developer", None]}),
        "purchases": pd.DataFrame(
            {
                ID_COL: ["u1"],
                "transaction_id": ["p1"],
                "purchase_time": ["1067-01-02 00:00:00+00:00"],
                "purchase_amount_dollars": [10.0],
                "purchase_type": ["subscription"],
            }
        ),
        "attempts": pd.DataFrame(
            {
                ID_COL: ["u1", "u1"],
                "transaction_id": ["a1", "a2"],
                "transaction_time": [
                    "1067-01-02 00:00:00+00:00",
                    "1067-01-03 00:00:00+00:00",
                ],
                "amount_in_usd": [10.0, 10.0],
                "failure_code": [None, "card_declined"],
                "card_brand": ["visa", "visa"],
                "bank_name": ["bank", "bank"],
                "card_country": ["US", "US"],
            }
        ),
    }


class PipelineTests(unittest.TestCase):
    def test_event_clock_filters_future_events(self):
        props = properties(["u1"])
        events = pd.DataFrame(
            {
                ID_COL: ["u1", "u1", "u1"],
                "transaction_time": [
                    "1067-01-01 00:00:00+00:00",
                    "1067-01-14 23:59:59+00:00",
                    "1067-01-15 00:00:00+00:00",
                ],
            }
        )
        result = add_event_clock(events, props, "transaction_time")
        self.assertEqual(len(result), 2)
        self.assertLess(result["event_day"].max(), 14)

    def test_event_clock_filters_pre_subscription_events(self):
        props = properties(["u1"])
        events = pd.DataFrame(
            {
                ID_COL: ["u1", "u1"],
                "transaction_time": [
                    "1066-12-31 23:59:59+00:00",
                    "1067-01-01 00:00:00+00:00",
                ],
            }
        )
        result = add_event_clock(events, props, "transaction_time")
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result["event_day"].min(), 0)

    @staticmethod
    def temporal_tables() -> dict[str, pd.DataFrame]:
        user_ids = [f"u{i:03d}" for i in range(90)]
        dates = pd.date_range("1067-01-01", periods=90, freq="h", tz="UTC")
        return {
            "users": pd.DataFrame({ID_COL: user_ids}),
            "properties": pd.DataFrame(
                {ID_COL: user_ids, "subscription_start_date": dates.astype(str)}
            ),
        }

    def test_temporal_split_has_no_overlap(self):
        folds = build_walk_forward_folds(self.temporal_tables(), block_users=15, n_folds=3)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            sets = [set(fold.train_idx), set(fold.tuning_idx), set(fold.evaluation_idx)]
            self.assertTrue(sets[0].isdisjoint(sets[1]))
            self.assertTrue(sets[0].isdisjoint(sets[2]))
            self.assertTrue(sets[1].isdisjoint(sets[2]))

    def test_training_period_precedes_final_evaluation_period(self):
        final_fold = build_walk_forward_folds(
            self.temporal_tables(), block_users=15, n_folds=3
        )[-1]
        self.assertLess(
            pd.Timestamp(final_fold.train_end), pd.Timestamp(final_fold.evaluation_start)
        )
        self.assertLess(
            pd.Timestamp(final_fold.tuning_end), pd.Timestamp(final_fold.evaluation_start)
        )

    def test_assemble_dataset_returns_one_row_per_user(self):
        dataset = assemble_dataset(minimal_tables())
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset[ID_COL].is_unique)
        validate_dataset_invariants(dataset, require_target=True)

    def test_target_not_in_features(self):
        features, _ = prepare_features(assemble_dataset(minimal_tables()))
        self.assertNotIn(TARGET, features.columns)
        self.assertNotIn(ID_COL, features.columns)

    def test_train_test_feature_columns_match(self):
        train, categorical = prepare_features(assemble_dataset(minimal_tables(True)))
        test, _ = prepare_features(assemble_dataset(minimal_tables(False)))
        aligned = align_feature_schemas(train, test, categorical)
        self.assertListEqual(list(train.columns), list(aligned.columns))

    def test_unknown_categories_are_supported(self):
        train = pd.DataFrame({"category": ["known"], "value": [1.0]})
        other = pd.DataFrame({"category": [None], "extra": [3.0]})
        aligned = align_feature_schemas(train, other, ["category"])
        self.assertEqual(aligned.loc[0, "category"], "unknown")
        self.assertNotIn("extra", aligned.columns)

    def test_missing_values_are_supported(self):
        dataset = pd.DataFrame(
            {ID_COL: ["u1"], TARGET: ["not_churned"], "cat": [None], "num": [np.nan]}
        )
        features, categorical = prepare_features(dataset)
        self.assertEqual(categorical, ["cat"])
        self.assertEqual(features.loc[0, "cat"], "unknown")
        self.assertEqual(features.loc[0, "num"], 0.0)

    def test_aggregate_attempts_correct_counts(self):
        result = aggregate_attempts(
            minimal_tables()["attempts"], properties(["u1", "u2"])
        )
        row = result.set_index(ID_COL).loc["u1"]
        self.assertEqual(row["attempt_count"], 2)
        self.assertEqual(row["attempt_failure_count"], 1)
        self.assertEqual(row["attempt_success_count"], 1)
        self.assertAlmostEqual(row["attempt_failure_rate"], 0.5)

    def test_aggregate_purchases_correct_counts(self):
        result = aggregate_purchases(
            minimal_tables()["purchases"], properties(["u1", "u2"])
        )
        row = result.set_index(ID_COL).loc["u1"]
        self.assertEqual(row["purchase_count"], 1)
        self.assertEqual(row["purchase_amount_sum"], 10.0)

    def test_weighted_argmax(self):
        prediction = decision_argmax(
            np.array([[0.20, 0.50, 0.30]]),
            ["invol_churn", "not_churned", "vol_churn"],
            {"invol_churn": 1.0, "not_churned": 1.0, "vol_churn": 2.0},
        )
        self.assertEqual(prediction.tolist(), ["vol_churn"])

    def test_saved_model_prediction_shape(self):
        model_path = Path(ARTIFACT_DIR) / "catboost_final.cbm"
        if not model_path.exists():
            self.skipTest("No saved CatBoost model artifact")
        model = CatBoostClassifier()
        model.load_model(model_path)
        cat_indices = set(model.get_cat_feature_indices())
        row = {
            name: ("unknown" if index in cat_indices else 0.0)
            for index, name in enumerate(model.feature_names_)
        }
        probabilities = model.predict_proba(pd.DataFrame([row, row]))
        self.assertEqual(probabilities.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
