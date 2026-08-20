# Error analysis — locked final temporal fold

Evaluation users: 15,000.
Model probabilities below are uncalibrated model outputs, not absolute risk estimates.

## Key transitions

- `vol_churn -> not_churned`: 2,363 users.
- `vol_churn -> invol_churn`: 698 users.
- `invol_churn -> not_churned`: 1,401 users.

## High-model-probability errors

- Errors with max raw model probability ≥ 0.70: 681.
- This threshold identifies confident-looking model outputs; it does not imply calibrated 70% risk.

## Most overrepresented segments in key errors

- `vol_churn -> not_churned`: `country_code=AU` is 1.65× as common as in the full final cohort (n=62).
- `vol_churn -> invol_churn`: `attempt_last_bank_name=REVOLUT BANK UAB` is 2.58× as common as in the full final cohort (n=56).
- `invol_churn -> not_churned`: `country_code=TR` is 1.79× as common as in the full final cohort (n=62).
