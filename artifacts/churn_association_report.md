# Churn association analysis

Candidate categorical segments were discovered on the historical training period
and tested once on the later locked temporal cohort. Numeric hypotheses were predefined.
FDR controls multiple testing. Associations are predictive/statistical, not causal.

Validated associations: 19 of 54 tested hypotheses.

## invol_churn

- `attempt_last_cvc_check` / `unavailable`: validation effect 1.823, n=270, q=1.50e-13.
- `attempt_last_card_3d_secure_support` / `required`: validation effect 1.697, n=248, q=2.21e-09.
- `attempt_last_card_funding` / `prepaid`: validation effect 1.629, n=731, q=2.90e-21.
- `attempt_last_failure_code` / `card_declined`: validation effect 1.519, n=925, q=9.46e-19.
- `attempt_last_card_3d_secure_support` / `not_supported`: validation effect 1.332, n=297, q=2.67e-03.
- `attempt_last_cvc_check` / `pass`: validation effect 1.159, n=9,398, q=1.96e-43.
- `attempt_last_card_funding` / `debit`: validation effect 1.045, n=7,844, q=2.67e-03.
- `attempt_last_failure_code` / `unknown`: validation effect 0.966, n=14,056, q=1.22e-18.

## vol_churn

- `quiz_usage_plan` / `marketing`: validation effect 0.879, n=1,829, q=2.67e-03.
- `quiz_role` / `creator`: validation effect 0.740, n=229, q=3.83e-02.
- `quiz_role` / `filmmaker`: validation effect 0.640, n=117, q=4.04e-02.
- `quiz_cost_concern` / `all`: validation effect 0.090, n=3,808, q=1.75e-05.
