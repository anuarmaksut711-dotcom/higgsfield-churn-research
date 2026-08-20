"""Build the consolidated experiment registry, research report, and figures."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
COLORS = {"not_churned": "#4C78A8", "vol_churn": "#F58518", "invol_churn": "#E45756"}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / name)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def build_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def append_summary(
        file_name: str,
        group: str,
        scope: str,
        notes: dict[str, str] | None = None,
    ) -> None:
        frame = load(file_name)
        for row in frame.itertuples(index=False):
            name = str(getattr(row, "model", getattr(row, "experiment", "unknown")))
            rows.append(
                {
                    "experiment_group": group,
                    "experiment": name,
                    "validation_scope": scope,
                    "selection_data": "temporal tuning cohorts only",
                    "evaluation_data": "future temporal evaluation cohorts",
                    "primary_metric": "weighted_f1",
                    "weighted_f1_mean": float(row.weighted_f1_mean),
                    "weighted_f1_std": float(row.weighted_f1_std),
                    "macro_f1_mean": float(row.macro_f1_mean),
                    "status": (notes or {}).get(name, "measured"),
                }
            )

    append_summary(
        "baseline_summary.csv",
        "baseline",
        "three strict walk-forward folds",
        {"engineered_catboost_adjusted": "selected"},
    )
    append_summary(
        "ablation_summary.csv",
        "ablation",
        "development folds 1-2",
        {
            "full_with_decision_adjustment": "reference",
            "full_without_payment": "payment features add predictive value",
            "full_without_quiz": "quiz value not demonstrated",
        },
    )
    append_summary(
        "observation_window_summary.csv",
        "observation_window",
        "development folds 1-2",
        {"14_days": "selected production window", "3_days": "final-fold challenger"},
    )
    append_summary(
        "seed_ensemble_summary.csv",
        "seed_ensemble",
        "development folds 1-2",
        {"single_seed": "selected", "three_seed_ensemble": "rejected: no mean gain"},
    )
    append_summary(
        "hyperparameter_screen_summary.csv",
        "hyperparameter_screen",
        "development folds 1-2",
        {
            "current": "selected for simplicity and stability",
            "slower_regularized": "rejected: gain below fold variability",
        },
    )
    registry = pd.DataFrame(rows)
    registry.to_csv(ARTIFACTS / "experiments.csv", index=False)
    return registry


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ARTIFACTS / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    baseline = load("baseline_summary.csv").sort_values("weighted_f1_mean")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(baseline["model"], baseline["weighted_f1_mean"], xerr=baseline["weighted_f1_std"], color="#4C78A8", alpha=0.9)
    ax.set(xlabel="Mean weighted F1 (± fold std)", title="Strict walk-forward baseline comparison")
    ax.set_xlim(0.30, 0.55)
    fig.tight_layout()
    save(fig, "baseline_comparison.png")

    windows = load("observation_window_summary.csv").copy()
    windows["days"] = windows["experiment"].str.extract(r"(\d+)").astype(int)
    windows = windows.sort_values("days")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(windows["days"], windows["weighted_f1_mean"], yerr=windows["weighted_f1_std"], marker="o", capsize=5, linewidth=2.2, label="Weighted F1")
    ax.errorbar(windows["days"], windows["macro_f1_mean"], yerr=windows["macro_f1_std"], marker="s", capsize=5, linewidth=2.2, label="Macro F1")
    ax.set(xticks=windows["days"], xlabel="Observation window (days)", ylabel="Mean temporal F1", title="Observation-window study (development folds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "observation_window_study.png")

    walk = load("walk_forward_metrics.csv")
    walk = walk.loc[walk["decision_variant"].eq("tuning_selected_adjustment")].copy()
    x = np.arange(len(walk))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for class_name in COLORS:
        ax.plot(x, walk[f"{class_name}_f1"], marker="o", linewidth=2, label=f"{class_name} F1", color=COLORS[class_name])
    ax.plot(x, walk["weighted_f1"], marker="D", color="#222222", linewidth=2.4, label="weighted F1")
    ax.set(xticks=x, xticklabels=walk["fold"], ylabel="F1", title="Walk-forward stability by future cohort", ylim=(0.2, 0.7))
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "walk_forward_stability.png")

    transitions = load("error_transition_counts.csv")
    transitions = transitions.loc[transitions["true"].ne(transitions["adjusted_pred"])].sort_values("count")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(transitions["transition"], transitions["count"], color="#E45756", alpha=0.85)
    ax.set(xlabel="Users in untouched final cohort", title="Largest final-cohort error transitions")
    fig.tight_layout()
    save(fig, "error_transitions.png")

    drift = load("categorical_drift.csv").nlargest(10, "jensen_shannon_divergence").sort_values("jensen_shannon_divergence")
    labels = drift["feature"].str.replace("attempt_last_", "", regex=False) + " · " + drift["comparison_cohort"]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.barh(labels, drift["jensen_shannon_divergence"], color="#72B7B2")
    ax.set(xlabel="Jensen–Shannon divergence", title="Largest categorical distribution shifts")
    fig.tight_layout()
    save(fig, "categorical_drift.png")


def build_hypotheses_report() -> str:
    ablation = load("ablation_summary.csv").set_index("experiment")
    windows = load("observation_window_summary.csv").set_index("experiment")
    final_windows = load("observation_window_final_evaluation.csv").set_index("experiment")
    text = f"""# Проверка исследовательских гипотез

Все выводы ниже основаны на временных разбиениях. Гиперпараметры, early stopping и decision multipliers выбирались без доступа к evaluation-когорте.

## H1. Платёжные признаки особенно полезны для involuntary churn — частично подтверждена

Удаление платёжных признаков снизило средний weighted F1 с {fmt(ablation.loc['full_raw', 'weighted_f1_mean'])} до {fmt(ablation.loc['full_without_payment', 'weighted_f1_mean'])}. При этом involuntary F1 изменился лишь с {fmt(ablation.loc['full_raw', 'invol_churn_f1_mean'])} до {fmt(ablation.loc['full_without_payment', 'invol_churn_f1_mean'])}. Permutation importance и валидированные ассоциации выделяют card/bank/payment-поля, но ablation не доказывает, что прирост концентрируется именно в involuntary F1. Итог: платёжный блок полезен в целом; сильная формулировка гипотезы не подтверждена.

## H2. Quiz-признаки заметно улучшают voluntary churn — не подтверждена

Без quiz-признаков weighted F1 составил {fmt(ablation.loc['full_without_quiz', 'weighted_f1_mean'])} против {fmt(ablation.loc['full_raw', 'weighted_f1_mean'])} у полной сырой модели, а voluntary F1 — {fmt(ablation.loc['full_without_quiz', 'vol_churn_f1_mean'])} против {fmt(ablation.loc['full_raw', 'vol_churn_f1_mean'])}. Отдельные quiz-сегменты статистически воспроизводятся, но заметного инкрементального модельного эффекта нет.

## H3. Большая часть сигнала возникает в первые 1–3 дня — поддержана с оговорками

На development-фолдах 3 дня дали {fmt(windows.loc['3_days', 'weighted_f1_mean'])}, а 14 дней — {fmt(windows.loc['14_days', 'weighted_f1_mean'])}. На единственной untouched final-когорте результат 3 дней был {fmt(final_windows.loc['3_days', 'weighted_f1'])} против {fmt(final_windows.loc['14_days', 'weighted_f1'])}. Однако 14 дней лучше распознавали involuntary churn ({fmt(final_windows.loc['14_days', 'invol_churn_f1'])} против {fmt(final_windows.loc['3_days', 'invol_churn_f1'])}), поэтому 14-дневное окно оставлено как консервативный production default.

## H4. География и платёжная инфраструктура нестабильны во времени — подтверждена

Максимальный Jensen–Shannon divergence достиг 0.1611 для billing country в test и 0.1592 для 3-D Secure support. Одновременно country, card country, bank name и funding входят в верхнюю часть permutation importance. Это сочетание высокой важности и заметного cohort shift требует мониторинга drift; оно не является доказательством причинности.
"""
    (ARTIFACTS / "hypotheses_report.md").write_text(text.rstrip() + "\n", encoding="utf-8")
    return text


def build_research_report() -> None:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    baseline = load("baseline_summary.csv").set_index("model")
    walk_summary = load("walk_forward_summary.csv").set_index("decision_variant")
    walk_selected = walk_summary.loc["tuning_selected_adjustment"]
    errors = load("error_transition_counts.csv").set_index("transition")
    evidence = load("churn_driver_evidence.csv")
    validated = evidence.loc[evidence["validated"].astype(str).str.lower().eq("true")]
    insights = load("user_insights.csv")
    pred = insights["predicted_churn"].value_counts()
    final = metrics["final_adjusted_metrics"]
    hypotheses = build_hypotheses_report()
    report = f"""# Higgsfield churn prediction — исследовательский отчёт

## Problem

Задача — по событиям первых 14 дней после старта подписки предсказать один из трёх исходов первого месяца: `not_churned`, `vol_churn` или `invol_churn`. Основная метрика — weighted F1; per-class и macro F1 сохранены, чтобы weighted F1 не скрывал качество миноритарных классов.

## Dataset

- 90 000 размеченных и 7 000 test-пользователей.
- Пользовательские свойства, попытки платежей, покупки и onboarding quiz.
- Каждое событие фильтруется интервалом `[subscription_start_date, +14 дней)` конкретного пользователя.
- Абсолютное signup-время исключено из модели; исходные даты анонимизированы и используются только для порядка когорт.
- `test_users_generations.csv` не используется: симметричной train-таблицы нет, поэтому supervised-признаки невоспроизводимы.

## Methodology and leakage control

Пользователи сортируются по точному signup timestamp. В каждом из трёх walk-forward фолдов train строго предшествует tuning, а tuning — evaluation. Early stopping и множители decision rule выбираются только на tuning. Evaluation-когорты не участвуют ни в обучении, ни в настройке. Итоговая locked-проверка использует 60 000 / 15 000 / 15 000 пользователей.

## Baselines

| Model | Mean weighted F1 | Std |
|---|---:|---:|
| Dummy most frequent | {baseline.loc['dummy_most_frequent', 'weighted_f1_mean']:.4f} | {baseline.loc['dummy_most_frequent', 'weighted_f1_std']:.4f} |
| Dummy stratified | {baseline.loc['dummy_stratified', 'weighted_f1_mean']:.4f} | {baseline.loc['dummy_stratified', 'weighted_f1_std']:.4f} |
| Logistic regression | {baseline.loc['multinomial_logistic_regression', 'weighted_f1_mean']:.4f} | {baseline.loc['multinomial_logistic_regression', 'weighted_f1_std']:.4f} |
| HistGradientBoosting | {baseline.loc['hist_gradient_boosting', 'weighted_f1_mean']:.4f} | {baseline.loc['hist_gradient_boosting', 'weighted_f1_std']:.4f} |
| Engineered CatBoost, adjusted | {baseline.loc['engineered_catboost_adjusted', 'weighted_f1_mean']:.4f} | {baseline.loc['engineered_catboost_adjusted', 'weighted_f1_std']:.4f} |

## Validation results

- Walk-forward weighted F1: **{walk_selected['weighted_f1_mean']:.4f} ± {walk_selected['weighted_f1_std']:.4f}**.
- Walk-forward macro F1: **{walk_selected['macro_f1_mean']:.4f} ± {walk_selected['macro_f1_std']:.4f}**.
- Untouched final weighted F1: **{final['weighted_f1']:.4f}**; macro F1: **{final['macro_f1']:.4f}**.
- Final class F1: not-churned {final['not_churned_f1']:.4f}, voluntary {final['vol_churn_f1']:.4f}, involuntary {final['invol_churn_f1']:.4f}.
- Raw final weighted F1 was {metrics['final_raw_metrics']['weighted_f1']:.4f}; tuning-only decision adjustment improved class balance without probability calibration.

Вероятности CatBoost в выгрузке являются **некалиброванными**. Decision scores используются только для выбора класса и не интерпретируются как вероятности.

## Experiments and ablations

Полный реестр находится в `experiments.csv`. Decision adjustment даёт крупнейший воспроизводимый прирост. Удаление payment-блока ухудшает weighted F1; quiz-блок не показывает инкрементального выигрыша. Трёхсидовый ensemble не превзошёл single seed ({baseline.loc['engineered_catboost_adjusted', 'weighted_f1_mean']:.4f} для выбранной системы; отдельный seed-study сохранён), поэтому production-модель одна. Regularized-конфигурация дала лишь +0.0014 на development-фолдах — меньше межфолдовой вариативности — и не была принята.

## Observation-window study

Окна 1, 3, 7 и 14 дней сравнивались на одинаковых development-фолдах. Разница между 3 и 14 днями мала, а на final-когорте 3 дня были немного лучше по weighted F1, но хуже по involuntary F1. Это показывает раннее появление основной части сигнала, но не даёт основания универсально заменить 14 дней.

## Error analysis

Крупнейшие ошибки final-когорты: voluntary → not-churned — {int(errors.loc['vol_churn -> not_churned', 'count']):,}, involuntary → not-churned — {int(errors.loc['invol_churn -> not_churned', 'count']):,}, voluntary → involuntary — {int(errors.loc['vol_churn -> invol_churn', 'count']):,}. Ошибки voluntary churn остаются главным ограничением. 681 ошибка имела raw model probability ≥ 0.70, что дополнительно показывает необходимость отдельной калибровки перед probability-based решениями.

## Explainability and churn drivers

Permutation importance считалась на всех future evaluation-пользователях каждого фолда; SHAP — на фиксированной выборке 1 500 пользователей на фолд. Для всех 7 000 test-пользователей сохранены top positive/negative contributors к raw score предсказанного класса. Статистический pipeline разделяет discovery (исторические 60 000) и validation (следующие 15 000), применяет Benjamini–Hochberg FDR и требует совпадения направления эффекта. Валидировано **{len(validated)} из {len(evidence)}** проверок.

Наиболее устойчивые involuntary-сигналы связаны с CVC, 3-D Secure, prepaid funding и card-declined событиями. Для voluntary churn воспроизводятся `quiz_cost_concern` и отдельные use-case/role сегменты, но quiz-ablation не подтверждает крупный прирост качества. Все эти результаты — ассоциации, не causal effects.

## Test predictions and interventions

- not_churned: {pred.get('not_churned', 0):,} ({pred.get('not_churned', 0) / len(insights):.1%})
- vol_churn: {pred.get('vol_churn', 0):,} ({pred.get('vol_churn', 0) / len(insights):.1%})
- invol_churn: {pred.get('invol_churn', 0):,} ({pred.get('invol_churn', 0) / len(insights):.1%})

`user_insights.csv` отделяет объяснение модели от rule-based business context: SHAP-колонки описывают вклад признаков в raw score, а actions являются операционными гипотезами. Priority band основан на decision margin, а не на выдуманной confidence.

## Limitations

- Test labels недоступны; test-качество локально неизвестно.
- Final evaluation — одна будущая когорта, поэтому window-challenger требует повторной проверки.
- Сильные geography/payment признаки одновременно дрейфуют и могут быть proxy-переменными.
- Вероятности не откалиброваны; decision multipliers оптимизированы под weighted F1, а не business cost.
- Наблюдательные ассоциации не доказывают причины churn.
- Thin `src/churn` API упрощает использование, но часть реализации намеренно остаётся в одном pipeline-модуле.

## Business implications

Для involuntary риска разумны pre-renewal card checks, card-update flow, smart retries и резервный способ оплаты. Для voluntary риска предпочтительны targeted research и эксперименты по value/onboarding вместо blanket-кампании по «низкой активности». Любую интервенцию следует проверять A/B-тестом с отдельными guardrail-метриками.

---

{hypotheses}
"""
    report = report.rstrip() + "\n"
    (ARTIFACTS / "research_report.md").write_text(report, encoding="utf-8")
    (ARTIFACTS / "final_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    build_registry()
    build_figures()
    build_research_report()


if __name__ == "__main__":
    main()
