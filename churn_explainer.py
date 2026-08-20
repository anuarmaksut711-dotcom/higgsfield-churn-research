"""Evidence-backed churn analysis and per-user interventions.

This module consumes artifacts produced by ``churn_pipeline.py``. It does not
claim causal effects: reported signals are validated associations that may be
useful for prediction and operational hypothesis generation.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from churn_pipeline import (
    ARTIFACT_DIR,
    CLASSES,
    ID_COL,
    TARGET,
    assemble_dataset,
    load_tables,
)
import matplotlib.pyplot as plt


LOG = logging.getLogger("churn_explainer")

NUMERIC_DRIVERS = {
    "invol_churn": [
        "attempt_failure_rate",
        "attempt_failure_count",
        "failure_insufficient_funds_count",
        "failure_card_declined_count",
        "failure_incorrect_cvc_count",
        "failure_expired_card_count",
        "is_3d_secure_authenticated_rate",
        "attempt_success_rate",
    ],
    "vol_churn": [
        "quiz_cost_concern",
        "quiz_missing_answers",
        "attempt_count",
        "attempt_active_days",
        "purchase_count",
        "purchase_amount_sum",
        "activity_days_total",
    ],
}

CATEGORICAL_DRIVERS = {
    "invol_churn": [
        "attempt_last_failure_code",
        "attempt_last_cvc_check",
        "attempt_last_card_funding",
        "attempt_last_card_3d_secure_support",
    ],
    "vol_churn": [
        "quiz_frustration",
        "quiz_usage_plan",
        "quiz_experience",
        "quiz_role",
        "subscription_plan",
    ],
}


def value(row: pd.Series, name: str, default: float = 0.0) -> float:
    raw = row.get(name, default)
    try:
        return float(raw) if pd.notna(raw) else default
    except (TypeError, ValueError):
        return default


def add_reason(
    reasons: list[str], actions: list[str], reason: str, action: str
) -> None:
    if reason not in reasons:
        reasons.append(reason)
        actions.append(action)


def explain_user(row: pd.Series) -> tuple[list[str], list[str]]:
    churn_type = str(row[TARGET])
    reasons: list[str] = []
    actions: list[str] = []
    if churn_type == "invol_churn":
        if value(row, "failure_insufficient_funds_count") > 0:
            add_reason(
                reasons,
                actions,
                "Недостаточно средств при попытке оплаты",
                "Повторить списание после payday-окна и заранее уведомить пользователя",
            )
        if value(row, "failure_card_declined_count") > 0:
            add_reason(
                reasons,
                actions,
                "Карта отклонялась банком",
                "Запросить резервный способ оплаты и применить smart retry",
            )
        if value(row, "failure_incorrect_cvc_count") + value(row, "failure_expired_card_count") > 0:
            add_reason(
                reasons,
                actions,
                "Некорректные или устаревшие реквизиты карты",
                "Показать безопасный card-update flow до продления",
            )
        if value(row, "attempt_failure_rate") >= 0.5:
            add_reason(
                reasons,
                actions,
                "Высокая доля неуспешных транзакций",
                "Передать пользователя в payment-recovery сценарий",
            )
        if str(row.get("attempt_last_card_funding", "")).lower() == "prepaid":
            add_reason(
                reasons,
                actions,
                "Используется prepaid-карта с повышенным риском отказа",
                "Предложить резервную банковскую карту или цифровой кошелёк",
            )
        if str(row.get("attempt_last_cvc_check", "")).lower() == "unavailable":
            add_reason(
                reasons,
                actions,
                "Проверка CVC недоступна",
                "Запросить обновление реквизитов карты до продления",
            )
        if str(row.get("attempt_last_card_3d_secure_support", "")).lower() in {
            "required",
            "not_supported",
        }:
            add_reason(
                reasons,
                actions,
                "Возможна дополнительная 3-D Secure аутентификация",
                "Предупредить о подтверждении платежа и предложить альтернативный метод",
            )
    elif churn_type == "vol_churn":
        if value(row, "quiz_cost_concern") > 0:
            add_reason(
                reasons,
                actions,
                "В онбординге отмечена высокая стоимость",
                "Предложить подходящий тариф или ограниченную retention-скидку",
            )
        role = str(row.get("quiz_role", "unknown")).lower()
        if role in {"developer", "prompt-engineer", "educator"}:
            add_reason(
                reasons,
                actions,
                "Профессиональный сегмент с повышенным оттоком в train",
                "Собрать обратную связь по качеству, контролю и недостающим функциям",
            )
        if value(row, "attempt_count") >= 2 or value(row, "purchase_count") > 1:
            add_reason(
                reasons,
                actions,
                "Высокий интерес не конвертируется в удержание",
                "Предложить сценарий ценности по фактически используемым функциям",
            )
        if value(row, "quiz_missing_answers") >= 5:
            add_reason(
                reasons,
                actions,
                "Профиль потребности неполно определён в онбординге",
                "Уточнить основной use case и адаптировать onboarding",
            )
    else:
        if value(row, "attempt_failure_rate") == 0:
            add_reason(
                reasons,
                actions,
                "Платежи проходят стабильно",
                "Поддерживать лояльность без скидки",
            )
        if value(row, "activity_days_total") >= 3:
            add_reason(
                reasons,
                actions,
                "Регулярная активность в первые 14 дней",
                "Рекомендовать следующий релевантный use case",
            )
        if value(row, "purchase_count") > 1:
            add_reason(
                reasons,
                actions,
                "Есть повторные покупки",
                "Предложить годовой план или пакет кредитов",
            )

    fallback = {
        "invol_churn": (
            "Комбинация платёжных сигналов модели",
            "Проверить способ оплаты до следующего списания",
        ),
        "vol_churn": (
            "Комбинация сигналов низкой вовлечённости",
            "Отправить персональный retention-сценарий",
        ),
        "not_churned": (
            "Стабильный профиль удержания",
            "Продолжить стандартный lifecycle-маркетинг",
        ),
    }[churn_type]
    while len(reasons) < 3:
        add_reason(reasons, actions, *fallback)
        if len(reasons) < 3:
            add_reason(
                reasons,
                actions,
                "Дополнительный контекст требует ручной проверки",
                "Использовать decision margin только для ранжирования очереди",
            )
        if len(reasons) < 3:
            reasons.append("Дополнительных критичных сигналов не обнаружено")
            actions.append("Не применять агрессивное вмешательство")
    return reasons[:3], actions[:3]


def build_user_insights(test_data: pd.DataFrame) -> pd.DataFrame:
    predictions_path = ARTIFACT_DIR / "submission_with_probabilities.csv"
    if not predictions_path.exists():
        raise FileNotFoundError("Run churn_pipeline.py before churn_explainer.py")
    predictions = pd.read_csv(predictions_path)
    data = predictions.merge(test_data, on=ID_COL, how="left", validate="one_to_one")
    probability_cols = [
        col for col in data.columns if col.startswith("model_probability_")
    ]
    decision_cols = [col for col in data.columns if col.startswith("decision_score_")]
    ordered_scores = np.sort(data[decision_cols].to_numpy(dtype=float), axis=1)
    data["decision_margin"] = ordered_scores[:, -1] - ordered_scores[:, -2]
    data["priority_band"] = pd.cut(
        data["decision_margin"],
        bins=[-np.inf, 0.10, 0.25, np.inf],
        labels=["review", "medium", "high"],
    ).astype(str)

    output_rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        reasons, actions = explain_user(row)
        record: dict[str, object] = {
            ID_COL: row[ID_COL],
            "predicted_churn": row[TARGET],
            "priority_band": row["priority_band"],
            "decision_margin": row["decision_margin"],
            "business_context_layer": "rule-based contextual signals; not model attribution",
        }
        for col in probability_cols + decision_cols:
            record[col] = row[col]
        for index in range(3):
            record[f"contextual_signal_{index + 1}"] = reasons[index]
            record[f"recommended_action_{index + 1}"] = actions[index]
        output_rows.append(record)
    output = pd.DataFrame(output_rows)
    shap_path = ARTIFACT_DIR / "test_model_explanations.csv"
    if shap_path.exists():
        shap = pd.read_csv(shap_path)
        output = output.merge(shap, on=ID_COL, how="left", validate="one_to_one")
    output.to_csv(ARTIFACT_DIR / "user_insights.csv", index=False)
    return output


CLASS_LABELS = {
    "not_churned": "Остались",
    "vol_churn": "Добровольный отток",
    "invol_churn": "Недобровольный отток",
}
CLASS_COLORS = {
    "not_churned": "#22A699",
    "vol_churn": "#F2BE22",
    "invol_churn": "#E45756",
}


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(ARTIFACT_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_performance() -> None:
    report = pd.read_csv(ARTIFACT_DIR / "classification_report.csv", index_col=0)
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    values = report.loc[CLASSES, ["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(CLASSES))
    width = 0.24
    palette = ["#4C78A8", "#72B7B2", "#F58518"]
    for index, column in enumerate(values.columns):
        bars = ax.bar(x + (index - 1) * width, values[column], width, label=column, color=palette[index])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(x, [CLASS_LABELS[item] for item in CLASSES])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Значение метрики")
    ax.set_title(
        "Качество на untouched final temporal evaluation\n"
        f"Weighted F1 = {metrics['final_adjusted_metrics']['weighted_f1']:.4f}, "
        f"n = {metrics['final_fold']['evaluation_rows']:,}"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    save_figure(fig, "model_performance.png")


def plot_class_distributions(train_data: pd.DataFrame, insights: pd.DataFrame) -> None:
    validation = pd.read_csv(ARTIFACT_DIR / "validation_predictions.csv")
    groups = {
        "Train: факт": train_data[TARGET].value_counts(normalize=True),
        "OOT: факт": validation["true"].value_counts(normalize=True),
        "Test: прогноз": insights["predicted_churn"].value_counts(normalize=True),
    }
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for class_name in CLASSES:
        values = np.array([groups[name].get(class_name, 0) for name in groups])
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=CLASS_LABELS[class_name],
            color=CLASS_COLORS[class_name],
        )
        for index, value_ in enumerate(values):
            if value_ >= 0.06:
                ax.text(index, bottom[index] + value_ / 2, f"{value_:.1%}", ha="center", va="center")
        bottom += values
    ax.set_xticks(x, list(groups))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Доля пользователей")
    ax.set_title("Распределение классов: факт и прогноз")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save_figure(fig, "class_distributions.png")


def plot_feature_importance() -> None:
    importance = (
        pd.read_csv(ARTIFACT_DIR / "permutation_importance_by_fold.csv")
        .groupby("feature")["weighted_f1_drop"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 7.2))
    bars = ax.barh(importance["feature"], importance["weighted_f1_drop"], color="#4C78A8")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Среднее падение weighted F1 после permutation")
    ax.set_title("15 признаков с наибольшей temporal permutation importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "feature_importance.png")


def plot_churn_drivers(evidence: pd.DataFrame, churn_type: str, file_name: str) -> None:
    subset = evidence.loc[
        evidence["churn_type"].eq(churn_type)
        & evidence["evidence_type"].eq("categorical_discovered_then_validated")
        & evidence["validated"]
    ].copy()
    subset = subset.sort_values(
        ["validation_standardized_effect", "validation_support"], ascending=False
    ).head(8)
    subset["label"] = subset["feature"] + " = " + subset["segment"].astype(str)
    subset = subset.sort_values("validation_standardized_effect")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.barh(
        subset["label"],
        subset["validation_standardized_effect"],
        color=CLASS_COLORS[churn_type],
    )
    labels = [f"{lift:.2f}× · n={support:,}" for lift, support in zip(
        subset["validation_standardized_effect"], subset["validation_support"], strict=True
    )]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=8)
    ax.axvline(1.0, color="#555555", linewidth=1)
    ax.set_xlabel("Lift вероятности класса относительно базовой доли")
    ax.set_title(f"Драйверы: {CLASS_LABELS[churn_type]} (FDR q < 0,05)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, file_name)


def plot_decision_margin(insights: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bins = np.linspace(0, max(0.01, insights["decision_margin"].max()), 21)
    for class_name in CLASSES:
        values = insights.loc[
            insights["predicted_churn"].eq(class_name), "decision_margin"
        ]
        ax.hist(
            values,
            bins=bins,
            alpha=0.55,
            label=CLASS_LABELS[class_name],
            color=CLASS_COLORS[class_name],
        )
    ax.set_xlabel("Разница между двумя максимальными decision scores")
    ax.set_ylabel("Количество пользователей")
    ax.set_title("Decision margin test-предсказаний (не вероятность)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "decision_margin.png")


def generate_visuals(
    train_data: pd.DataFrame,
    evidence: pd.DataFrame,
    insights: pd.DataFrame,
) -> None:
    plot_model_performance()
    plot_class_distributions(train_data, insights)
    plot_feature_importance()
    plot_churn_drivers(evidence, "invol_churn", "invol_churn_drivers.png")
    plot_churn_drivers(evidence, "vol_churn", "vol_churn_drivers.png")
    plot_decision_margin(insights)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_data = assemble_dataset(load_tables("train"))
    test_data = assemble_dataset(load_tables("test"))
    evidence_path = ARTIFACT_DIR / "churn_driver_evidence.csv"
    if not evidence_path.exists():
        raise FileNotFoundError("Run association_analysis.py before churn_explainer.py")
    evidence = pd.read_csv(evidence_path)
    insights = build_user_insights(test_data)
    generate_visuals(train_data, evidence, insights)
    metrics_path = ARTIFACT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    LOG.info("Churn-association evidence rows: %d", len(evidence))
    LOG.info("User insights rows: %d", len(insights))
    LOG.info(
        "Final temporal weighted F1: %s",
        metrics.get("final_adjusted_metrics", {}).get("weighted_f1"),
    )


if __name__ == "__main__":
    main()
