# Higgsfield churn prediction — исследовательский отчёт

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
| Dummy most frequent | 0.3335 | 0.0094 |
| Dummy stratified | 0.3785 | 0.0008 |
| Logistic regression | 0.4779 | 0.0200 |
| HistGradientBoosting | 0.4823 | 0.0284 |
| Engineered CatBoost, adjusted | 0.5141 | 0.0173 |

## Validation results

- Walk-forward weighted F1: **0.5141 ± 0.0173**.
- Walk-forward macro F1: **0.4718 ± 0.0188**.
- Untouched final weighted F1: **0.5068**; macro F1: **0.4616**.
- Final class F1: not-churned 0.6395, voluntary 0.2803, involuntary 0.4649.
- Raw final weighted F1 was 0.4674; tuning-only decision adjustment improved class balance without probability calibration.

Вероятности CatBoost в выгрузке являются **некалиброванными**. Decision scores используются только для выбора класса и не интерпретируются как вероятности.

## Experiments and ablations

Полный реестр находится в `experiments.csv`. Decision adjustment даёт крупнейший воспроизводимый прирост. Удаление payment-блока ухудшает weighted F1; quiz-блок не показывает инкрементального выигрыша. Трёхсидовый ensemble не превзошёл single seed (0.5141 для выбранной системы; отдельный seed-study сохранён), поэтому production-модель одна. Regularized-конфигурация дала лишь +0.0014 на development-фолдах — меньше межфолдовой вариативности — и не была принята.

## Observation-window study

Окна 1, 3, 7 и 14 дней сравнивались на одинаковых development-фолдах. Разница между 3 и 14 днями мала, а на final-когорте 3 дня были немного лучше по weighted F1, но хуже по involuntary F1. Это показывает раннее появление основной части сигнала, но не даёт основания универсально заменить 14 дней.

## Error analysis

Крупнейшие ошибки final-когорты: voluntary → not-churned — 2,363, involuntary → not-churned — 1,401, voluntary → involuntary — 698. Ошибки voluntary churn остаются главным ограничением. 681 ошибка имела raw model probability ≥ 0.70, что дополнительно показывает необходимость отдельной калибровки перед probability-based решениями.

## Explainability and churn drivers

Permutation importance считалась на всех future evaluation-пользователях каждого фолда; SHAP — на фиксированной выборке 1 500 пользователей на фолд. Для всех 7 000 test-пользователей сохранены top positive/negative contributors к raw score предсказанного класса. Статистический pipeline разделяет discovery (исторические 60 000) и validation (следующие 15 000), применяет Benjamini–Hochberg FDR и требует совпадения направления эффекта. Валидировано **19 из 54** проверок.

Наиболее устойчивые involuntary-сигналы связаны с CVC, 3-D Secure, prepaid funding и card-declined событиями. Для voluntary churn воспроизводятся `quiz_cost_concern` и отдельные use-case/role сегменты, но quiz-ablation не подтверждает крупный прирост качества. Все эти результаты — ассоциации, не causal effects.

## Test predictions and interventions

- not_churned: 5,090 (72.7%)
- vol_churn: 962 (13.7%)
- invol_churn: 948 (13.5%)

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

# Проверка исследовательских гипотез

Все выводы ниже основаны на временных разбиениях. Гиперпараметры, early stopping и decision multipliers выбирались без доступа к evaluation-когорте.

## H1. Платёжные признаки особенно полезны для involuntary churn — частично подтверждена

Удаление платёжных признаков снизило средний weighted F1 с 0.4954 до 0.4875. При этом involuntary F1 изменился лишь с 0.4269 до 0.4258. Permutation importance и валидированные ассоциации выделяют card/bank/payment-поля, но ablation не доказывает, что прирост концентрируется именно в involuntary F1. Итог: платёжный блок полезен в целом; сильная формулировка гипотезы не подтверждена.

## H2. Quiz-признаки заметно улучшают voluntary churn — не подтверждена

Без quiz-признаков weighted F1 составил 0.4971 против 0.4954 у полной сырой модели, а voluntary F1 — 0.2109 против 0.1984. Отдельные quiz-сегменты статистически воспроизводятся, но заметного инкрементального модельного эффекта нет.

## H3. Большая часть сигнала возникает в первые 1–3 дня — поддержана с оговорками

На development-фолдах 3 дня дали 0.5080, а 14 дней — 0.5163. На единственной untouched final-когорте результат 3 дней был 0.5021 против 0.4997. Однако 14 дней лучше распознавали involuntary churn (0.4722 против 0.4548), поэтому 14-дневное окно оставлено как консервативный production default.

## H4. География и платёжная инфраструктура нестабильны во времени — подтверждена

Максимальный Jensen–Shannon divergence достиг 0.1611 для billing country в test и 0.1592 для 3-D Secure support. Одновременно country, card country, bank name и funding входят в верхнюю часть permutation importance. Это сочетание высокой важности и заметного cohort shift требует мониторинга drift; оно не является доказательством причинности.
