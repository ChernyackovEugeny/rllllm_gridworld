import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.contingency_tables import mcnemar

from src.llm_high_level_planning.testing.promt_AB_testing import run_episode
from src.llm_high_level_planning.testing.promt_AB_testing import PROMPT_STRATEGIES

# ПИЛОТНЫЙ ТЕСТ (N = 20)
def run_pilot_test(pilot_n=20):
    print(f"--- Запуск пилотного теста: {pilot_n} эпизодов ---")
    pilot_results = []

    # Используем range(pilot_n) как seed
    for seed in range(pilot_n):
        # Словарь для результатов одного эпизода (пара стратегий)
        episode_data = {'seed': seed}

        for strategy_name in PROMPT_STRATEGIES.keys():
            metrics = run_episode(strategy_name, seed)

            # Сохраняем метрики с префиксом стратегии
            # Нас интересует 'success' как primary metric
            episode_data[f"{strategy_name}_success"] = int(metrics['success'])
            episode_data[f"{strategy_name}_steps"] = metrics['total_steps']

        pilot_results.append(episode_data)

    df_pilot = pd.DataFrame(pilot_results)
    print("Пилот завершен.\n")
    return df_pilot


# POWER ANALYSIS (Расчет N)
def calculate_required_sample_size(df_pilot, alpha=0.05, power=0.8):
    print("--- Расчет необходимого количества эпизодов (Sample Size) ---")

    # Определяем названия стратегий (предполагаем, что их две: Base и Treatment)
    strategies = list(PROMPT_STRATEGIES.keys())
    strat_a = strategies[0]
    strat_b = strategies[1]

    col_a = f"{strat_a}_success"
    col_b = f"{strat_b}_success"

    # 1. Оцениваем базовые конверсии
    p1 = df_pilot[col_a].mean()
    p2 = df_pilot[col_b].mean()

    print(f"Pilot Success Rate {strat_a}: {p1:.2%}")
    print(f"Pilot Success Rate {strat_b}: {p2:.2%}")
    print(f"Observed Difference: {p2 - p1:.2%}")

    # 2. Считаем Effect Size (Cohen's h для пропорций)
    # Это стандартизированная разница между пропорциями
    effect_size = proportion_effectsize(p2, p1)

    # 3. Создаем объект для анализа мощности
    analysis = NormalIndPower()

    # 4. Считаем N для НЕЗАВИСИМОГО теста (консервативная оценка)
    # ratio=1.0 значит группы равны
    n_indiv = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)

    # 5. Корректировка на ПАРНЫЙ тест (Paired Test)
    # Парный тест мощнее. Насколько мощнее - зависит от корреляции.
    # Если на сложных картах обе стратегии проигрывают, а на легких выигрывают, корреляция высокая.
    corr = df_pilot[[col_a, col_b]].corr().iloc[0, 1]

    print(f"Корреляция результатов между стратегиями: {corr:.3f}")

    # Формула поправки: N_paired ≈ N_independent * (1 - r)
    # Где r - корреляция. Чем выше корреляция, тем меньше нужно семплов.
    # Однако для пропорций формула сложнее, используем приближение через дисперсию.
    # Простой способ: Variance of difference = Var(A) + Var(B) - 2*Cov(A,B)
    # Variance reduction factor = 1 - corr (грубо)

    required_n = int(np.ceil(n_indiv))

    # Добавим буфер 10-15% на всякий случай (выбросы, недетерминированность LLM)
    required_n_safe = int(required_n * 1.15)

    print(f"\nРезультаты Power Analysis:")
    print(f"Effect Size (Cohen's h): {effect_size:.3f}")
    print(f"Требуемый N (независимый тест, консервативно): {required_n}")
    print(f"Корреляция стратегий: {corr:.2f}")
    if corr > 0.1:
        print(
            f"Т.к. корреляция положительная, парный тест потребует меньше данных (около {int(required_n * (1 - corr))}).")
        required_n = int(required_n * (1 - corr)) * 1.15
        print(f"\n>>> ИТОГО ЗАПУСКАЕМ ЭПИЗОДОВ: {required_n} (с запасом 15%) <<<\n")
        return required_n

    print(f"\n>>> ИТОГО ЗАПУСКАЕМ ЭПИЗОДОВ: {required_n_safe} (с запасом 15%) <<<\n")
    return required_n_safe

# ОСНОВНОЙ ТЕСТ (Парный запуск)
def run_full_test(final_n):
    print(f"--- Запуск основного теста: {final_n} эпизодов ---")
    full_results = []

    for seed in range(final_n):
        # Прогресс бар для наглядности
        if seed % 50 == 0:
            print(f"Episode {seed}/{final_n}")

        episode_data = {'seed': seed}
        for strategy_name in PROMPT_STRATEGIES.keys():
            metrics = run_episode(strategy_name, seed)
            # Сохраняем всё, что нужно для анализа
            episode_data[f"{strategy_name}_success"] = int(metrics['success'])
            episode_data[f"{strategy_name}_steps"] = metrics['total_steps']
            episode_data[f"{strategy_name}_input_tokens"] = metrics['input_tokens']
            episode_data[f"{strategy_name}_output_tokens"] = metrics['output_tokens']

        full_results.append(episode_data)

    return pd.DataFrame(full_results)

# ФИНАЛЬНЫЙ АНАЛИЗ (Paired)
def analyze_final_results(df):
    strategies = list(PROMPT_STRATEGIES.keys())
    strat_a = strategies[0]
    strat_b = strategies[1]

    print("\n" + "=" * 30)
    print("ФИНАЛЬНЫЙ ОТЧЕТ")
    print("=" * 30)

    # 1. Первичная метрика: Success Rate (McNemar's Test)
    print("\n[Primary Metric: Success Rate]")

    col_a = f"{strat_a}_success"
    col_b = f"{strat_b}_success"

    rate_a = df[col_a].mean()
    rate_b = df[col_b].mean()

    print(f"{strat_a}: {rate_a:.2%}")
    print(f"{strat_b}: {rate_b:.2%}")
    print(f"Difference: {(rate_b - rate_a):.2%}")

    # McNemar Test для парных бинарных данных
    # Строим таблицу сопряженности
    #                  B=0      B=1
    # A=0        (n00)    (n01)
    # A=1        (n10)    (n11)

    # n01: A неудача, B успех (B лучше)
    # n10: A успех, B неудача (A лучше)
    # n00 и n11: разницы нет

    # McNemar тест проверяет, значимо ли отличие n01 от n10

    n01 = sum((df[col_a] == 0) & (df[col_b] == 1))
    n10 = sum((df[col_a] == 1) & (df[col_b] == 0))
    n11 = sum((df[col_a] == 1) & (df[col_b] == 1))
    n00 = sum((df[col_a] == 0) & (df[col_b] == 0))

    table = [[n11, n10],
             [n01, n00]]

    # exact=True использует биномиальное распределение (точнее для малых N)
    result = mcnemar(table, exact=True)

    print(f"Discordant pairs: B better={n01}, A better={n10}")
    print(f"McNemar's Test p-value: {result.pvalue:.5f}")

    if result.pvalue < 0.05:
        winner = strat_b if n01 > n10 else strat_a
        print(f"✅ РЕЗУЛЬТАТ СТАТИСТИЧЕСКИ ЗНАЧИМ. Победитель: {winner}")
    else:
        print("❌ Результат статистически незначим. Различия могут быть случайными.")

    # 2. Вторичные метрики (Steps, Tokens) - Парный t-тест
    print("\n[Secondary Metrics: Steps & Tokens]")

    for metric in ['steps', 'input_tokens', 'output_tokens']:
        col_a_m = f"{strat_a}_{metric}"
        col_b_m = f"{strat_b}_{metric}"

        # Парный t-тест
        t_stat, p_val = stats.ttest_rel(df[col_b_m], df[col_a_m])

        diff = df[col_b_m] - df[col_a_m]
        mean_diff = diff.mean()

        print(f"\nMetric: {metric}")
        print(f"Mean diff ({strat_b} - {strat_a}): {mean_diff:.2f}")
        print(f"P-value (paired t-test): {p_val:.5f}")

# MAIN EXECUTION FLOW
# 1. Пилот
df_pilot = run_pilot_test(pilot_n=20)

required_n = calculate_required_sample_size(df_pilot, alpha=0.05, power=0.8)

df_final = run_full_test(required_n)

analyze_final_results(df_final)

df_final.to_csv("final_ab_test_paired.csv", index=False)