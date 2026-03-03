import pandas as pd
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.contingency_tables import mcnemar

from src.environment.environment import GridWorldEnv
from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper

PROMPT_STRATEGIES = {
    # Стратегия A: Контрольная (Твой текущний base)
    "base_random": {
        "system": (
            "You are a strategic exploration AI. "
            "Your task is to choose a coordinate (x, y) to explore next. "
            "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
            "Prioritize unexplored areas."
        ),
        "description": "Базовая случайная стратегия (Control)"
    },

    # Стратегия B: Систематическая (Спираль/Змейка)
    # Гипотеза: Упорядоченное движение покроет карту быстрее, чем хаотичное.
    "systematic_sweep": {
        "system": (
            "You are a methodical exploration AI. "
            "Your goal is to cover the grid efficiently. "
            "Use a 'zig-zag' or 'spiral' search pattern. "
            "Move along rows or columns systematically. "
            "Avoid returning to visited coordinates unless necessary. "
            "Return ONLY a JSON object with 'target_coordinate' as [x, y]."
        ),
        "description": "Систематический поиск (Sweep)"
    },

    # Стратегия C: Жадный фронт (Ближайшая граница)
    # Гипотеза: Идти к ближайшей неизвестной точке эффективнее по шагам.
    "greedy_frontier": {
        "system": (
            "You are an efficient exploration AI. "
            "Identify the 'frontier' — the border between explored and unexplored areas. "
            "Select the closest unexplored coordinate that is adjacent to a known safe zone. "
            "Minimize travel distance. "
            "Return ONLY a JSON object with 'target_coordinate' as [x, y]."
        ),
        "description": "Жадный поиск границы (Frontier)"
    },

    # Стратегия D: Цепочка рассуждений (Chain of Thought)
    # Гипотеза: Если модель подумает перед ответом, она реже будет ходить в бомбы.
    "reasoning_cot": {
        "system": (
            "You are a careful exploration AI. "
            "Analyze the map before moving. "
            "Think step-by-step: "
            "1. Where are the dangers? "
            "2. Where are the unexplored areas? "
            "3. Which unexplored area is safest to approach? "
            "Return a JSON object with a 'reasoning' string and 'target_coordinate' as [x, y]."
        ),
        "description": "Стратегия с рассуждением (CoT)"
    }
}

def make_env(strategy):
    env = GridWorldEnv(size=5, num_bombs=3)
    env = HighLevelPlannerWrapper(env, strategy)
    return env

def run_episode(strategy_name, seed):
    strategy = PROMPT_STRATEGIES[strategy_name]['system']

    env = make_env(strategy)
    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    episode_rewards = []
    total_reward = 0
    total_steps = 0
    success = False
    bombed = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(None)

        episode_rewards.append(reward)
        total_reward += reward
        total_steps += 1

    input_tokens, output_tokens = env.planner.total_input_tokens, env.planner.total_output_tokens
    exploration_efficiency = len([coord for coord, val in env.known_world.items() if val in ['VISITED', 'TARGET']])
    if terminated and episode_rewards[-1] == -1:
        bombed = True
    elif terminated and episode_rewards[-1] == 1:
        success = True

    env.close()
    return {
        "strategy": strategy_name,
        "seed": seed,
        "total_reward": total_reward,
        "total_steps": total_steps,
        "success": success,
        "bombed": bombed,
        "truncated": truncated,
        "llm_calls": env.llm_calls,
        'exploration_efficiency': exploration_efficiency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

# ПИЛОТНЫЙ ТЕСТ (Сбор данных для всех стратегий)
def run_pilot_test(pilot_n=20):
    print(f"--- Запуск пилотного теста: {pilot_n} эпизодов ---")
    pilot_results = []

    for seed in range(pilot_n):
        episode_data = {'seed': seed}
        # Прогоняем все стратегии на одном seed
        for strategy_name in PROMPT_STRATEGIES.keys():
            metrics = run_episode(strategy_name, seed)
            episode_data[f"{strategy_name}_success"] = int(metrics['success'])
            episode_data[f"{strategy_name}_steps"] = metrics['total_steps']
            episode_data[f"{strategy_name}_input_tokens"] = metrics['input_tokens']
            episode_data[f"{strategy_name}_output_tokens"] = metrics['output_tokens']

        pilot_results.append(episode_data)

    df_pilot = pd.DataFrame(pilot_results)
    print("Пилот завершен.\n")
    return df_pilot


# POWER ANALYSIS (Расчет максимального N)
def calculate_required_sample_size(df_pilot, alpha=0.05, power=0.8):
    print("--- Расчет необходимого количества эпизодов ---")

    strategies = list(PROMPT_STRATEGIES.keys())
    strat_control = strategies[0]
    max_n_required = 0

    # Проходимся по всем тестовым стратегиям, чтобы найти максимальный требуемый N
    for strat_treat in strategies[1:]:
        col_a = f"{strat_control}_success"
        col_b = f"{strat_treat}_success"

        p1 = df_pilot[col_a].mean()
        p2 = df_pilot[col_b].mean()

        # Если разницы нет или она очень мала, используем MDE = 5% (0.05)
        if abs(p2 - p1) < 0.05:
            # Предполагаем, что мы хотим найти разницу хотя бы в 5%
            p2_calc = min(p1 + 0.05, 0.99)
        else:
            p2_calc = p2

        # Effect size должен быть положительным для solve_power
        effect_size = abs(proportion_effectsize(p2_calc, p1))

        analysis = NormalIndPower()
        try:
            # Считаем N для независимого теста (консервативно)
            n_indiv = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)

            # Учитываем корреляцию (парный тест мощнее)
            corr = df_pilot[[col_a, col_b]].corr().iloc[0, 1]
            if corr > 0.1:
                n_adj = n_indiv * (1 - corr)
            else:
                n_adj = n_indiv

            if n_adj > max_n_required:
                max_n_required = n_adj

        except Exception as e:
            print(f"Ошибка расчета для {strat_treat}: {e}. Берем запасной вариант.")
            max_n_required = max(max_n_required, 150)  # Fallback

    # Буфер 15%
    final_n = int(np.ceil(max_n_required * 1.15))
    print(f"Максимальный требуемый N (с учетом буфера): {final_n}")
    return final_n


# ОСНОВНОЙ ТЕСТ
def run_full_test(final_n):
    print(f"--- Запуск основного теста: {final_n} эпизодов ---")
    full_results = []

    for seed in range(final_n):
        if seed % 20 == 0:
            print(f"Episode {seed}/{final_n}")

        episode_data = {'seed': seed}
        for strategy_name in PROMPT_STRATEGIES.keys():
            metrics = run_episode(strategy_name, seed)
            episode_data[f"{strategy_name}_success"] = int(metrics['success'])
            episode_data[f"{strategy_name}_steps"] = metrics['total_steps']
            episode_data[f"{strategy_name}_input_tokens"] = metrics['input_tokens']
            episode_data[f"{strategy_name}_output_tokens"] = metrics['output_tokens']

        full_results.append(episode_data)

    return pd.DataFrame(full_results)

# АНАЛИЗ И ТАБЛИЦА ЛИДЕРОВ
def analyze_and_rank_results(df):
    strategies = list(PROMPT_STRATEGIES.keys())
    strat_control = strategies[0]

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЕТ: СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 60)

    # --- 1. ТАБЛИЦА РЕЗУЛЬТАТОВ (LEADERBOARD) ---
    leaderboard_data = []
    for strat in strategies:
        success_rate = df[f"{strat}_success"].mean()
        avg_steps = df[f"{strat}_steps"].mean()
        avg_tokens = df[f"{strat}_output_tokens"].mean()

        leaderboard_data.append({
            "Strategy": strat,
            "Success Rate": f"{success_rate:.2%}",
            "Avg Steps": f"{avg_steps:.1f}",
            "Avg Tokens": f"{avg_tokens:.1f}",
            "Raw_Success": success_rate  # для сортировки
        })

    df_leaderboard = pd.DataFrame(leaderboard_data)
    df_leaderboard = df_leaderboard.sort_values(by="Raw_Success", ascending=False)
    df_leaderboard = df_leaderboard.drop(columns=["Raw_Success"])

    print("\n[ТАБЛИЦА ЛИДЕРОВ]")
    print(df_leaderboard.to_string(index=False))
    df_leaderboard.to_csv("promts_leaderboard.csv", index=False)

    # --- 2. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ (Base vs Others) ---
    print(f"\n[ПОПАРНЫЕ СРАВНЕНИЯ С БАЗОЙ ({strat_control})]")
    print(f"Используем McNemar's Test. Значимость: p < 0.05")

    bonferroni_alpha = 0.05 / (len(strategies) - 1)  # Поправка на множественные тесты

    for strat_treat in strategies[1:]:
        print(f"\n>>> {strat_control} vs {strat_treat} <<<")

        col_a = f"{strat_control}_success"
        col_b = f"{strat_treat}_success"

        # McNemar Test
        n01 = sum((df[col_a] == 0) & (df[col_b] == 1))  # Treat лучше
        n10 = sum((df[col_a] == 1) & (df[col_b] == 0))  # Control лучше

        n11 = sum((df[col_a] == 1) & (df[col_b] == 1))
        n00 = sum((df[col_a] == 0) & (df[col_b] == 0))

        table = [[n11, n10], [n01, n00]]
        result = mcnemar(table, exact=True)

        print(f"Success Diff: {df[col_b].mean():.2%} vs {df[col_a].mean():.2%}")
        print(f"McNemar p-value: {result.pvalue:.5f} (порог с поправкой: {bonferroni_alpha:.3f})")

        if result.pvalue < bonferroni_alpha:
            winner = strat_treat if n01 > n10 else strat_control
            print(f"✅ РАЗНИЦА ЗНАЧИМА. Победитель: {winner}")
        else:
            print(f"❌ Разница статистически незначима.")

        # Вторичные метрики (Steps, Tokens)
        for metric in ['steps', 'input_tokens', 'output_tokens']:
            col_a_m = f"{strat_control}_{metric}"
            col_b_m = f"{strat_treat}_{metric}"
            diff = df[col_b_m].mean() - df[col_a_m].mean()
            print(f"   {metric} diff: {diff:.2f}")


# MAIN EXECUTION FLOW
if __name__ == '__main__':
    # 1. Пилот
    df_pilot = run_pilot_test(pilot_n=20)

    # 2. Расчет N
    required_n = calculate_required_sample_size(df_pilot, alpha=0.05, power=0.8)

    # 3. Полный запуск
    df_final = run_full_test(required_n)

    # 4. Анализ и Рейтинг
    analyze_and_rank_results(df_final)

    # 5. Сохранение
    df_final.to_csv("final_multi_strategy_test.csv", index=False)
    print("\nРезультаты сохранены в final_multi_strategy_test.csv")