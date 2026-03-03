import pandas as pd
from src.environment.environment import GridWorldEnv
from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper

# 1. Стратегии промптинга для High-Level Planner
PROMPT_STRATEGIES = {
    "v2_base": {
        "system": (
            "You are a strategic exploration AI. "
            "Your task is to choose a coordinate (x, y) to explore next. "
            "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
            "Prioritize unexplored areas. "
            "Avoid coordinates marked as BOMB or WALL. "
            "If TARGET is known, return its coordinate."
        ),
        "description": "Базовая стратегия для выбора подцели"
    },
    "base": {
        "system": (
            "You are a strategic exploration AI. "
            "Your task is to choose a coordinate (x, y) to explore next. "
            "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
            "Prioritize unexplored areas."
        ),
        "description": "супер базовая"
    }
}

N_EPISODES = 100

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

def run_ab_test():
    results = []
    for strategy_name in PROMPT_STRATEGIES:
        for episode in range(N_EPISODES):
            print(episode)
            metrics = run_episode(strategy_name, episode)
            results.append(metrics)

    df = pd.DataFrame(results)

    # --- Считаем агрегированные метрики ---
    summary = df.groupby("strategy").agg(
        success_rate=("success", "mean"),  # % успеха
        bomb_hit_rate=("bombed", "mean"),  # % взрывов
        avg_reward=("total_reward", "mean"),
        avg_steps=("total_steps", "mean"),
        avg_llm_calls=("llm_calls", "mean"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_exploration_efficiency=("exploration_efficiency", "mean")
    ).reset_index()

    # Можно сразу преобразовать в проценты для rate
    summary["success_rate"] *= 100
    summary["bomb_hit_rate"] *= 100

    return df, summary

if __name__ == '__main__':
    df, summary = run_ab_test()
    df.to_csv("ab_test_episodes.csv", index=False)
    summary.to_csv("ab_test_summary.csv", index=False)
