from src.environment.environment import GridWorldEnv
from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper
import pygame
import time

# --- Конфигурация сценариев ---
SCENARIOS = [
    {
        "name": "1. The Great Wall",
        "description": "LLM must find a way around the wall.",
        "agent": [0, 4],
        "target": [9, 4],
        "bombs": [
            [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9],
            [1, 1]
        ]
    },
    {
        "name": "2. The Corridor",
        "description": "Navigate narrow path.",
        "agent": [0, 5],
        "target": [9, 5],
        "bombs": [
            [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3],
            [2, 7], [3, 7], [4, 7], [5, 7]
        ]
    },
    {
        "name": "3. Diagonal Trap",
        "description": "Avoid diagonal bombs.",
        "agent": [0, 0],
        "target": [9, 9],
        "bombs": [
            [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8],
            [3, 2], [6, 5]
        ]
    }
]

GRID_SIZE = 10
NUM_BOMBS = 10


def wait_for_key(env):
    print("\nPress Space/Enter in Pygame window...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                    waiting = False
                elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    return False
        time.sleep(0.05)
    return True


def run_inference(env, scenario):
    print(f"\n{'=' * 60}\nScenario: {scenario['name']}\n{'=' * 60}")

    # 1. Сброс среды
    env.reset(seed=42)

    # 2. Устанавливаем состояние вручную
    env.unwrapped._set_custom_state(
        agent_loc=scenario['agent'],
        target_loc=scenario['target'],
        bombs_locs=scenario['bombs']
    )

    # 3. ВАЖНО: Сброс и обновление памяти обертки
    # Так как мы изменили состояние среды "через голову" обертки,
    # нужно вручную обновить её память, чтобы LLM увидел стартовые условия.
    env.known_world = {}
    env._update_memory()  # Сканируем окружение старта
    env.current_goal = None
    env.current_path = []

    done = False
    total_reward = 0
    step = 0

    # Получаем актуальное наблюдение после сетапа
    obs = env.unwrapped._get_obs()

    while not done:
        env.render()

        print(f"\nStep {step} | Pos: {tuple(env.unwrapped._agent_location)}")
        if env.current_goal:
            print(f"LLM Plan: Goal {env.current_goal}, Steps left in cache: {len(env.current_path)}")

        if not wait_for_key(env):
            print("Skipping scenario...")
            break

        # Шаг среды (действие заглушка)
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        total_reward += reward
        step += 1

        if done:
            print(f"Result: {'WIN' if reward > 0 else 'FAIL'}. Reward: {total_reward}")
            time.sleep(1)


def run_agents():
    # Инициализация
    base_env = GridWorldEnv(render_mode='human', size=GRID_SIZE, num_bombs=NUM_BOMBS)

    # Цикл тестирования
    for scenario in SCENARIOS:
        # Создаем свежий wrapper для каждого сценария
        env = HighLevelPlannerWrapper(base_env, plan_frequency=10)
        run_inference(env, scenario)

    base_env.close()


run_agents()