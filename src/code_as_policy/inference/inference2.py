import os
import pygame
import time
import gymnasium as gym
from src.environment.environment import GridWorldEnv
from src.code_as_policy.CodeGeneratorWrapper import CodeGeneratorWrapper

# --- Конфигурация сценариев ---
# Координаты: [row, col] (0,0 - верхний левый угол)
SCENARIOS = [
    {
        "name": "1. The Great Wall",
        "description": "Сплошная стена из бомб преграждает прямой путь. Нужно найти обход.",
        "grid_size": 10,
        "agent": [1, 4],
        "target": [8, 4],
        "bombs": [
            [6, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9]  # Сплошная стена
        ]
    },
    {
        "name": "2. The Maze",
        "description": "Лабиринт. Прямой путь заблокирован, нужно искать проход.",
        "grid_size": 10,
        "agent": [0, 0],
        "target": [9, 9],
        "bombs": [
            # Вертикальные стены с проходами
            [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2],
            [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5],
            # Горизонтальные барьеры
            [3, 3], [3, 4],
            [6, 6], [6, 7], [6, 8]
        ]
    },
    {
        "name": "3. The Snake",
        "description": "Длинный извилистый коридор. Нельзя срезать углы.",
        "grid_size": 7,
        "agent": [0, 0],
        "target": [6, 6],
        "bombs": [
            # Создаем S-образный путь
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
            [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6],
            [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5]
        ]
    }
]

# Определяем путь к файлу навыков относительно этого скрипта
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_PATH = os.path.join(CURRENT_DIR, '..', 'skills', 'skills.json')


def run_scenario(env, scenario_config):
    """
    Запускает один сценарий.
    """
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {scenario_config['name']}")
    print(f"DESC: {scenario_config['description']}")
    print(f"{'=' * 60}")

    # 1. Сбрасываем среду и обертку
    # reset() инициализирует known_world и память
    obs, info = env.reset()

    # 2. Устанавливаем кастомное состояние
    # Важно: делаем это после reset, чтобы очистить старую память
    env.unwrapped._set_custom_state(
        agent_loc=scenario_config['agent'],
        target_loc=scenario_config['target'],
        bombs_locs=scenario_config['bombs']
    )

    # 3. ВАЖНЫЙ МОМЕНТ: Сброс памяти обертки
    # Так как мы изменили состояние среды "через голову" обертки,
    # нужно вручную обновить её память, чтобы LLM не видел "призраков" прошлой карты.
    env.known_world = {}
    # Принудительно сканируем стартовую позицию
    env._update_memory()

    # Обновляем obs, т.к. он устарел после _set_custom_state
    obs = env.unwrapped._get_obs()

    done = False
    total_reward = 0
    step = 0
    max_steps = 200  # Лимит шагов на сценарий

    while not done and step < max_steps:
        env.render()

        # Вывод позиции для отладки
        # print(f"Step {step} | Pos: {tuple(env.unwrapped._agent_location)}")

        # Передаем dummy action (0), т.к. Wrapper его проигнорирует и спросит LLM
        obs, reward, terminated, truncated, info = env.step(0)

        done = terminated or truncated
        total_reward += reward
        step += 1

        # Небольшая задержка для визуализации
        time.sleep(0.2)

        # Обработка событий PyGame (чтобы окно не зависло)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Сигнал остановить всё

    print(f"\nRESULT: {'WIN 🏆' if terminated and reward > 0 else 'FAIL 💥'}")
    print(f"Total Reward: {total_reward:.2f} | Steps: {step}")

    # Пауза перед следующим сценарием
    time.sleep(1)
    return True


def main():
    # Создаем базовую среду (размер будет меняться динамически в сценариях,
    # но базовую инициализацию делаем под самый большой сценарий)
    base_env = GridWorldEnv(render_mode='human', size=10, num_bombs=20)

    # Создаем обертку с LLM
    # Передаем абсолютный путь к скиллам
    env = CodeGeneratorWrapper(base_env, skills_path=SKILLS_PATH)

    running = True
    for scenario in SCENARIOS:
        if not running:
            break

        # Обновляем размер среды под сценарий (если он меняется)
        if env.unwrapped.size != scenario['grid_size']:
            env.unwrapped.size = scenario['grid_size']
            env.unwrapped.max_steps = scenario['grid_size'] ** 2 * 2
            env.unwrapped.window_size = 512  # Можно оставить фиксированным

        running = run_scenario(env, scenario)

    env.close()
    print("\nAll scenarios completed.")


if __name__ == "__main__":
    main()