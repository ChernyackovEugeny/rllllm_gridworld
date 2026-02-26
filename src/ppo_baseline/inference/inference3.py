import torch as th
from stable_baselines3 import PPO
from src.environment.environment import GridWorldEnv
import pygame
import time
import os

# --- Конфигурация ---
# SCENARIOS = [
#     {
#         "name": "Simple Run",
#         "agent": [0, 0],
#         "target": [4, 4],
#         "bombs": [[2, 2], [2, 3], [3, 2]]
#     },
#     {
#         "name": "Maze Trap",
#         "agent": [2, 0],
#         "target": [2, 4],
#         "bombs": [[1, 2], [3, 2], [2, 1]]
#     }
# ]

SCENARIOS = [
    {
        "name": "1. The Great Wall (Навык обхода препятствий)",
        "description": "Сплошная стена из бомб преграждает прямой путь. Агент должен найти край стены и обойти её.",
        "agent": [0, 4],
        "target": [9, 4],
        # Стена на 4-й строке. Единственный проход - слева в колонке 0.
        # Агент должен пойти влево, вниз и потом вправо.
        "bombs": [
            [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], # Стена (оставили дырку в 4,0)
            [1, 1] # Просто чтобы было ровно 10 бомб (рандомная помеха)
        ]
    },
    {
        "name": "2. The Corridor (Узкий коридор)",
        "description": "Агент должен пройти по узкому коридору, не задевая стены из бомб.",
        "agent": [0, 5],
        "target": [9, 5],
        # Две вертикальные линии бомб, создающие коридор шириной в 3 клетки.
        # Bombs на колонках 3 и 7.
        "bombs": [
            [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], # Левая стена
            [2, 7], [3, 7], [4, 7], [5, 7]                  # Правая стена (частичная)
        ]
    },
    {
        "name": "3. Diagonal Trap (Блокировка диагонали)",
        "description": "Бомбы блокируют кратчайший диагональный путь. Агент должен 'зигзагом' обходить опасные зоны.",
        "agent": [0, 0],
        "target": [9, 9],
        # Бомбы выстроены по диагонали, заставляя агента отклоняться.
        "bombs": [
            [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8],
            [3, 2], [6, 5] # Дополнительные бомбы, чтобы закрыть "легкие" обходы
        ]
    },
    {
        "name": "4. Fortress (Поиск входа)",
        "description": "Цель окружена кольцом бомб. Агент должен найти единственный вход в 'крепость'.",
        "agent": [0, 4],
        "target": [4, 4],
        # Кольцо вокруг цели с одним проходом слева.
        "bombs": [
            [3, 4], [3, 5], [3, 6], [4, 6], [5, 6], [5, 5], [5, 4], [5, 3], [4, 3], [3, 3]
            # Вход открыт с 4-й колонки слева ([4, 3] - нет бомбы, [3, 3] есть, [5, 3] есть)
            # Схема:
            # . . B B B .
            # . . B T B .
            # . . . B B .
            # Проход через [3, 2] -> [4, 2] -> [4, 3] -> Target [4, 4]
        ]
    },
    {
        "name": "5. Sparse Field (Хаотичное поле)",
        "description": "Случайное распределение бомб на дальних дистанциях. Проверка общего интеллекта.",
        "agent": [9, 0], # Нижний левый угол
        "target": [0, 9], # Верхний правый угол
        "bombs": [
            [7, 2], [6, 3], [5, 5], [4, 4], [3, 6],
            [2, 8], [8, 1], [1, 2], [0, 5], [9, 8]
        ]
    }
]

MODELS_TO_TEST = [
    {'name': 'Agent Final', 'path': './models/ppo_base_newreward_10size_600k.t.s.'},
    # {'name': 'Agent Best', 'path': './best_model/best_model.zip'},
]

GRID_SIZE = 10
NUM_BOMBS = 10

def wait_for_key(env):
    """
    Ждет нажатия клавиши (Enter/Space), не блокируя окно Pygame.
    Возвращает False, если пользователь закрыл окно или нажал Q/ESC (для выхода).
    """
    print("\nНажмите 'Enter' или 'Пробел' в окне Pygame для следующего шага...")
    print("(Нажмите 'Q' или 'ESC' для пропуска сценария)")

    waiting = True
    while waiting:
        # Обрабатываем события Pygame, чтобы окно было "живым"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Закрытие окна

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    waiting = False  # Продолжаем
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False  # Прерываем текущий сценарий

        # Небольшая пауза, чтобы не грузить процессор на 100%
        time.sleep(0.05)

    return True

def run_inference(env, model, model_name, scenario):
    print(f"\n{'=' * 60}")
    print(f"Testing: {model_name} on Scenario: {scenario['name']}")
    print(f"{'=' * 60}")

    # 1. Сбрасываем среду
    env.reset(seed=42)

    # 2. Устанавливаем конкретную сетку
    # Убедитесь, что метод в environment.py называется именно set_custom_state (без подчеркивания)
    obs, info = env._set_custom_state(
        agent_loc=scenario['agent'],
        target_loc=scenario['target'],
        bombs_locs=scenario['bombs']
    )

    done = False
    total_reward = 0
    step = 0

    while not done:
        # --- Визуализация ---
        env.render()

        # --- Логиты и вероятности ---
        with th.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            distribution = model.policy.get_distribution(obs_tensor)
            logits = distribution.distribution.logits
            probs = distribution.distribution.probs

        print(f"\n--- Step {step} ---")
        print(f"Agent Pos: {env._agent_location}")

        # В environment.py: LEFT=0, RIGHT=1, UP=2, DOWN=3
        action_names = ["Влево", "Вправо", "Вверх", "Вниз"]

        for i, p in enumerate(probs.cpu().numpy()[0]):
            logit_val = logits.cpu().numpy()[0][i]
            print(f"  {action_names[i]:<7}: Prob {p * 100:5.2f}%  (Logit: {logit_val:6.2f})")

        # --- Пауза и Ввод (БЕЗ input()) ---
        should_continue = wait_for_key(env)
        if not should_continue:
            print("Прерывание сценария пользователем...")
            break

        # --- Действие ---
        action, _ = model.predict(obs, deterministic=False)
        action = int(action)

        print(f"-> Выбранное действие: {action} ({action_names[action]})")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        if done:
            print(f"\nЭпизод завершен! Причина: {'Win!' if reward > 0 else 'Fail/Limit'}. Total Reward: {total_reward}")
            # Пауза, чтобы увидеть финальное состояние
            time.sleep(1)


def run_agents():
    # Инициализация среды
    env = GridWorldEnv(render_mode='human', size=GRID_SIZE, num_bombs=NUM_BOMBS)

    loaded_models = []
    for model_info in MODELS_TO_TEST:
        path = model_info['path']
        # SB3 load ищет .zip, но в пути может не быть расширения
        if os.path.exists(path + '.zip') or os.path.exists(path):
            try:
                model = PPO.load(path, env=env)
                loaded_models.append({'name': model_info['name'], 'model': model})
                print(f"Модель загружена: {model_info['name']}")
            except Exception as e:
                print(f"Ошибка загрузки модели {model_info['name']}: {e}")
        else:
            print(f"Путь к модели не найден: {path}")

    if not loaded_models:
        print("Нет загруженных моделей. Выход.")
        return

    # Основной цикл тестирования
    for scenario in SCENARIOS:
        for agent_data in loaded_models:
            run_inference(env, agent_data['model'], agent_data['name'], scenario)

    env.close()

run_agents()