from src.environment.environment import GridWorldEnv
from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper
import pygame
import time

# 1. Создаем базовую среду
base_env = GridWorldEnv(render_mode='human', size=5, num_bombs=3)

# 2. Оборачиваем в LLM планировщика
# plan_frequency=10 означает, что LLM будет думать каждые 10 шагов
env = HighLevelPlannerWrapper(base_env, plan_frequency=10)

obs, info = env.reset()
done = False
total_reward = 0

print("Запуск LLM-автопилота (High-Level Planning)...")

# Основной цикл
while not done:
    # Визуализация
    env.render()

    # PPO не нужен, так как Wrapper игнорирует входящее действие.
    # Передаем 0 (или любое другое), внутри step оно будет заменено на действие от LLM/BFS.
    dummy_action = 0

    # Шаг среды
    obs, reward, terminated, truncated, info = env.step(dummy_action)
    done = terminated or truncated
    total_reward += reward

    # Обработка событий PyGame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    time.sleep(0.3)

print(f"Total Reward: {total_reward}")

# Корректное закрытие
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
env.close()