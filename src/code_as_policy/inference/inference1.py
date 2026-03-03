from src.environment.environment import GridWorldEnv
from src.code_as_policy.CodeGeneratorWrapper import CodeGeneratorWrapper
import pygame
import time

base_env = GridWorldEnv(render_mode='human', size=5, num_bombs=3)
env = CodeGeneratorWrapper(base_env, skills_path='../skills/skills.json')

obs, info = env.reset()
done = False
total_reward = 0

print("Запуск LLM-автопилота (Code-Generator)...")

# Основной цикл
while not done:
    # Визуализация
    env.render()

    # PPO не нужен, так как Wrapper игнорирует входящее действие.
    # Передаем 0 (или любое другое), внутри step оно будет заменено на действие от LLM.
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

env.render()
print(f"Total Reward: {total_reward}")

# Корректное закрытие
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
env.close()