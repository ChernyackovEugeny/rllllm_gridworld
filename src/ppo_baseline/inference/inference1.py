from stable_baselines3 import PPO
from src.environment.environment import GridWorldEnv
import pygame
import time

# Создаем среду
env = GridWorldEnv(render_mode='human', size=5, num_bombs=3)
model = PPO.load('../../models/ppo_base_newreward_5size_600k.t.s', env=env)
# model = PPO.load('./best_model/best_model.zip', env=env)

obs, info = env.reset()
done = False
total_reward = 0

# Основной цикл
while not done:
    # Получаем действие от агента
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)

    # Шаг среды
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    # Обработка событий PyGame, чтобы окно реагировало на закрытие
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Отрисовка
    env.render()

    # Небольшая пауза для визуализации
    time.sleep(0.3)

print(f"Total Reward: {total_reward}")

# Ждем закрытия окна пользователем
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()