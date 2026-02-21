from stable_baselines3 import PPO
from environment import GridWorldEnv
import pygame
import time
import numpy as np

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env():
    # Убедитесь, что рендер включен, если нужно визуально видеть процесс
    env = GridWorldEnv(render_mode='human', size=5, num_bombs=3)
    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
env = VecNormalize.load("./env/vecnormalize.pkl", env)

# Включаем нормализацию наблюдений (это важно!)
env.norm_obs = True
env.norm_reward = False
env.training = False

# Исправление 1: reset возвращает только obs для VecEnv
obs = env.reset()

# Проверка типа (должно быть dict или numpy array)
print(f"Type of obs after reset: {type(obs)}")
if isinstance(obs, dict):
    print("Keys:", obs.keys())

model = PPO.load('./models/ppo_gridworld_final', env=env)

done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)

    # VecEnv сам ожидает векторизованное действие, predict уже возвращает нужный формат
    # но если вы хотите вручную контролировать action, убедитесь, что он обернут в список/массив
    # action для DummyVecEnv должен быть массивом или списком действий для каждой среды

    # Исправление 2: step возвращает 4 значения (obs, reward, done, info)
    # done здесь - это массив булевых значений [False] или [True]
    obs, reward, done, info = env.step(action)

    total_reward += reward[0]

    env.render()

    # Обработка событий PyGame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = [True]  # Нужно передать массив, так как среда векторизована

    # Небольшая пауза
    time.sleep(0.3)

print(f"Total Reward: {total_reward}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()