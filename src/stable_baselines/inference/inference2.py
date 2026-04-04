"""
inference2.py — Step-by-step rollout with action probability display.
Works for PPO and A2C (both use categorical distributions).
For DQN use inference1.py (Q-values available via model.policy.q_net).
For RecurrentPPO use inference_recurrent.py.

Usage: run from project root
    python src/stable_baselines/inference/inference2.py
Controls: Space or Enter in the PyGame window to advance one step.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pygame
from stable_baselines3 import PPO, A2C

from src.environment.environment import GridWorldEnv

# --- Config ---
ALGO = PPO
MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/models/ppo_size5')
SIZE = 5
NUM_BOMBS = 3 if SIZE == 5 else 10

ACTION_NAMES = ['Left', 'Right', 'Up', 'Down']


def wait_for_key(env):
    """Wait for Space/Enter in PyGame window. Returns False if window closed."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return True
        time.sleep(0.05)


env = GridWorldEnv(render_mode='human', size=SIZE, num_bombs=NUM_BOMBS)
model = ALGO.load(MODEL_PATH, env=env)

obs, info = env.reset()
done = False
total_reward = 0.0

print(f"Step-by-step mode. Press Space/Enter in the game window to advance.")

while not done:
    env.render()

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.cpu().numpy()[0]

    print('\n' + '=' * 35)
    for i, (name, p) in enumerate(zip(ACTION_NAMES, probs)):
        print(f"  {i} {name:6s}: {p * 100:5.1f}%")
    print('=' * 35)

    if not wait_for_key(env):
        break

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    total_reward += reward
    done = terminated or truncated

    print(f"--> Action: {int(action)} ({ACTION_NAMES[int(action)]})  reward: {reward:.4f}")

print(f"\nEpisode finished. Total reward: {total_reward:.3f}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()
