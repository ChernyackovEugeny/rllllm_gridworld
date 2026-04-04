"""
inference1.py — Visual rollout for any standard SB3 model (PPO, DQN, A2C).
For RecurrentPPO use inference_recurrent.py.

Usage: run from project root
    python src/stable_baselines/inference/inference1.py
"""
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pygame
from stable_baselines3 import PPO, DQN, A2C

from src.environment.environment import GridWorldEnv

# --- Config ---
ALGO = PPO                                                  # Change to DQN or A2C as needed
MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/models/ppo_size5')  # Path without .zip
SIZE = 5
NUM_BOMBS = 3 if SIZE == 5 else 10
STEP_DELAY = 0.3  # seconds between steps

env = GridWorldEnv(render_mode='human', size=SIZE, num_bombs=NUM_BOMBS)
model = ALGO.load(MODEL_PATH, env=env)

obs, info = env.reset()
done = False
total_reward = 0.0

print(f"Running {ALGO.__name__} on {SIZE}x{SIZE} grid. Close window to exit.")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    total_reward += reward
    done = terminated or truncated

    env.render()
    time.sleep(STEP_DELAY)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

print(f"Episode finished. Total reward: {total_reward:.3f}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()
