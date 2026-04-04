"""
inference_recurrent.py — Visual rollout for RecurrentPPO (sb3-contrib).
RecurrentPPO requires passing LSTM hidden states between steps.

Usage: run from project root
    python src/stable_baselines/inference/inference_recurrent.py
"""
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pygame
from sb3_contrib import RecurrentPPO

from src.environment.environment import GridWorldEnv

# --- Config ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/models/recurrent_ppo_size5')
SIZE = 5
NUM_BOMBS = 3 if SIZE == 5 else 10
STEP_DELAY = 0.3

env = GridWorldEnv(render_mode='human', size=SIZE, num_bombs=NUM_BOMBS)
model = RecurrentPPO.load(MODEL_PATH, env=env)

obs, info = env.reset()
done = False
total_reward = 0.0

# RecurrentPPO needs LSTM state passed explicitly
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

print(f"Running RecurrentPPO on {SIZE}x{SIZE} grid. Close window to exit.")

while not done:
    action, lstm_states = model.predict(
        obs,
        state=lstm_states,
        episode_start=episode_starts,
        deterministic=True,
    )
    episode_starts = np.zeros((1,), dtype=bool)

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
