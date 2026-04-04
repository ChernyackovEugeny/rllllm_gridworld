import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import GridWorldEnv

# --- Config ---
SIZE = 10           # Grid size: 5 or 10
NUM_BOMBS = 3 if SIZE == 5 else 10
TOTAL_TIMESTEPS = 600_000

# Tuned hyperparams per grid size (Optuna)
PARAMS = {
    5: dict(
        learning_rate         = 0.0006944721026320229,
        buffer_size           = 50_000,
        batch_size            = 128,
        gamma                 = 0.990289383985644,
        train_freq            = 1,
        target_update_interval= 500,
        exploration_fraction  = 0.23024656219884343,
        exploration_final_eps = 0.0974534463575501,
    ),
    10: dict(
        learning_rate         = 0.00012125494787012271,
        buffer_size           = 10_000,
        batch_size            = 32,
        gamma                 = 0.9652756155337863,
        train_freq            = 1,
        target_update_interval= 1_000,
        exploration_fraction  = 0.0537590546479088,
        exploration_final_eps = 0.053468780352593934,
    ),
}

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, f'src/models/dqn_size{SIZE}')
LOG_DIR         = os.path.join(PROJECT_ROOT, 'src/logs')
BEST_MODEL_DIR  = os.path.join(PROJECT_ROOT, f'src/best_models/dqn_size{SIZE}')

# DQN does not support multiple envs in SB3
env = Monitor(GridWorldEnv(render_mode=None, size=SIZE, num_bombs=NUM_BOMBS))

model = DQN(
    'MultiInputPolicy',
    env,
    verbose=1,
    learning_starts=1_000,
    tensorboard_log=LOG_DIR,
    **PARAMS[SIZE],
)

eval_env = Monitor(GridWorldEnv(render_mode=None, size=SIZE, num_bombs=NUM_BOMBS))
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=5_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

print(f"Training DQN on {SIZE}x{SIZE} grid for {TOTAL_TIMESTEPS} steps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}.zip")

env.close()
eval_env.close()
