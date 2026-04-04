import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import GridWorldEnv

# --- Config ---
SIZE = 10           # Grid size: 5 or 10
NUM_BOMBS = 3 if SIZE == 5 else 10
TOTAL_TIMESTEPS = 600_000
N_ENVS = 8

# Tuned hyperparams per grid size (Optuna)
PARAMS = {
    5: dict(
        learning_rate = 7e-4,
        n_steps       = 512,    # 128 → слишком шумный gradient на sparse reward
        gamma         = 0.99,   # 0.965 слишком низкая: 0.965^15=0.59 (цель обесценивается)
        ent_coef      = 0.05189322158038423,
    ),
    10: dict(
        learning_rate = 0.005051471660486723,
        n_steps       = 512,    # 64 → слишком шумный gradient на sparse reward
        gamma         = 0.9920441422619156,
        ent_coef      = 0.04053657460498488,
    ),
}

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, f'src/models/a2c_size{SIZE}')
LOG_DIR         = os.path.join(PROJECT_ROOT, 'src/logs')
BEST_MODEL_DIR  = os.path.join(PROJECT_ROOT, f'src/best_models/a2c_size{SIZE}')


def make_env():
    env = GridWorldEnv(render_mode=None, size=SIZE, num_bombs=NUM_BOMBS)
    return Monitor(env)


def main():
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    model = A2C(
        'MultiInputPolicy',
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        **PARAMS[SIZE],
    )

    eval_env = Monitor(GridWorldEnv(render_mode=None, size=SIZE, num_bombs=NUM_BOMBS))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5_120,   # must be multiple of n_steps=128 for N_ENVS=8
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print(f"Training A2C on {SIZE}x{SIZE} grid for {TOTAL_TIMESTEPS} steps ({N_ENVS} envs)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, log_interval=10)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
