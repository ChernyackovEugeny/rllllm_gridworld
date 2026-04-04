# PPO_22

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import GridWorldEnv

# --- Config ---
SIZE = 10           # Grid size: 5 or 10
NUM_BOMBS = 3 if SIZE == 5 else 10
TOTAL_TIMESTEPS = 600_000
N_ENVS = 4

# Tuned hyperparams per grid size (Optuna)
PARAMS = {
    5: dict(
        learning_rate = 0.000648238159116623,
        n_steps       = 1024,
        batch_size    = 64,
        gamma         = 0.9584035517822008,
        ent_coef      = 0.025678975506009463,
        clip_range    = 0.29267642972448105,
    ),
    10: dict(
        learning_rate = 8.737526343968795e-05,
        n_steps       = 512,
        batch_size    = 32,
        gamma         = 0.9988366686361552,
        ent_coef      = 0.04982802269809609,
        clip_range    = 0.10476157417326579,
    ),
}

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, f'src/models/ppo_size{SIZE}')
LOG_DIR         = os.path.join(PROJECT_ROOT, 'src/logs')
BEST_MODEL_DIR  = os.path.join(PROJECT_ROOT, f'src/best_models/ppo_size{SIZE}')


def make_env():
    env = GridWorldEnv(render_mode=None, size=SIZE, num_bombs=NUM_BOMBS)
    return Monitor(env)


def main():
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    model = PPO(
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
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print(f"Training PPO on {SIZE}x{SIZE} grid for {TOTAL_TIMESTEPS} steps ({N_ENVS} envs)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
