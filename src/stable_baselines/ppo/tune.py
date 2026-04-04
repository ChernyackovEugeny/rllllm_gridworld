"""
Optuna hyperparameter search for PPO.
Run from project root:
    python src/stable_baselines/ppo/tune.py

Results are saved to src/models/optuna/ppo_size{SIZE}.db (SQLite).
After the study finishes, best params are printed and written to
src/models/optuna/ppo_size{SIZE}_best_params.json.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.environment import GridWorldEnv

# --- Config ---
SIZE       = 10      # Grid size: 5 or 10
NUM_BOMBS  = 3 if SIZE == 5 else 10
N_TRIALS   = 25     # number of Optuna trials
TUNE_STEPS = 100_000  # timesteps per trial (~17% of full training)
N_ENVS     = 4
EVAL_EPISODES = 20
OPTUNA_DIR = os.path.join(PROJECT_ROOT, 'src/models/optuna')


class PruningCallback(BaseCallback):
    """Reports intermediate rewards to Optuna and prunes bad trials early.

    Uses self.num_timesteps (actual env steps = n_calls × n_envs) instead of
    self.n_calls so eval_freq is independent of n_envs. Uses >= instead of %
    because num_timesteps jumps by n_envs per call and may skip exact multiples.
    """

    def __init__(self, trial, eval_env, eval_freq=20_000):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq          # in actual env steps
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep >= self.eval_freq:
            self._last_eval_timestep = self.num_timesteps
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5, warn=False
            )
            self.trial.report(mean_reward, step=self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True


def make_env():
    env = GridWorldEnv(render_mode=None, size=SIZE, num_bombs=NUM_BOMBS)
    return Monitor(env)


def objective(trial: optuna.Trial) -> float:
    lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps    = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma      = trial.suggest_float("gamma", 0.95, 0.9999, log=False)
    ent_coef   = trial.suggest_float("ent_coef", 0.0, 0.05)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    # batch_size must divide n_steps * N_ENVS (PPO requirement)
    if (n_steps * N_ENVS) % batch_size != 0:
        raise optuna.TrialPruned()

    vec_env  = make_vec_env(make_env, n_envs=N_ENVS)
    eval_env = make_env()

    model = PPO(
        'MultiInputPolicy',
        vec_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=TUNE_STEPS,
            callback=PruningCallback(trial, eval_env),
        )
        mean_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=EVAL_EPISODES, warn=False
        )
    except optuna.TrialPruned:
        raise
    finally:
        vec_env.close()
        eval_env.close()

    return mean_reward


def main():
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    db_path = os.path.join(OPTUNA_DIR, f'ppo_size{SIZE}.db')
    out_path = os.path.join(OPTUNA_DIR, f'ppo_size{SIZE}_best_params.json')

    study = optuna.create_study(
        study_name=f'ppo_size{SIZE}',
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30_000),
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n=== Best trial ===")
    print(f"  Value : {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    with open(out_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest params saved -> {out_path}")
    print(f"Full study DB     -> {db_path}")
    print("Visualize with: optuna-dashboard sqlite:///" + db_path)


if __name__ == '__main__':
    main()
