"""
Optuna hyperparameter search for DQN.
Run from project root:
    python src/stable_baselines/dqn/tune.py

Note: DQN does not support multiple envs in SB3 — uses a single env per trial.

Results are saved to src/models/optuna/dqn_size{SIZE}.db (SQLite).
After the study finishes, best params are printed and written to
src/models/optuna/dqn_size{SIZE}_best_params.json.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.environment import GridWorldEnv

# --- Config ---
SIZE       = 10      # Grid size: 5 or 10
NUM_BOMBS  = 3 if SIZE == 5 else 10
N_TRIALS   = 20
# DQN is slower per step (no parallel envs) — keep steps low
TUNE_STEPS = 50_000
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
    lr                   = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size          = trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000])
    batch_size           = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma                = trial.suggest_float("gamma", 0.95, 0.9999)
    train_freq           = trial.suggest_categorical("train_freq", [1, 4, 8])
    target_update_interval = trial.suggest_int("target_update_interval", 500, 2_000, step=500)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.3)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)

    env      = make_env()
    eval_env = make_env()

    model = DQN(
        'MultiInputPolicy',
        env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        learning_starts=1_000,
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=TUNE_STEPS,
            callback=PruningCallback(trial, eval_env, eval_freq=10_000),
        )
        mean_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=EVAL_EPISODES, warn=False
        )
    except optuna.TrialPruned:
        raise
    finally:
        env.close()
        eval_env.close()

    return mean_reward


def main():
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    db_path = os.path.join(OPTUNA_DIR, f'dqn_size{SIZE}.db')
    out_path = os.path.join(OPTUNA_DIR, f'dqn_size{SIZE}_best_params.json')

    study = optuna.create_study(
        study_name=f'dqn_size{SIZE}',
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=15_000),
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
