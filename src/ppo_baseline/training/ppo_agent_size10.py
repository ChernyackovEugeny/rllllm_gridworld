#PPO9-600k-size10/PPO10-1m-size10/PPO12-new_reward-600k-size-5/

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import GridWorldEnv

def make_env():
    env = GridWorldEnv(render_mode=None, size=10, num_bombs=10)
    env = Monitor(env)
    return env

vec_env = make_vec_env(make_env, n_envs=4)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.999,
    tensorboard_log="./logs/"
)

eval_env = Monitor(
    GridWorldEnv(render_mode=None, size=10, num_bombs=10)
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='../../best_model/',
    log_path='../../logs/',
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

print("Начинаем обучение...")
model.learn(total_timesteps=600_000, callback=eval_callback)

model.save('./models/ppo_base_newreward_10size_600k.t.s.')