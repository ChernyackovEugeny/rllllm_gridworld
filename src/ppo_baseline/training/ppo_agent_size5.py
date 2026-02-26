#PPO8-gridworld_final-size5/PPO11-newreward_600k-size5

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import GridWorldEnv

def make_env():
    env = GridWorldEnv(render_mode=None, size=5, num_bombs=3)
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
    GridWorldEnv(render_mode=None, size=5, num_bombs=3)
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
model.learn(total_timesteps=600000, callback=eval_callback)

model.save('./models/ppo_base_newreward_5size_600k.t.s.')