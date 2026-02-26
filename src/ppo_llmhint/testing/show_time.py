from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import GridWorldEnv
from wrappers import LLMExplorerWrapper

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

def make_env():
    env = GridWorldEnv(render_mode=None, size=5, num_bombs=3)
    env = Monitor(LLMExplorerWrapper(env))
    return env

vec_env = make_vec_env(make_env, n_envs=1)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=10,
    batch_size=10,
    gamma=0.999,
    tensorboard_log="./logs/"
)

print("Начинаем обучение...")
model.learn(total_timesteps=10)

profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumtime")  # сортировка по суммарному времени
stats.print_stats(20)         # топ-20 функций