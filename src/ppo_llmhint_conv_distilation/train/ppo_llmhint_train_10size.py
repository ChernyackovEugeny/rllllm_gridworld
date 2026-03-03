#PPO20_600k_10size/PPO21_1200k_10size

from src.environment.environment import GridWorldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.ppo_llmhint_conv_distilation.FastLLMHintWrapper import FastLLMHintWrapper

def make_env():
    env = GridWorldEnv(render_mode=None, size=10, num_bombs=10)
    env = Monitor(FastLLMHintWrapper(env, map_size=10))
    return env

vec_env = make_vec_env(make_env, n_envs=4)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="../../logs/"
)

eval_env = Monitor(
    FastLLMHintWrapper(GridWorldEnv(render_mode=None, size=10, num_bombs=10), map_size=10)
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

print("Начинаем обучение PPO со Студентом...")
model.learn(total_timesteps=1_200_000)
model.save('../../models/ppo_llmhint_student_1200k_10size')