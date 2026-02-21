#PPO13-mocktest/PPO14-llmtest

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from environment import GridWorldEnv
from wrappers import LLMExplorerWrapper

def make_env():
    env = GridWorldEnv(render_mode=None, size=5, num_bombs=3)
    env = Monitor(LLMExplorerWrapper(env))
    return env

vec_env = make_vec_env(make_env, n_envs=4)

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

# eval_env = Monitor(
#     LLMExplorerWrapper(GridWorldEnv(render_mode=None, size=5, num_bombs=3))
# )
# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path='./best_model/',
#     log_path='./logs/',
#     eval_freq=100,
#     n_eval_episodes=10,
#     deterministic=True,
#     render=False
# )

print("Начинаем обучение...")
model.learn(total_timesteps=50)

model.save('./models/ppo_llm_test')