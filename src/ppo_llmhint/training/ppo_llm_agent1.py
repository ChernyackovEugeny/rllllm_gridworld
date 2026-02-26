#PPO13-mocktest/PPO14-llmtest/PPO15-local_disc_check/PPO16-firstlearning/PPO17-bestpromt/PPO18-20k

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import GridWorldEnv
from src.ppo_llmhint.wrappers import LLMExplorerWrapper

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
    n_steps=128,
    batch_size=64,
    gamma=0.99,
    n_epochs=10,
    tensorboard_log="../../logs/"
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
model.learn(total_timesteps=20000)

model.save('./models/ppo_llm_test')
