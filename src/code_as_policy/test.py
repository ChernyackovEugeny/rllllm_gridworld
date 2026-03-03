from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper
from stable_baselines3.common.monitor import Monitor
from src.environment.environment import GridWorldEnv

env = HighLevelPlannerWrapper(GridWorldEnv(num_bombs=3, size=5))
obs, info = env.reset()
print(obs)
print(env.known_world)