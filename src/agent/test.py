from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper
from stable_baselines3.common.monitor import Monitor
from src.environment.environment import GridWorldEnv

from src.code_as_policy.CodeExecutor import get_path_to_target

env = HighLevelPlannerWrapper(GridWorldEnv(num_bombs=3, size=5))
obs, info = env.reset()
known_world = env.known_world
agent_pos = tuple(map(int, env.unwrapped._agent_location))
print(agent_pos, get_path_to_target(agent_pos, (3,3), known_world, 5))