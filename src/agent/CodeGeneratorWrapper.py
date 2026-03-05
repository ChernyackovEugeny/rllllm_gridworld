import gymnasium as gym

from src.environment.environment import Places, Actions
from src.agent.CodeExecutor import CodeExecutor
from src.agent.CodeGenerator import CodeGenerator
from src.agent.SkillManager import SkillManager

from src.agent.Perception import Perception
from src.agent.Memory import Memory
from src.agent.MetaController import MetaController

class CodeGeneratorWrapper(gym.Wrapper):
    def __init__(self, env, max_fix_retries=3, skills_path='skills.json', max_skill_steps=100, strategy=None):
        super().__init__(env)

        self.strategy = strategy
        self.max_fix_retries = max_fix_retries
        self.max_skill_steps = max_skill_steps

        self.executor = CodeExecutor(env)
        self.generator = CodeGenerator()
        self.skill_manager = SkillManager(step_penalty=env.unwrapped._step_penalty, skills_path=skills_path)
        self.perception = Perception()
        self.memory = Memory(size=env.unwrapped.size)

        # мозг
        self.meta_controller = MetaController(
            skill_manager=self.skill_manager,
            planner=self.generator,
            executor=self.executor,
            strategy=strategy,
            max_skill_steps=max_skill_steps,
            max_fix_retries=max_fix_retries
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        current_view_objects = self.perception.get_local_observations(self.env)
        agent_pos = self.perception.get_agent_position(self.env)

        self.memory.reset(agent_pos)
        self.memory.update(current_view_objects, agent_pos)

        return obs, info

    def step(self, action):
        # Сначала смотрим, что вокруг, прежде чем принимать решение
        current_view_objects = self.perception.get_local_observations(self.env)
        agent_pos = self.perception.get_agent_position(self.env)

        self.memory.update(current_view_objects, agent_pos)

        context = self.memory.get_context(agent_pos)
        print(context['map_string'])

        action, code_used, source, skill_id = self.meta_controller.decide_action(context)

        # 4. Исполнение действия в среде
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 5. Обучение (Learn)
        self.meta_controller.learn(
            source=source,
            code_used=code_used,
            skill_id=skill_id,
            reward=reward
        )

        return obs, reward, terminated, truncated, info