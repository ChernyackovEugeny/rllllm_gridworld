import gymnasium as gym
import numpy as np
from llm_advisor import DeepSeekExplorer
from llm_advisor import MockExplorer
import time


class LLMExplorerWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # self.llm = MockExplorer()
        self.llm = DeepSeekExplorer()

        # Память агента
        self.visited_cells = set()

        # Расширяем observation_space (добавляем вектор подсказки)
        # Старое пространство + новый ключ 'llm_hint'
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            'llm_hint': gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Очищаем память при новом эпизоде
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.env.unwrapped._agent_location))
        return self.observation(obs), info

    def observation(self, obs):
        # 1. Получаем данные для LLM
        agent_pos = tuple(self.env.unwrapped._agent_location)
        local_view = self._get_local_view_description()

        # 2. Запрашиваем совет у DeepSeek
        # (Этот вызов идет по интернету, занимает ~0.5-2 секунды)

        # t0 = time.time()
        # print("LLM thinking...")  # <--- Добавь это
        action_idx = self.llm.get_exploration_action(
            agent_pos,
            local_view,
            self.visited_cells
        )
        # t1 = time.time()
        # print(f"LLM decided: {action_idx}")  # <--- И это
        # print("api step:", t1 - t0)

        # 3. Превращаем в One-Hot вектор
        hint_vec = np.zeros(4, dtype=np.float32)
        hint_vec[action_idx] = 1.0

        # 4. Добавляем в наблюдение
        obs['llm_hint'] = hint_vec

        # 5. Обновляем память (для следующего шага)
        self.visited_cells.add(agent_pos)

        return obs

    def _get_local_view_description(self):
        """Простой перевод матрицы 5x5 в текст"""
        # Твоя матрица local_patch из среды (0=пусто, 2=стена и т.д.)
        # Для простоты примера вернем заглушку, но логика такая:
        # проверь self.env.unwrapped._agent_observations
        return "Empty N, Empty E, Wall S, Empty W"