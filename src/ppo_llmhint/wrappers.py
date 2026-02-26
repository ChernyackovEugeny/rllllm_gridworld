import gymnasium as gym
import numpy as np
from src.ppo_llmhint.llm_advisor import DeepSeekExplorer
from src.environment.environment import Places


class LLMExplorerWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # self.llm = MockExplorer()
        self.llm = DeepSeekExplorer()

        self.cache = {}

        self.cache_count = 0
        self.total_calls = 0

        # Хранит {(x, y): 'TYPE'}. Это и есть суммаризация опыта.
        self.known_world = {}

        # Расширяем observation_space (добавляем вектор подсказки)
        # Старое пространство + новый ключ 'llm_hint'
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            'llm_hint': gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Сбрасываем память мира
        self.known_world = {}

        self.cache = {}

        # Сразу добавляем стартовую позицию в память
        start_pos = tuple(self.env.unwrapped._agent_location)
        self.known_world[start_pos] = 'VISITED'

        return self.observation(obs), info

    def observation(self, obs):
        # 1. Получаем данные для LLM
        agent_pos = tuple(self.env.unwrapped._agent_location)

        # 1. Получаем текущий обзор
        local_view_str, current_view_objects = self._get_local_view_description()

        # 2. Обновляем глобальную память (Summarizer)
        # Добавляем увиденное в known_world
        for coord, obj_type in current_view_objects.items():
            # Если мы видим цель или бомбу — перезаписываем (это важно)
            # Если видим пустоту — можно не сохранять или помечать SAFE
            if obj_type in ['TARGET', 'DANGER', 'WALL']:
                self.known_world[coord] = obj_type

        # Помечаем текущую позицию как посещенную
        self.known_world[agent_pos] = 'VISITED'

        action_idx = self._get_smart_hint(
            agent_pos,
            local_view_str,
            self.known_world
        )

        hint_vec = np.zeros(4, dtype=np.float32)
        hint_vec[action_idx] = 1.0

        obs['llm_hint'] = hint_vec

        return obs

    def _get_smart_hint(self, agent_pos, local_view_text, known_world):
        self.total_calls += 1  # Учитываем вызов

        # Ключ должен учитывать накопленную память!
        # Иначе агент, вернувшись на клетку и зная о цели, пойдет рандомно (по старому кэшу)
        memory_hash = frozenset(known_world.items())
        cache_key = (agent_pos, local_view_text, memory_hash)

        if cache_key in self.cache:
            self.cache_count += 1
            return self.cache[cache_key]

        # Если в кэше нет - зовем API
        action_idx = self.llm.get_exploration_action(agent_pos, local_view_text, known_world)

        self.cache[cache_key] = action_idx
        return action_idx

    def _get_local_view_description(self):
        """
        Возвращает строку для промпта И словарь увиденных объектов для памяти.
        """
        env = self.env.unwrapped
        agent_x, agent_y = env._agent_location
        interesting_objects = []  # Для строки
        objects_dict = {}  # Для памяти

        # Определяем зону обзора.
        # Если agent_observations имеет размер 5x5, то радиус = 2.
        view_range = range(-2, 3)

        for dx in view_range:
            for dy in view_range:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = agent_x + dx, agent_y + dy

                # --- ЛОГИКА ФИЛЬТРА ---

                # 1. Проверяем, находится ли клетка ВНУТРИ карты
                if 0 <= nx < env.size and 0 <= ny < env.size:
                    cell_value = env.grid[nx, ny]

                    # Внутри карты игнорируем пустоту
                    if cell_value == Places.NOTHING.value:
                        continue

                    # Обрабатываем полезные объекты
                    if cell_value == Places.TARGET.value:
                        obj_type = "TARGET"
                    elif cell_value == Places.BOMB.value:
                        obj_type = "DANGER"
                    interesting_objects.append(f"({nx}, {ny}): {obj_type}")
                    objects_dict[(nx, ny)] = obj_type

                # 2. Если клетка СНАРУЖИ карты (это стены и паддинг)
                else:
                    # Проверяем, является ли эта клетка "честным слоем" стены.
                    # Честный слой — это непосредственные соседи карты:
                    # координаты -1 или size.

                    is_honest_wall = False

                    # Проверка по X
                    if nx == -1 or nx == env.size:
                        is_honest_wall = True
                    # Проверка по Y
                    if ny == -1 or ny == env.size:
                        is_honest_wall = True

                    # Если это не честный слой (например, nx = -2), то это паддинг.
                    # Мы его просто пропускаем (continue).
                    if not is_honest_wall:
                        continue

                    # Если это честный слой стены, добавляем в описание
                    interesting_objects.append(f"({nx}, {ny}): WALL")
                    objects_dict[(nx, ny)] = "WALL"

        # Собираем все в одну строку через запятую
        if not interesting_objects:
            return "Open space around", objects_dict

        view_disc = ", ".join(interesting_objects)
        # print(view_disc)
        # print(agent_x, agent_y)

        return ", ".join(interesting_objects), objects_dict