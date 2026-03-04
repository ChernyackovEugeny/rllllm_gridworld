import gymnasium as gym

from src.environment.environment import Places, Actions
from src.code_as_policy.CodeExecutor import CodeExecutor
from src.code_as_policy.CodeGenerator import CodeGenerator
from src.code_as_policy.SkillManager import SkillManager

class CodeGeneratorWrapper(gym.Wrapper):
    def __init__(self, env, max_fix_retries=3, skills_path='skills.json', strategy=None):
        super().__init__(env)

        self.executor = CodeExecutor(env)
        self.generator = CodeGenerator()
        self.skill_manager = SkillManager(step_penalty=env.unwrapped._step_penalty, skills_path=skills_path)

        self.known_world = {}
        self.strategy = strategy
        self.max_fix_retries = max_fix_retries

        self.used_skills_path = []  # сохраняем айдишники использованных скилов

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.known_world = {}

        self.used_skills_path = {}  # dict, key = skill_id, values = {'usage_count': ..., 'total_reward': ...,
        # 'mean_reward': ..., 'success_count': ..., 'success_rate': ..., 'skill_score': ...}

        # Сразу добавляем стартовую позицию в память
        start_pos = tuple(map(int, self.env.unwrapped._agent_location))
        self.known_world[start_pos] = 'VISITED'

        # Обновляем память тем, что видно на старте
        self._update_memory()

        return obs, info

    def step(self, action):
        # Сначала смотрим, что вокруг, прежде чем принимать решение
        self._update_memory()

        # Получаем текстовую карту
        map_string = self._get_map_string()
        agent_pos = tuple(map(int, self.env.unwrapped._agent_location))
        _, local_view = self._get_local_view_description()
        print(map_string)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Формируем сводку ситуации для поиска навыка
        situation_summary = (
            f"Current Map (A=Agent, T=Target, B=Bomb, V=Visited, .=Unknown):\n{map_string}\n"
            f"Current Position: {agent_pos}"
        )

        llm_action = None
        code_used = None
        source = "None"

        # --- ПОПЫТКА 1: Использовать Навык ---
        # dict, key = skill_id, values = {'usage_count': ..., 'total_reward': ...,
        # 'mean_reward': ..., 'success_count': ..., 'success_rate': ..., 'skill_score': ...}

        skill_code, skill_id = self.skill_manager.get_relevant_skill_code(situation_summary)
        if skill_code:
            llm_action, error = self.executor.execute_llm_code(skill_code, self.known_world, agent_pos)

            if llm_action is None:
                # Это значит go_to вернул None (путь заблокирован) или LLM вернул ерунду
                error = "Error: go_to() failed or returned None. Path might be blocked or target unreachable."

            code_used = skill_code
            source = "Skill Library"

            # Если навык сработал, идем дальше
            if not error:
                obs, reward, terminated, truncated, info = self.env.step(llm_action)
                self.skill_manager.update_skill_data(skill_id, reward)
                return obs, reward, terminated, truncated, info
            else:
                print("⚠️ Retrieved skill failed, falling back to generation...")

        # --- ПОПЫТКА 2: Генерация нового кода (если навыка нет или он упал) ---
        print('pikpik')
        code_data = self.generator.get_code(agent_pos, self.known_world, self.strategy, map_string)
        llm_action, error = self.executor.execute_llm_code(code_data, self.known_world, agent_pos)

        if llm_action is None:
            # Это значит go_to вернул None (путь заблокирован) или LLM вернул ерунду
            error = "Error: go_to() failed or returned None. Path might be blocked or target unreachable."

        retries = 0
        while error and retries < self.max_fix_retries:
            print(error)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!1
            code_data = self.generator.fix_code(code_data, error)
            llm_action, error = self.executor.execute_llm_code(code_data, self.known_world, agent_pos)
            retries += 1
        code_used = code_data
        source = "New Generation"

        # fallback
        if error:
            print('fallback')  # !!!!!!!!!!!!!!!!!!!!!!!
            llm_action = Actions.UP.value

        if not isinstance(llm_action, int):
            print("Invalid action, fallback")
            llm_action = Actions.UP.value
            return self.env.step(llm_action)

        obs, reward, terminated, truncated, info = self.env.step(llm_action)

        # --- ЭКОНОМИЯ И ОБУЧЕНИЕ: Если сгенерированный код сработал, сохраняем его ---
        if source == "New Generation" and not error:
            self.skill_manager.critique_and_save(code_used, reward)

        return obs, reward, terminated, truncated, info

    def _update_memory(self):
        """Обновляет known_world на основе текущего обзора."""
        _, current_view_objects = self._get_local_view_description()
        agent_pos = tuple(map(int, self.env.unwrapped._agent_location))

        for coord, obj_type in current_view_objects.items():
            if obj_type in ['TARGET', 'DANGER', 'WALL']:
                self.known_world[coord] = obj_type
            elif obj_type == 'EMPTY':
                self.known_world[coord] = 'SAFE'

        self.known_world[agent_pos] = 'VISITED'

    def _get_local_view_description(self):
        """Парсит обзор для LLM и памяти."""
        env = self.env.unwrapped
        agent_x, agent_y = map(int, env._agent_location)
        interesting_objects = []
        objects_dict = {}

        view_range = range(-2, 3)

        for dx in view_range:
            for dy in view_range:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = agent_x + dx, agent_y + dy

                if 0 <= nx < env.size and 0 <= ny < env.size:
                    cell_value = env.grid[nx, ny]

                    if cell_value == Places.NOTHING.value:
                        objects_dict[(nx, ny)] = 'EMPTY'
                        continue

                    if cell_value == Places.TARGET.value:
                        obj_type = "TARGET"
                    elif cell_value == Places.BOMB.value:
                        obj_type = "DANGER"
                    else:
                        continue

                    interesting_objects.append(f"({nx}, {ny}): {obj_type}")
                    objects_dict[(nx, ny)] = obj_type
                else:
                    # Стены по краям
                    is_honest_wall = (nx == -1 or nx == env.size or ny == -1 or ny == env.size)
                    if is_honest_wall:
                        interesting_objects.append(f"({nx}, {ny}): WALL")
                        objects_dict[(nx, ny)] = "WALL"

        if not interesting_objects:
            return "Open space around", objects_dict

        return ", ".join(interesting_objects), objects_dict

    def _get_map_string(self):
        """
        Создает ASCII представление карты для LLM.
        Символы:
        A - Agent
        T - Target
        B - Bomb (Danger)
        # - Wall
        V - Visited
        . - Unknown/Empty
        """
        env = self.env.unwrapped
        size = env.size

        # Создаем пустую сетку
        map_grid = [['.' for _ in range(size)] for _ in range(size)]

        # Заполняем известными данными
        for (x, y), cell_type in self.known_world.items():
            if 0 <= x <= size-1 and 0 <= y <= size-1:
                if cell_type == 'WALL':
                    map_grid[x][y] = '#'
                elif cell_type == 'DANGER':
                    map_grid[x][y] = 'B'
                elif cell_type == 'TARGET':
                    map_grid[x][y] = 'T'
                elif cell_type in ['VISITED', 'SAFE']:
                    map_grid[x][y] = 'V'

        # Ставим агента
        agent_pos = tuple(map(int, env._agent_location))
        map_grid[agent_pos[0]][agent_pos[1]] = 'A'

        # Преобразуем в строки
        return "\n".join(["".join(row) for row in map_grid])