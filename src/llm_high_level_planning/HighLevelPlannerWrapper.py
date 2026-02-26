import gymnasium as gym
from collections import deque

from src.environment.environment import Places, Actions
from src.llm_high_level_planning.DeepSeekPlanner import DeepSeekPlanner

class HighLevelPlannerWrapper(gym.Wrapper):
    def __init__(self, env, plan_frequency=10):
        super().__init__(env)

        # Инициализация LLM
        self.planner = DeepSeekPlanner()

        # Параметры планирования
        self.plan_frequency = plan_frequency
        self.current_goal = None  # Глобальная цель от LLM
        self.current_path = []  # Очередь действий
        self.steps_since_plan = 0

        # Счетчики
        self.total_calls = 0

        # Память карты
        self.known_world = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Сброс памяти мира
        self.known_world = {}

        # Сброс плана
        self.current_goal = None
        self.current_path = []
        self.steps_since_plan = 0

        # Сразу добавляем стартовую позицию в память
        start_pos = tuple(self.env.unwrapped._agent_location)
        self.known_world[start_pos] = 'VISITED'

        # Обновляем память тем, что видно на старте
        self._update_memory()

        return obs, info

    def step(self, action):
        # --- 1. ОБНОВЛЕНИЕ ПАМЯТИ ---
        # Сначала смотрим, что вокруг, прежде чем принимать решение
        self._update_memory()
        agent_pos = tuple(self.env.unwrapped._agent_location)

        # --- 2. ПРИНЯТИЕ РЕШЕНИЯ (LLM + BFS) ---

        # Условия для запроса НОВОЙ цели к LLM:
        # - Нет цели
        # - Путь закончился
        # - Прошло N шагов (время обновить план)
        # - Цель недействительна (стала стеной/бомбой)
        # - Мы пришли в цель
        need_new_plan = (
                not self.current_goal or
                not self.current_path or
                self.steps_since_plan >= self.plan_frequency or
                not self._is_goal_valid()
        )

        # Если мы уже пришли в цель, точно нужен новый план
        if self.current_goal and agent_pos == self.current_goal:
            need_new_plan = True
            self.current_goal = None

        if need_new_plan:
            self.steps_since_plan = 0
            self.total_calls += 1

            # Запрос к LLM
            llm_goal = self.planner.get_next_goal(agent_pos, self.known_world, len(self.known_world))

            if llm_goal:
                self.current_goal = llm_goal
                # Строим маршрут локально (BFS)
                self.current_path = self._calculate_path_to_goal(agent_pos, llm_goal)
                print(f"[Planner] New goal: {llm_goal}, path len: {len(self.current_path)}")
            else:
                # Fallback логика
                self.current_goal = self._get_random_frontier(agent_pos)
                # Если фронт найден, путь строится внутри _get_random_frontier или тут:
                if self.current_goal:
                    self.current_path = self._calculate_path_to_goal(agent_pos, self.current_goal)

        # --- 3. ВЫБОР ДЕЙСТВИЯ ---
        if self.current_path:
            action = self.current_path.pop(0)
            self.steps_since_plan += 1
        else:
            # Если пути нет (тупик), делаем случайный шаг
            action = self.env.action_space.sample()

        # --- 4. ВЫПОЛНЕНИЕ ШАГА ---
        # Передаем вычисленное действие в среду
        return super().step(action)

    # --- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ---

    def _update_memory(self):
        """Обновляет known_world на основе текущего обзора."""
        _, current_view_objects = self._get_local_view_description()
        agent_pos = tuple(self.env.unwrapped._agent_location)

        for coord, obj_type in current_view_objects.items():
            if obj_type in ['TARGET', 'DANGER', 'WALL']:
                self.known_world[coord] = obj_type
            elif obj_type == 'EMPTY':
                # ВАЖНО: Для BFS нам нужно знать, где проходимо
                self.known_world[coord] = 'SAFE'

        self.known_world[agent_pos] = 'VISITED'

    def _is_goal_valid(self):
        if not self.current_goal: return False
        val = self.known_world.get(self.current_goal)
        # Нельзя идти в стену или бомбу
        return val not in ['WALL', 'DANGER']

    def _calculate_path_to_goal(self, start_pos, goal_pos):
        """BFS поиск пути по ИЗВЕСТНОЙ карте."""
        queue = deque([start_pos])
        visited = {start_pos: None}

        while queue:
            curr = queue.popleft()

            if curr == goal_pos:
                # Восстанавливаем путь
                path = []
                while curr != start_pos:
                    prev = visited[curr]
                    diff = (curr[0] - prev[0], curr[1] - prev[1])
                    action = self._diff_to_action(diff)
                    path.append(action)
                    curr = prev
                return path[::-1]  # Разворачиваем

            # Перебор соседей
            for action, direction in self.env.unwrapped._action_to_direction.items():
                nx, ny = curr[0] + direction[0], curr[1] + direction[1]
                neighbor = (nx, ny)

                cell_type = self.known_world.get(neighbor)
                # Идем только по безопасному или неизвестному
                is_safe = cell_type in ['SAFE', 'VISITED', 'TARGET']
                is_unknown = neighbor not in self.known_world

                if (is_safe or is_unknown) and neighbor not in visited:
                    if cell_type not in ['DANGER', 'WALL']:
                        visited[neighbor] = curr
                        queue.append(neighbor)
        return []

    def _diff_to_action(self, diff):
        if diff == (0, 1): return Actions.RIGHT.value
        if diff == (0, -1): return Actions.LEFT.value
        if diff == (-1, 0): return Actions.UP.value
        if diff == (1, 0): return Actions.DOWN.value
        return 0

    def _get_random_frontier(self, agent_pos):
        """Fallback: ищем границу известного мира."""
        candidates = []
        for coord, typ in self.known_world.items():
            if typ in ['SAFE', 'VISITED']:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (coord[0] + dx, coord[1] + dy)
                    if neighbor not in self.known_world:
                        candidates.append(neighbor)

        if candidates:
            # Ищем ближайшую
            candidates.sort(key=lambda x: abs(x[0] - agent_pos[0]) + abs(x[1] - agent_pos[1]))
            return candidates[0]
        return None

    def _get_local_view_description(self):
        """Парсит обзор для LLM и памяти."""
        env = self.env.unwrapped
        agent_x, agent_y = env._agent_location
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