class Memory:
    def __init__(self, size):
        self.size = size
        self.known_world = {}

    def reset(self, agent_start_pos):
        """Полный сброс памяти."""
        self.known_world = {}
        self.known_world[agent_start_pos] = 'VISITED'

    def update(self, observation_data, agent_pos):
        """
        Обновляет память на основе данных от Perception.
        observation_data: словарь {(x,y): 'TYPE'}
        """
        # Обновляем известные клетки
        for coord, obj_type in observation_data.items():
            if obj_type in ['TARGET', 'DANGER', 'WALL']:
                self.known_world[coord] = obj_type
            elif obj_type == 'EMPTY':
                # Если клетка неизвестна, помечаем как SAFE.
                # Если уже известна (например, DANGER), лучше не менять (для надежности)
                if coord not in self.known_world:
                    self.known_world[coord] = 'SAFE'

        self.known_world[agent_pos] = 'VISITED'

    def get_context(self, agent_pos):
        """
        Возвращает данные, необходимые для Planner'а:
        - known_world (dict)
        - map_string (str)
        """
        map_string = self._render_map(agent_pos)
        return {
            "known_world": self.known_world,
            "map_string": map_string,
            "agent_pos": agent_pos
        }

    def _render_map(self, agent_pos):
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
        size = self.size
        map_grid = [['.' for _ in range(size)] for _ in range(size)]

        for (x, y), cell_type in self.known_world.items():
            if 0 <= x < size and 0 <= y < size:
                if cell_type == 'WALL':
                    map_grid[x][y] = '#'
                elif cell_type == 'DANGER':
                    map_grid[x][y] = 'B'
                elif cell_type == 'TARGET':
                    map_grid[x][y] = 'T'
                elif cell_type in ['VISITED', 'SAFE']:
                    map_grid[x][y] = 'V'

        # Ставим агента поверх всего
        if 0 <= agent_pos[0] < size and 0 <= agent_pos[1] < size:
            map_grid[agent_pos[0]][agent_pos[1]] = 'A'

        # Преобразуем в строки
        return "\n".join(["".join(row) for row in map_grid])