from src.environment.environment import Places


class Perception:
    def __init__(self, observation_size=5):
        # Вычисляем радиус обзора. Для 5x5 радиус = 2.
        assert observation_size % 2 == 1, "Observation size must be odd"
        self.view_radius = observation_size // 2

    def get_local_observations(self, env):
        """
        Сканирует окружающее пространство агента.
        Возвращает словарь: {(x, y): 'TYPE', ...}
        Использует глобальные координаты среды.
        """
        agent_x, agent_y = map(int, env._agent_location)
        objects_dict = {}

        view_range = range(-self.view_radius, self.view_radius+1)

        for dx in view_range:
            for dy in view_range:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = agent_x + dx, agent_y + dy

                if 0 <= nx < env.size and 0 <= ny < env.size:
                    cell_value = env.grid[nx, ny]
                    obj_type = None

                    if cell_value == Places.NOTHING.value:
                        obj_type = 'EMPTY'
                    elif cell_value == Places.TARGET.value:
                        obj_type = "TARGET"
                    elif cell_value == Places.BOMB.value:
                        obj_type = "DANGER"
                    else:
                        continue
                    objects_dict[(nx, ny)] = obj_type
                else:
                    # Стены по краям
                    is_honest_wall = (nx == -1 or nx == env.size or ny == -1 or ny == env.size)
                    if is_honest_wall:
                        objects_dict[(nx, ny)] = "WALL"

        return objects_dict

    def get_agent_position(self, env):
        """Вспомогательный метод для получения позиции."""
        return tuple(map(int, env.unwrapped._agent_location))