from src.environment.environment import Actions, Places
from collections import deque

def get_path_to_target(start_pos, target_pos, known_world, size):
    """
    BFS поиск пути.
    known_world: словарь {(x,y): type}.
    Игнорируем неизвестные клетки (чтобы агент мог исследовать)
    и избегаем стен/бомб.
    """
    queue = deque([start_pos])
    visited = {start_pos}
    parent = {start_pos: None}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    path_found = False
    while queue:
        cur = queue.popleft()
        if cur == target_pos:
            path_found = True
            break

        for dx, dy in directions:
            nx, ny = cur[0] + dx, cur[1] + dy
            neighbor = (nx, ny)

            # Проверка границ
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            if known_world.get(neighbor) in ['WALL', 'DANGER']:
                continue
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = cur
    if not path_found:
        return None
    path = []
    cur = target_pos
    while cur != start_pos:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]

def get_next_move_direction(start_pos, next_pos):
    """Преобразует разницу координат в действие Action"""
    dx = next_pos[0] - start_pos[0]
    dy = next_pos[1] - start_pos[1]

    if dx == 1: return Actions.DOWN.value
    if dx == -1: return Actions.UP.value
    if dy == 1: return Actions.RIGHT.value
    if dy == -1: return Actions.LEFT.value
    return None

class CodeExecutor():
    def __init__(self, env):
        self.env = env

    def execute_llm_code(self, code_string, known_world, agent_pos):
        # Создаем функции-обертки, которые "знают" текущий контекст
        def go_to(target_coord):
            """Говорит агенту идти в точку. Возвращает действие или None, если нельзя пройти."""
            # Преобразуем numpy array в tuple, если нужно
            target_tuple = tuple(target_coord)

            path = get_path_to_target(agent_pos, target_tuple, known_world, self.env.size)
            if path:
                return get_next_move_direction(agent_pos, path[0])
            return None  # Путь заблокирован

        def get_nearest_unknown():
            """Ищет ближайшую неизвестную клетку для исследования"""
            q = deque([agent_pos])
            visited = {agent_pos}
            while q:
                curr = q.popleft()
                # Если клетка неизвестна - это цель для исследования
                if curr not in known_world:
                    return curr
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_c = (curr[0] + dx, curr[1] + dy)
                    if 0 <= next_c[0] < self.env.size and 0 <= next_c[1] < self.env.size:
                        if next_c not in visited:
                            cell = known_world.get(next_c)
                            if cell not in ['WALL', 'DANGER']:
                                visited.add(next_c)
                                q.append(next_c)
            return None

        # Определяем "примитивы" (действия), которые может использовать код
        primitives = {
            "__builtins__": {},
            'go_to': go_to,
            'get_nearest_unknown': get_nearest_unknown,
            'agent_pos': agent_pos,
            'known_world': known_world
        }

        action_to_take = None

        # Локальное пространство имен для exec
        local_scope = {}

        try:
            # Выполняем код LLM
            # Мы ожидаем, что код объявит функцию decide_action()
            exec(code_string, primitives, local_scope)

            if "decide_action" not in local_scope:
                return None, False, "decide_action() not defined"

            result = local_scope["decide_action"]()

            if not isinstance(result, tuple):
                return None, False, "decide_action must return (action, is_done)"
            if len(result) != 2:
                return None, False, "decide_action must return exactly 2 values"

            action_to_take, is_done = result
            return action_to_take, is_done, None

        except Exception as e:
            print(f"CRASH! Code failed: {e}")
            # Отправляем ошибку e обратно в LLM: "Fix this code..."
            return None, None, str(e)