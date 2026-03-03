from src.environment.environment import Actions, Places

class CodeExecutor():
    def __init__(self, env):
        self.env = env

    def execute_llm_code(self, code_string, known_world, agent_pos):
        # 1. Определяем "примитивы" (действия), которые может использовать код
        primitives = {
            'move_up': lambda: Actions.UP.value,
            'move_right': lambda: Actions.RIGHT.value,
            'move_down': lambda: Actions.DOWN.value,
            'move_left': lambda: Actions.LEFT.value,
            'known_world': known_world,
            'agent_pos': agent_pos
        }

        action_to_take = None

        # 2. Локальное пространство имен для exec
        local_scope = {}

        try:
            # 3. Выполняем код LLM
            # Мы ожидаем, что код объявит функцию decide_action()
            exec(code_string, primitives, local_scope)

            # 4. Вызываем функцию, которую написал LLM
            if 'decide_action' in local_scope:
                action_to_take = local_scope['decide_action']()
            else:
                raise ValueError("LLM didn't define decide_action()")

        except Exception as e:
            print(f"CRASH! Code failed: {e}")
            # VOYAGER MOMENT: Отправляем ошибку e обратно в LLM: "Fix this code..."
            return None, str(e)

        if callable(action_to_take):
            action_to_take = action_to_take()

        if not isinstance(action_to_take, int):
            raise ValueError(f"Action must be int, got {type(action_to_take)}")

        return action_to_take, ''