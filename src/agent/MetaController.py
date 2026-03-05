from src.environment.environment import Actions

class MetaController:
    def __init__(self, skill_manager, planner, executor, strategy=None, max_skill_steps=100, max_fix_retries=3):
        self.skill_manager = skill_manager
        self.planner = planner  # CodeGenerator
        self.executor = executor
        self.strategy = strategy
        self.max_fix_retries = max_fix_retries

        # Execution Loop (Skill Horizon)
        self.current_skill_code = None
        self.current_skill_steps = 0
        self.max_skill_steps = max_skill_steps
        self.current_skill_id = None

    def decide_action(self, context):
        """
        Основной цикл принятия решения.
        Context содержит: known_world, map_string, agent_pos
        """
        known_world = context['known_world']
        map_string = context['map_string']
        agent_pos = context['agent_pos']

        # 1. Формируем промпт для поиска навыка
        situation_summary = (
            f"Current Map (A=Agent, T=Target, B=Bomb, V=Visited, .=Unknown):\n{map_string}\n"
            f"Current Position: {agent_pos}"
        )

        # действуем в соответствии со скилом
        if self.current_skill_code is not None:
            action, is_done, error = self.executor.execute_llm_code(
                self.current_skill_code,
                context["known_world"],
                context["agent_pos"]
            )
            self.current_skill_steps += 1

            if error:
                print(f"⚠️ Error during skill execution: {error}. Aborting skill.")
                self.current_skill_code = None
                self.current_skill_id = None
                self.current_skill_steps = 0
                return Actions.UP.value, None, "Skill Crash", None

            if self.current_skill_steps >= self.max_skill_steps or is_done:
                print(f"✅ Skill finished (Steps: {self.current_skill_steps}, Done: {is_done})")
                self.current_skill_code = None
                self.current_skill_id = None
                self.current_skill_steps = 0

            print('reusing skill without llm, skill id:', self.current_skill_id)
            return action, self.current_skill_code, "Skill Execution", self.current_skill_id

        # выбираем скил
        # --- ПОПЫТКА 1: Использовать существующий навык ---
        skill_code, skill_id = self.skill_manager.get_relevant_skill_code(situation_summary)

        if skill_code:
            action, is_done, error = self.executor.execute_llm_code(skill_code, known_world, agent_pos)

            if action is None:
                error = "Error: go_to() failed or returned None. Path might be blocked."

            if not error:
                # Успех! Возвращаем действие и источник
                self.current_skill_code = skill_code
                self.current_skill_steps = 1
                self.current_skill_id = skill_id
                return action, skill_code, "Skill", skill_id
            else:
                print("⚠️ Retrieved skill failed, falling back to generation...")

        # --- ПОПЫТКА 2: Генерация нового кода ---
        print('🧠 Generating new code...')

        # Генерируем код
        code_data = self.planner.get_code(agent_pos, known_world, self.strategy, map_string)

        # Пытаемся исполнить
        action, is_done, error = self.executor.execute_llm_code(code_data, known_world, agent_pos)

        if action is None:
            error = "Error: go_to() failed or returned None."

        # Цикл исправления ошибок (Iterative Repair)
        retries = 0
        while error and retries < self.max_fix_retries:
            print(f"Fixing code... (attempt {retries + 1})")
            code_data = self.planner.fix_code(code_data, error)
            action, is_done, error = self.executor.execute_llm_code(code_data, known_world, agent_pos)
            retries += 1

        if error:
            # Полный провал, используем fallback
            print('fallback to "UP" action')
            action = Actions.UP.value
            return action, None, "Fallback", None

        if not isinstance(action, int):
            print("Invalid action, fallback")
            action = Actions.UP.value
            return action, None, "Fallback", None

        # Успешная генерация
        self.current_skill_code = code_data
        self.current_skill_steps = 1
        self.current_skill_id = None
        return action, code_data, "New Generation", None

    def learn(self, source, code_used, skill_id, reward):
        """Обучение после получения награды."""
        if source == "New Generation" and code_used:
            new_skill_id = self.skill_manager.critique_and_save(code_used, reward)
            # Обновляем ID текущего активного скилла
            if new_skill_id is not None:
                self.current_skill_id = new_skill_id
        elif source == "Skill Execution" and skill_id is not None:
            self.skill_manager.update_skill_data(skill_id, reward)