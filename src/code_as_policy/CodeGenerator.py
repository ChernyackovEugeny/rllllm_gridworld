import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class CodeGenerator():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url='https://api.deepseek.com'
        )
        self.model_name = 'deepseek-chat'

        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def get_code(self, agent_pos, known_world, strategy, map_string=None):
        """
        Запрашивает у LLM код.
        """
        prompt = self._build_prompt(agent_pos, known_world, map_string)

        strategy = strategy or (
            "You are a navigation AI. Your goal is to reach the Target 'T'.\n"
            "You have access to the current map state and high-level movement functions.\n\n"
            
            "AVAILABLE VARIABLES:\n"
            "1. `agent_pos` (tuple): Your current (x, y) coordinates.\n"
            "2. `known_world` (dict): A dictionary mapping (x, y) coordinates to cell types.\n"
            "   Possible types: 'TARGET', 'DANGER', 'WALL', 'VISITED', 'SAFE'.\n"
            "   Example: `known_world.get((2, 3)) == 'TARGET'`.\n\n"
            
            "AVAILABLE FUNCTIONS:\n"
            "1. `go_to(coords)`: Returns the action (int) to move one step towards coords.\n"
            "   - `coords` must be a tuple (x, y).\n"
            "   - Returns None if the path is blocked.\n"
            "2. `get_nearest_unknown()`: Returns coordinates (x, y) of the nearest unexplored cell.\n\n"
            
            "STRATEGY RULES:\n"
            "1. Analyze `known_world` to find the 'TARGET' coordinates.\n"
            "2. If TARGET found: call `go_to(target_coords)`.\n"
            "3. If TARGET not found or path blocked: call `go_to(get_nearest_unknown())`.\n"
            "4. Return the result of the function call directly.\n\n"
            
            "Write ONLY the python function `decide_action()`."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": strategy
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            # --- СОБИРАЕМ МЕТРИКИ ТОКЕНОВ ---
            if hasattr(response, "usage") and response.usage is not None:
                usage = response.usage
                self.total_input_tokens += getattr(usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(usage, "completion_tokens", 0)

            content = response.choices[0].message.content

            # Иногда модели возвращают ```python ... ```, нужно почистить
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0].strip()

            return content

        except Exception as e:
            print(f"Planner API Error: {e}")
            # Возвращаем код действия по умолчанию при ошибке API
            return "def decide_action():\n\treturn 0"

    def fix_code(self, code, error):
        """
        Запрашивает у LLM зафиксить код.
        """
        prompt = (
            f"The following python code failed with error: {error}.\n"
            f"Code: {code}\n"
            "Rewrite the code to fix the error.Do not change the logic, just fix the error."""
        )

        strategy = (
            "You are a coding AI. Fix all the errors in code.\n"
            "AVAILABLE VARIABLES: `known_world` (dict), `agent_pos` (tuple).\n"
            "AVAILABLE FUNCTIONS: `go_to(coords)`, `get_nearest_unknown()`.\n\n"
            "Task: Fix the errors.\n"
            "Return ONLY the full function code. Do not explain.\n"
            "Example format:\n"
            "def decide_action():\n"
            "\treturn go_to((1, 2))"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": strategy
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            # --- СОБИРАЕМ МЕТРИКИ ТОКЕНОВ ---
            if hasattr(response, "usage") and response.usage is not None:
                usage = response.usage
                self.total_input_tokens += getattr(usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(usage, "completion_tokens", 0)

            content = response.choices[0].message.content

            # Иногда модели возвращают ```python ... ```, нужно почистить
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0].strip()

            return content

        except Exception as e:
            print(f"Fixer API Error: {e}")
            # Возвращаем код действия по умолчанию при ошибке API
            return "def decide_action():\n\treturn 0"

    def _build_prompt(self, agent_pos, known_world, map_string=None):
        # Преобразуем словарь known_world в читаемую строку
        # world_summary = [f"{coord}: {typ}" for coord, typ in known_world.items() if typ != 'SAFE']

        # Добавляем карту в промпт, если она есть
        map_context = f"Map Visualization:\n{map_string}" if map_string else ""

        return (
            f"Current Position: {agent_pos}\n"
            f"{map_context}\n"
            # f"Explored Map Summary: {json.dumps(world_summary)}\n"
            "Write a python function decide_action()."
        )
