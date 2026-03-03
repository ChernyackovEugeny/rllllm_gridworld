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
            "You are a Python navigator AI.\n"
            "Goal: Reach the target 'T'. Avoid bombs 'B' and walls '#'.\n"
            "Available functions: move_up(), move_down(), move_left(), move_right().\n"
            "Available variables: `known_world` (dict), `agent_pos` (tuple).\n\n"
            
            "RULES:\n"
            "1. DO NOT oscillate (move back and forth) if stuck.\n"
            "2. If the direct path to target is blocked by a wall, try to move SIDEWAYS to go around it.\n"
            "3. If target is above but blocked, do NOT just move down. Try moving left or right to find an opening.\n"
            "4. Prioritize moves that decrease distance to target, but avoid walls.\n\n"
            
            "Return ONLY the python function `decide_action()`."
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
            "You are a coding AI. Fix all the errors in code\n"
            "Available functions: move_up(), move_down(), move_left(), move_right().\n"
            "Available variables: `known_world` (dict), `agent_pos` (tuple).\n\n"
            "Task: Fix the errors.\n"
            "Return ONLY the full function code. Do not explain.\n"
            "Example format:\n"
            "def decide_action():\n"
            "\treturn move_right()"
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
        world_summary = [f"{coord}: {typ}" for coord, typ in known_world.items() if typ != 'SAFE']

        # Добавляем карту в промпт, если она есть
        map_context = f"Map Visualization:\n{map_string}" if map_string else ""

        return (
            f"Current Position: {agent_pos}\n"
            f"{map_context}\n"
            f"Explored Map Summary: {json.dumps(world_summary)}\n"
            "Write a python function decide_action()."
        )
