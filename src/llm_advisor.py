import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DeepSeekExplorer:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url='https://api.deepseek.com'
        )
        self.model_name = 'deepseek-chat'

    def get_exploration_action(self, agent_pos, local_view, visited_cells):
        """
        agent_pos: tuple (x, y) - где агент сейчас
        local_view: list/str - что он видит вокруг (например, "Стена слева")
        visited_cells: set/list - где он уже был
        """

        prompt = self._build_promt(agent_pos, local_view, visited_cells)

        try:
            # Делаем запрос к API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an exploration AI for a grid world. "
                            "Your goal is to explore unknown areas (cells you haven't visited yet). "
                            "Avoid walls. Return ONLY a JSON object with the key 'action' "
                            "values: 'UP', 'DOWN', 'LEFT', 'RIGHT'."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Немного креатива, но почти детерминизм
                max_tokens=50,  # Нам нужен короткий ответ
                response_format={"type": "json_object"}  # Гарантирует JSON на выходе
            )

            # Парсим ответ
            content = response.choices[0].message.content
            parsed = json.loads(content)
            action_text = parsed.get("action", "UP").upper()

            # Конвертируем текст в цифры для среды (0,1,2,3)
            mapping = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
            return mapping.get(action_text, 2)  # Если ошибка, идем вверх

        except Exception as e:
            print(f"API Error: {e}")
            return 2  # Fallback


    def _build_promt(self, agent_pos, local_view, visited_cells):
        visited_list = list(visited_cells)
        return (
            f"Current Position: {agent_pos}\n"
            f"Visited Cells: {visited_list}\n"
            f"Local Vision: {local_view}\n"
            "Which action leads to an unvisited cell? Avoid walls."
        )

class MockExplorer():
    """Эмулятор LLM для быстрой проверки пайплайна."""
    def get_exploration_action(self, agent_pos, local_view, visited_cells):
        # Просто возвращаем случайное действие мгновенно
        import random
        return random.randint(0, 3)

# explorer = DeepSeekExplorer()
#
# pos = (2, 2)
# view = "Empty N, Empty E, Wall S, Empty W"
# visited = {(2, 2), (2, 1), (1, 2)}
#
# action = explorer.get_exploration_action(pos, view, visited)
# print(f"LLM decided to go: {action}")