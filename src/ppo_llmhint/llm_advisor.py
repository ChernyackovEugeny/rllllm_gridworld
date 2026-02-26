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

    def get_exploration_action(self, agent_pos, local_view, known_world):
        """
        agent_pos: tuple (x, y) - текущая позиция
        local_view: str - что видно прямо сейчас (радиус 2)
        known_world: dict - суммарная карта мира {(x,y): 'TYPE'}
        """

        prompt = self._build_promt(agent_pos, local_view, known_world)

        try:
            # Делаем запрос к API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                                "You are an expert pathfinding AI.\n"
                                "OBJECTIVE: Reach TARGET. Avoid DANGER.\n"
                                "AXIS DEFINITION: X is Vertical (UP/DOWN), Y is Horizontal (LEFT/RIGHT).\n"
                                "- UP decreases X (x-1). DOWN increases X (x+1).\n"
                                "- LEFT decreases Y (y-1). RIGHT increases Y (y+1).\n\n"
                                
                                "LOGIC PROCESS:\n"
                                "1. Identify the direction towards the TARGET. If there is no TARGET yet, your TARGET is to explore unvisited cells and directions.\n"
                                "2. Check if that direction is safe (not DANGER/WALL).\n"
                                "3. IF BLOCKED: Execute a 'Sidestep' maneuver. \n"
                                "   - If blocked moving LEFT/RIGHT (horizontally), you MUST try UP or DOWN first.\n"
                                "   - If blocked moving UP/DOWN (vertically), you MUST try LEFT or RIGHT first.\n"
                                "   - Only move AWAY from the target (backtrack) if no sidestep is possible.\n"
                                "4. Avoid VISITED cells if unexplored options exist.\n\n"
                                
                                "Return JSON with keys 'reasoning' and 'action'.\n"
                                "'action' values: 'UP', 'DOWN', 'LEFT', 'RIGHT'."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Немного креатива, но почти детерминизм
                max_tokens=1000,  # Нам нужен короткий ответ
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

    def _build_promt(self, agent_pos, local_view, known_world):
        # Преобразуем словарь known_world в читаемую строку
        # Фильтруем, чтобы не передавать пустые клетки, если они там есть
        world_summary = [f"{coord}: {typ}" for coord, typ in known_world.items() if typ != 'EMPTY']

        return (
            f"Current Position: {agent_pos}\n"
            f"Current Vision: {local_view}\n"
            f"Explored Map Summary: {json.dumps(world_summary)}\n"
            "Analyze and choose the best action."
        )
