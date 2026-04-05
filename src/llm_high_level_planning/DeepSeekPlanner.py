import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DeepSeekPlanner:
    """
    Отвечает за HIGH-LEVEL планирование.
    Возвращает не действие, а целевую точку (subgoal).
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url='https://api.deepseek.com'
        )
        self.model_name = 'deepseek-chat'

        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def get_next_goal(self, agent_pos, local_view, known_world, strategy, map_string=None):
        """
        Запрашивает у LLM следующую целевую точку.
        Возвращает tuple (x, y).

        agent_pos:  tuple (x, y) — текущая позиция
        local_view: str          — что видно прямо сейчас (радиус 2)
        known_world: dict        — суммарная карта мира {(x,y): 'TYPE'}
        map_string: str          — ASCII-карта известного мира
        """
        prompt = self._build_exploration_prompt(agent_pos, known_world, map_string)

        strategy = strategy or (
            "You are a grid navigation AI. Goal: reach TARGET (T), avoid DANGER (B).\n\n"
            "MAP LEGEND: A=Agent, T=Target, B=Bomb/Danger, #=Wall, V=Visited/Safe, .=Unknown\n"
            "AXES: Row 0 is TOP. Coordinate (x, y) = (row, column). "
            "x increases downward, y increases rightward.\n\n"
            "RULES (strict priority order):\n"
            "1. If T is visible on the map → return its exact coordinates immediately.\n"
            "2. Pick an unexplored (.) cell on the frontier (adjacent to a V or A cell).\n"
            "3. Among frontier cells, prefer the one with smallest Manhattan distance to A.\n"
            "4. NEVER return a B or # cell.\n"
            "5. NEVER return a V cell if any unexplored (.) cells remain.\n\n"
            "Return ONLY JSON: {\"target_coordinate\": [x, y]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": strategy},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            # --- СОБИРАЕМ МЕТРИКИ ТОКЕНОВ ---
            if hasattr(response, "usage") and response.usage is not None:
                usage = response.usage
                self.total_input_tokens += getattr(usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(usage, "completion_tokens", 0)

            content = response.choices[0].message.content
            parsed = json.loads(content)
            target = parsed.get("target_coordinate")

            if target and len(target) == 2:
                return tuple(target)

        except Exception as e:
            print(f"Planner API Error: {e}")

        return None

    def _build_exploration_prompt(self, agent_pos, known_world, map_string=None):
        if map_string:
            return (
                f"Current Position (A): {agent_pos}\n"
                f"Map:\n{map_string}\n"
                "Choose a coordinate to explore next."
            )

        # Fallback: список координат (если карта не передана)
        world_summary = [f"{coord}: {typ}" for coord, typ in known_world.items() if typ != 'SAFE']
        return (
            f"Current Position: {agent_pos}\n"
            f"Explored Map Summary: {json.dumps(world_summary)}\n"
            "Choose a coordinate to explore."
        )