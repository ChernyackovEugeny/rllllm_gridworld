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

    def get_next_goal(self, agent_pos, known_world, step_count):
        """
        Запрашивает у LLM следующую целевую точку.
        Возвращает tuple (x, y).
        """
        # Если цель уже известна, LLM не нужен
        target_pos = self._find_target_in_dict(known_world)
        if target_pos:
            return target_pos

        prompt = self._build_exploration_prompt(agent_pos, known_world)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strategic exploration AI. "
                            "Your task is to choose a coordinate (x, y) to explore next. "
                            "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
                            "Prioritize unexplored areas."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            target = parsed.get("target_coordinate")

            if target and len(target) == 2:
                return tuple(target)

        except Exception as e:
            print(f"Planner API Error: {e}")

        return None

    def _find_target_in_dict(self, known_world):
        for coord, typ in known_world.items():
            if typ == 'TARGET':
                return coord
        return None

    def _build_exploration_prompt(self, agent_pos, known_world):
        # Формируем краткую сводку для LLM
        obstacles = [c for c, t in known_world.items() if t in ['WALL', 'DANGER']]
        return (
            f"Current Position: {agent_pos}\n"
            f"Known Obstacles: {json.dumps(obstacles)}\n"
            "Choose a coordinate to explore."
        )