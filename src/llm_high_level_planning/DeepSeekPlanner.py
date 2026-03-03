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

    def get_next_goal(self, agent_pos, local_view, known_world, strategy):
        """
        Запрашивает у LLM следующую целевую точку.
        Возвращает tuple (x, y).
        """
        prompt = self._build_exploration_prompt(agent_pos, local_view, known_world)

        strategy = strategy or (
                            "You are a strategic exploration AI. "
                            "Your task is to choose a coordinate (x, y) to explore next. "
                            "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
                            "Prioritize unexplored areas."
                        )

        try:
            print(prompt)
            print(strategy)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": strategy
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000,
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

    def _build_exploration_prompt(self, agent_pos, local_view, known_world):
        # Преобразуем словарь known_world в читаемую строку
        # Фильтруем, чтобы не передавать пустые клетки, если они там есть
        world_summary = [f"{coord}: {typ}" for coord, typ in known_world.items() if typ != 'SAFE']

        return (
            f"Current Position: {agent_pos}\n"
            f"Current Vision: {local_view}\n"
            f"Explored Map Summary: {json.dumps(world_summary)}\n"
            "Choose a coordinate to explore."
        )