import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

# --- КОНФИГУРАЦИЯ ---

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url='https://api.deepseek.com'
)
MODEL_NAME = 'deepseek-chat'

# 1. Стратегии промптинга для High-Level Planner
PROMPT_STRATEGIES = {
    "v2_base": {
        "system": (
            "You are a strategic exploration AI. "
            "Your task is to choose a coordinate (x, y) to explore next. "
            "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
            "Prioritize unexplored areas. "
            "Avoid coordinates marked as BOMB or WALL. "
            "If TARGET is known, return its coordinate."
        ),
        "description": "Базовая стратегия для выбора подцели"
    },
    "v2_frontier_focus": {
        "system": (
            "You are a strategic navigation AI. "
            "Analyze the 'Known Obstacles' and 'Current Position'. "
            "1. If TARGET is known, go to it. "
            "2. Otherwise, identify 'Frontier Cells' (cells adjacent to known safe areas but unexplored). "
            "3. Select the closest Frontier Cell that is not blocked by obstacles. "
            "Return JSON: {'target_coordinate': [x, y]}."
        ),
        "description": "Акцент на границах исследования (Frontier-based)"
    }
}

# 2. Сценарии тестирования
# expected_valid_coords: Список координат, которые считаются "хорошим ответом".
# Если модель вернет что-то другое (но тоже валидное), это можно анализировать вручную.
TEST_SCENARIOS = [
    # --- СЦЕНАРИИ ИССЛЕДОВАНИЯ (Target Unknown) ---
    {
        "id": 1, "name": "Start Exploration (Center)",
        "agent_pos": (5, 5),
        "known_obstacles": [],  # Нет препятствий
        "known_world_state": "Empty grid 10x10, visited only (5,5)",
        "expected_valid_coords": [(4, 5), (6, 5), (5, 4), (5, 6)],  # Любой сосед
        "comment": "Модель должна предложить любую соседнюю клетку для начала исследования."
    },
    {
        "id": 2, "name": "Wall Avoidance (North)",
        "agent_pos": (1, 5),
        "known_obstacles": [(0, 5)],  # Стена сверху (x=0)
        "known_world_state": "Wall to the North.",
        "expected_valid_coords": [(1, 4), (1, 6), (2, 5)],  # Не идти вверх
        "comment": "Не предлагать (0,5) или (-1,5)."
    },
    {
        "id": 3, "name": "Bomb Avoidance (East)",
        "agent_pos": (5, 5),
        "known_obstacles": [(5, 6)],  # Бомба справа
        "known_world_state": "Bomb at (5,6).",
        "expected_valid_coords": [(4, 5), (6, 5), (5, 4)],
        "comment": "Избегать координаты бомбы."
    },
    {
        "id": 4, "name": "Corner Trap",
        "agent_pos": (0, 0),
        "known_obstacles": [(0, 1), (1, 0)],  # Заблокирован справа и снизу
        "known_world_state": "Walls to the right and down.",
        "expected_valid_coords": [],  # Некуда идти (тупик)
        "comment": "Модель должна предложить что-то, даже если это тупик, или выдать ошибку. В идеале - не в стены."
    },
    {
        "id": 5, "name": "Frontier Selection",
        "agent_pos": (3, 3),
        "known_obstacles": [(2, 3), (3, 2), (4, 3)],  # Блокировка сверху, слева, снизу
        "known_world_state": "Only way is Right (3,4).",
        "expected_valid_coords": [(3, 4)],
        "comment": "Единственный путь развития - вправо."
    },

    # --- СЦЕНАРИИ ДОСТИЖЕНИЯ ЦЕЛИ (Target Known) ---
    # Важный момент: Python-код планировщика сам перехватывает цель.
    # Но здесь мы тестируем именно ПРОМПТ, спросим модель, что бы она выбрала, если бы не знала, что это цель (проверка логики).
    # Или укажем цель в описании.
    {
        "id": 6, "name": "Target Nearby (Direct Move)",
        "agent_pos": (5, 5),
        "known_obstacles": [],
        "target_known_at": (5, 6),
        "known_world_state": "Target is at (5,6).",
        "expected_valid_coords": [(5, 6)],
        "comment": "Если цель известна, модель должна указать на неё."
    },
    {
        "id": 7, "name": "Target Blocked (Bypass Logic)",
        "agent_pos": (5, 5),
        "known_obstacles": [(5, 6)],  # Прямой путь заблокирован
        "target_known_at": (5, 7),  # Цель за бомбой
        "known_world_state": "Target (5,7), Bomb at (5,6). Need detour.",
        "expected_valid_coords": [(4, 6), (6, 6)],  # Попытка обойти сверху или снизу
        "comment": "Модель не должна предлагать (5,6). Должна предложить обходной путь."
    },
    {
        "id": 8, "name": "Long Distance Target",
        "agent_pos": (0, 0),
        "known_obstacles": [(0, 1)],  # Стена справа
        "target_known_at": (9, 9),
        "known_world_state": "Target far away. Wall to the right.",
        "expected_valid_coords": [(1, 0)],  # Идти вниз, так как справа стена
        "comment": "Стратегическое движение в сторону цели, обходя препятствие."
    },

    # --- СЛОЖНЫЕ СЦЕНАРИИ ---
    {
        "id": 9, "name": "Zig-Zag Start",
        "agent_pos": (5, 5),
        "known_obstacles": [(4, 5), (5, 4), (6, 5)],  # Сверху, слева, снизу
        "target_known_at": None,
        "known_world_state": "Corridor to the right.",
        "expected_valid_coords": [(5, 6)],
        "comment": "Единственное направление - вправо."
    },
    {
        "id": 10, "name": "Dead End Escape",
        "agent_pos": (2, 2),
        "known_obstacles": [(1, 2), (2, 3), (3, 2), (2, 1)],  # Полное окружение
        "target_known_at": None,
        "known_world_state": "Completely surrounded.",
        "expected_valid_coords": [],  # Идеально - вернуться (но памяти нет). Модель в растерянности.
        "comment": "Проверка поведения в тупике."
    }
]


# --- ФУНКЦИИ ---

def build_user_prompt(strategy_name, scenario):
    """Собирает User Prompt в формате DeepSeekPlanner."""

    # Базовая часть
    base = f"Current Position: {scenario['agent_pos']}\n"

    # Препятствия
    obstacles = scenario.get('known_obstacles', [])
    base += f"Known Obstacles: {json.dumps(obstacles)}\n"

    # Если цель известна (добавляем в промпт)
    if scenario.get('target_known_at'):
        base += f"Target Location: {scenario['target_known_at']}\n"

    base += "Choose a coordinate to explore."
    return base


def get_model_response(strategy_name, scenario):
    """Запрос к API."""
    strategy = PROMPT_STRATEGIES[strategy_name]
    user_prompt = build_user_prompt(strategy_name, scenario)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": strategy["system"]},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        # Новый формат ответа
        target = parsed.get("target_coordinate")

        # Нормализация ответа
        if target and isinstance(target, list) and len(target) == 2:
            result_str = tuple(target)
        else:
            result_str = f"INVALID FORMAT: {content}"

        return result_str, content, user_prompt

    except Exception as e:
        return f"ERROR: {str(e)}", "", user_prompt


def validate_response(response_coord, scenario):
    """Проверка правильности выбора координаты."""
    if isinstance(response_coord, str):  # Error case
        return "ERROR"

    # 1. Попадание в список ожидаемых
    if response_coord in scenario['expected_valid_coords']:
        return "PASS (Expected)"

    # 2. Проверка на безопасности (не в стену/бомбу)
    if response_coord in scenario['known_obstacles']:
        return "FAIL (Dangerous)"

    # 3. Если список ожидаемых пуст (тупик), любой безопасный ход - ок
    if not scenario['expected_valid_coords']:
        # Простая проверка на адекватность (например, не далеко ли?)
        ax, ay = scenario['agent_pos']
        tx, ty = response_coord
        dist = abs(ax - tx) + abs(ay - ty)
        if dist <= 1: return "PASS (Alternative)"
        return "FAIL (Weird Logic)"

    # 4. Логичный альтернативный ход
    # Если модель предложила ход, которого нет в списке, но он безопасен
    # Считаем это проходным, но помечаем как Alternative
    return "PASS (Alternative)"


# --- ЗАПУСК ТЕСТОВ ---

results = []
print(f"Starting High-Level Planner Tests for {len(PROMPT_STRATEGIES)} strategies...")

for strat_name in PROMPT_STRATEGIES:
    print(f"\nTesting Strategy: {strat_name}...")
    for scenario in TEST_SCENARIOS:
        result_coord, raw_json, prompt_used = get_model_response(strat_name, scenario)

        validation = validate_response(result_coord, scenario)

        results.append({
            "Strategy": strat_name,
            "Scenario ID": scenario['id'],
            "Scenario Name": scenario['name'],
            "Agent Pos": scenario['agent_pos'],
            "Known Obstacles": scenario['known_obstacles'],
            "Target Known": scenario.get('target_known_at', 'Unknown'),
            "Expected (Valid)": scenario['expected_valid_coords'],
            "Model Choice": result_coord,
            "Validation": validation,
            "Raw JSON": raw_json,
            "Prompt Used": prompt_used
        })
        time.sleep(0.5)

# --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---

df = pd.DataFrame(results)

# Сводка
print("\n=== SUMMARY ===")
# Считаем PASS (включая Expected и Alternative)
summary = df.groupby("Strategy")['Validation'].apply(
    lambda x: x.str.startswith('PASS').sum()
).reset_index()
summary.columns = ["Strategy", "Passed Tests"]
print(summary)

filename = "llm_planner_test_results.csv"
df.to_csv(filename, index=False)
print(f"\nDetailed results saved to {filename}")

# Вывод ошибок
failures = df[~df['Validation'].str.startswith('PASS')]
if not failures.empty:
    print("\n--- FAILED CASES ---")
    for idx, row in failures.iterrows():
        print(f"Scenario: {row['Scenario Name']}")
        print(f"  Obstacles: {row['Known Obstacles']}")
        print(f"  Expected: {row['Expected (Valid)']}")
        print(f"  Got: {row['Model Choice']} ({row['Validation']})")
        print("-" * 30)