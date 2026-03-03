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
    # "v2_base": {
    #     "system": (
    #         "You are a strategic exploration AI. "
    #         "Your task is to choose a coordinate (x, y) to explore next. "
    #         "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
    #         "Prioritize unexplored areas. "
    #         "Avoid coordinates marked as BOMB or WALL. "
    #         "If TARGET is known, return its coordinate."
    #     ),
    #     "description": "Базовая стратегия для выбора подцели"
    # },
    "base": {
        "system": (
            "You are a strategic exploration AI. "
            "Your task is to choose a coordinate (x, y) to explore next. "
            "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
            "Prioritize unexplored areas."
        ),
        "description": "супер базовая"
    }
}

# 2. Сценарии тестирования
# expected_valid_coords: Список координат, которые считаются "хорошим ответом".
# Если модель вернет что-то другое (но тоже валидное), это можно анализировать вручную.
TEST_SCENARIOS = [
    # ==========================================
    # КАТЕГОРИЯ 1: БАЗОВОЕ ИССЛЕДОВАНИЕ (Navigation)
    # ==========================================
    {
        "id": 1, "name": "Basic Expansion (Center Start)",
        "agent_pos": (5, 5),
        "known_obstacles": [],
        "known_visited": [(5, 5)],
        "target_known_at": None,
        "comment": "Должен выбрать любую соседнюю клетку для старта. 49282",
        "expected_valid_coords": [(4, 5), (6, 5), (5, 4), (5, 6)]
    },
    {
        "id": 2, "name": "Avoiding Local Wall",
        "agent_pos": (1, 5),
        "known_obstacles": [(0, 5)], # Стена сверху
        "known_visited": [(1, 5), (2, 5)], # Пришел снизу
        "target_known_at": None,
        "comment": "Не идти вверх. Идти влево, вправо или вниз.",
        "expected_valid_coords": [(1, 4), (1, 6), (2, 5)]
    },
    {
        "id": 3, "name": "Simple Bomb Bypass",
        "agent_pos": (5, 5),
        "known_obstacles": [(5, 6)], # Бомба справа
        "known_visited": [(5, 5)],
        "target_known_at": None,
        "comment": "Не идти вправо. Выбрать любой другой путь.",
        "expected_valid_coords": [(4, 5), (6, 5), (5, 4)]
    },
    {
        "id": 4, "name": "Corner Deadlock",
        "agent_pos": (0, 0),
        "known_obstacles": [(0, 1), (1, 0)], # Стены справа и снизу
        "known_visited": [(0, 0)],
        "target_known_at": None,
        "comment": "Тупик. Модель может растеряться, но не должна предлагать стены.",
        "expected_valid_coords": [] # Сложный кейс: нет валидных путей, ожидается fallback или ошибка
    },
    {
        "id": 5, "name": "Narrow Corridor",
        "agent_pos": (5, 2),
        "known_obstacles": [(4, 2), (6, 2), (5, 1), (5, 3)], # Туннель 1x1, открыт только вверх/вниз? Нет, туннель по Y.
        # Исправим: стены слева (4,2) и справа (6,2). Путь впереди (5,3) свободен? Или назад?
        # Давай сделаем стены слева/справа/сзади.
        "known_obstacles": [(4, 2), (6, 2), (5, 1)],
        "known_visited": [(5, 1), (5, 2)],
        "target_known_at": None,
        "comment": "Единственный путь - вперед по коридору.",
        "expected_valid_coords": [(5, 3)]
    },

    # ==========================================
    # КАТЕГОРИЯ 2: СТРАТЕГИЯ ИССЛЕДОВАНИЯ (Frontier Logic)
    # ==========================================
    {
        "id": 6, "name": "Closest Frontier Priority",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5), (5, 4), (5, 3)], # Пришел слева
        "known_obstacles": [(5, 6)], # Справа стена
        "target_known_at": None,
        "comment": "Слева исследовано. Справа стена. Лучший выбор - вверх или вниз (ближайшие границы).",
        "expected_valid_coords": [(4, 5), (6, 5)]
    },
    {
        "id": 7, "name": "Avoiding Backtracking",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5), (5, 4), (6, 4), (4, 4)], # Большой посещенный остров слева
        "known_obstacles": [],
        "target_known_at": None,
        "comment": "Не идти влево (посещено). Идти вправо (неизвестно).",
        "expected_valid_coords": [(5, 6)]
    },
    {
        "id": 8, "name": "Large Unexplored Area Bias",
        "agent_pos": (2, 5),
        "known_visited": [(0, 5), (1, 5), (2, 5)], # Пришел сверху
        "known_obstacles": [(2, 6)], # Справа стена
        "target_known_at": None,
        "comment": "Снизу огромная неисследованная область. Модель должна выбрать DOWN.",
        "expected_valid_coords": [(3, 5)]
    },
    {
        "id": 9, "name": "Maze Entrance Decision",
        "agent_pos": (1, 0),
        "known_visited": [(0, 0), (1, 0)],
        "known_obstacles": [(0, 1), (1, 1)], # Вход в лабиринт заблокирован?
        "target_known_at": None,
        "comment": "Идти вниз (2,0), так как справа стены.",
        "expected_valid_coords": [(2, 0)]
    },
    {
        "id": 10, "name": "The Pivot Point",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5), (5, 6), (5, 7)], # Пошел вправо, уперся в...
        "known_obstacles": [(5, 8)], # ...стену
        "target_known_at": None,
        "comment": "Идти вправо нельзя. Нужно разворачиваться (влево) или обходить (вверх/вниз).",
        "expected_valid_coords": [(5, 4), (4, 5), (6, 5)]
    },

    # ==========================================
    # КАТЕГОРИЯ 3: ЦЕЛЕВОЕ НАПРАВЛЕНИЕ (Target Known)
    # ==========================================
    {
        "id": 11, "name": "Direct Target Approach",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5)],
        "known_obstacles": [],
        "target_known_at": (5, 6), # Цель справа
        "comment": "Цель известна и доступна. Идти прямо к ней.",
        "expected_valid_coords": [(5, 6)]
    },
    {
        "id": 12, "name": "Target Blocked by Wall (Simple Bypass)",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5)],
        "known_obstacles": [(5, 6)], # Стена преграждает путь
        "target_known_at": (5, 7), # Цель за стеной
        "comment": "Нельзя идти прямо. Нужно обойти сверху или снизу.",
        "expected_valid_coords": [(4, 6), (6, 6)] # Координаты обхода
    },
    {
        "id": 13, "name": "Target Blocked by Bomb (Risk Assessment)",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5)],
        "known_obstacles": [(5, 6)], # Бомба преграждает путь
        "target_known_at": (5, 7),
        "comment": "Аналогично стене, но строже. Обход.",
        "expected_valid_coords": [(4, 6), (6, 6)]
    },
    {
        "id": 14, "name": "Long Distance Strategic Move",
        "agent_pos": (0, 0),
        "known_visited": [(0, 0)],
        "known_obstacles": [(0, 1)], # Стена справа
        "target_known_at": (9, 9), # Цель далеко вниз-вправо
        "comment": "Идти вниз (DOWN), так как вправо нельзя.",
        "expected_valid_coords": [(1, 0)]
    },
    {
        "id": 15, "name": "Target vs Unexplored Conflict",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5)],
        "known_obstacles": [],
        "target_known_at": (0, 0), # Цель сзади
        "comment": "Цель важнее исследования. Идти назад/вверх.",
        "expected_valid_coords": [(4, 5), (5, 4)] # Движение к (0,0)
    },

    # ==========================================
    # КАТЕГОРИЯ 4: ПРОДВИНУТЫЕ ТАКТИКИ (Tactics)
    # ==========================================
    {
        "id": 16, "name": "Diagonal Blockade",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5)],
        "known_obstacles": [(4, 5), (5, 4)], # Блок сверху и слева
        "target_known_at": (2, 2), # Цель вверху слева
        "comment": "Прямой путь перекрыт. Нужен сложный маневр (вниз или вправо для захода с тыла).",
        "expected_valid_coords": [(6, 5), (5, 6)] # Выход из зоны
    },
    {
        "id": 17, "name": "U-Turn Necessity",
        "agent_pos": (5, 8), # В конце туннеля
        "known_visited": [(5, 5), (5, 6), (5, 7), (5, 8)],
        "known_obstacles": [(5, 9), (4, 8), (6, 8)], # Тупик
        "target_known_at": None,
        "comment": "Единственный выход - вернуться назад.",
        "expected_valid_coords": [(5, 7)]
    },
    {
        "id": 18, "name": "Island Hopping (Risky Path)",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5)],
        "known_obstacles": [(4, 5), (6, 5), (5, 4)], # Окружен стенами, только один проход
        "target_known_at": (8, 5), # Цель далеко вправо
        "comment": "Единственный путь - через единственный проход.",
        "expected_valid_coords": [(5, 6)]
    },
    {
        "id": 19, "name": "Optimal Frontier Angle",
        "agent_pos": (5, 5),
        "known_visited": [(5, 5), (5, 6), (6, 5), (6, 6)], # Посещен квадрат справа-снизу
        "known_obstacles": [],
        "target_known_at": None,
        "comment": "Исследовано справа и снизу. Остались слева и сверху. Выбрать ближайшее.",
        "expected_valid_coords": [(4, 5), (5, 4)]
    },
    {
        "id": 20, "name": "Trap Awareness",
        "agent_pos": (1, 1),
        "known_visited": [(0, 1), (1, 0), (1, 1)], # Вход в комнату
        "known_obstacles": [(0, 0), (0, 2), (1, 2)], # Стены формируют ловушку
        "target_known_at": None,
        "comment": "Идти вниз (2,1), чтобы не застрять в углу.",
        "expected_valid_coords": [(2, 1)]
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