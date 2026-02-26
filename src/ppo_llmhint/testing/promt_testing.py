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

# 1. Определяем 4 стратегии промптинга
PROMPT_STRATEGIES = {
    "last_best": {
        "system": (
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
    ),
        "use_warnings": False,
        "use_reasoning": False
    }
    # "new_new": {
    #     "system": (
    #         "You are an expert navigator AI for a grid.\n"
    #         "AXIS DEFINITION: X is Vertical (UP/DOWN), Y is Horizontal (LEFT/RIGHT).\n"
    #         "- UP decreases X (x-1). DOWN increases X (x+1).\n"
    #         "- LEFT decreases Y (y-1). RIGHT increases Y (y+1).\n\n"
    #
    #         "DECISION PROCESS:\n"
    #         "1. MODE SELECTION:\n"
    #         "   - IF TARGET in Known World -> Set 'Goal' to TARGET coords.\n"
    #         "   - ELSE -> Set 'Goal' to 'Explore'.\n\n"
    #
    #         "2. PATHFINDING LOGIC (If Goal is TARGET):\n"
    #         "   - Calculate Delta: dx = Goal_x - Agent_x, dy = Goal_y - Agent_y.\n"
    #         "   - Primary Moves (Priority order):\n"
    #         "     * If |dx| > |dy|: Move Vertical (UP if dx<0, DOWN if dx>0).\n"
    #         "     * If |dy| > |dx|: Move Horizontal (LEFT if dy<0, RIGHT if dy>0).\n"
    #         "     * If |dx| == |dy|: Move Horizontal first.\n"
    #         "   - SAFETY CHECK: If Primary Move leads to DANGER/WALL/INVALID:\n"
    #         "     * Try Secondary Move (the other axis).\n"
    #         "     * If blocked again, try remaining safe options.\n\n"
    #
    #         "3. EXPLORATION LOGIC (If Goal is Explore):\n"
    #         "   - Check neighbors. Prioritize UNEXPLORED (not in Known World).\n"
    #         "   - Avoid VISITED cells if possible.\n\n"
    #
    #         "4. GENERAL RULES:\n"
    #         "   - NEVER move into DANGER or WALL.\n"
    #         "   - NEVER move outside bounds (e.g., x=-1 is invalid).\n\n"
    #
    #         "Return JSON: {'reasoning': 'step-by-step thought', 'action': 'MOVE'}."
    #     ),
    #     "use_warnings": False,
    #     "use_reasoning": False
    # },
    # "new": {
    #     "system": (
    #         "You are an expert pathfinding AI.\n"
    #         "COORDINATES: UP=(x-1,y), DOWN=(x+1,y), LEFT=(x,y-1), RIGHT=(x,y+1).\n\n"
    #
    #         "LOGIC PROCESS:\n"
    #         "1. TARGET CHECK: Is TARGET in the Known World map?\n\n"
    #
    #         "2. IF TARGET IS KNOWN (Navigation Mode):\n"
    #         "   - Identify direction towards TARGET.\n"
    #         "   - If direct path is blocked by DANGER/WALL: Execute 'Sidestep'.\n"
    #         "     * Blocked horizontally (Left/Right)? Move UP or DOWN.\n"
    #         "     * Blocked vertically (Up/Down)? Move LEFT or RIGHT.\n"
    #         "   - Only move AWAY (backtrack) if no other option exists.\n\n"
    #
    #         "3. IF TARGET IS UNKNOWN (Exploration Mode):\n"
    #         "   - Goal: Find new cells.\n"
    #         "   - Prioritize UNEXPLORED safe cells.\n"
    #         "   - Avoid VISITED cells if possible.\n\n"
    #
    #         "4. SAFETY: NEVER move into DANGER or WALL.\n\n"
    #
    #         "Return JSON with keys 'reasoning' and 'action'.\n"
    #         "'action' values: 'UP', 'DOWN', 'LEFT', 'RIGHT'."
    #                     ),
    #     "use_warnings": False,
    #     "use_reasoning": True
    # }
}

# 2. Определяем 10 сценариев тестирования
# Format: {"agent_pos": (x,y), "vision": "...", "memory": {...}, "expected": "действие или список действий"}
TEST_SCENARIOS = [
    # --- БАЗОВЫЕ СЦЕНАРИИ (1-11) ---
    {
        "id": 1, "name": "Simple Goal",
        "agent_pos": (2, 2),
        "vision": "(2, 3): TARGET",
        "memory": {(2, 2): 'VISITED', (2, 3): 'TARGET'},
        "expected": ["RIGHT"]
    },
    {
        "id": 2, "name": "Simple Danger Avoidance",
        "agent_pos": (2, 2),
        "vision": "(2, 3): DANGER",
        "memory": {(2, 2): 'VISITED', (2, 3): 'DANGER'},
        "expected": ["UP", "DOWN", "LEFT"]
    },
    {
        "id": 3, "name": "Bypass Bomb (The Hard Case)",
        "agent_pos": (2, 2),
        "vision": "(2, 3): DANGER, (2, 4): TARGET",
        "memory": {(2, 2): 'VISITED', (2, 3): 'DANGER', (2, 4): 'TARGET'},
        "expected": ["UP", "DOWN"]
    },
    {
        "id": 4, "name": "Memory Navigation (Blind)",
        "agent_pos": (2, 2),
        "vision": "Open space around",
        "memory": {(2, 2): 'VISITED', (0, 0): 'TARGET'},
        "expected": ["UP", "LEFT"]
    },
    {
        "id": 5, "name": "Dead End (Walls)",
        "agent_pos": (0, 0),
        "vision": "(-1, 0): WALL, (0, -1): WALL",
        "memory": {(0, 0): 'VISITED', (-1, 0): 'WALL', (0, -1): 'WALL'},
        "expected": ["RIGHT", "DOWN"]
    },
    {
        "id": 6, "name": "Visited Cells Avoidance",
        "agent_pos": (2, 2),
        "vision": "Open space around",
        "memory": {(2, 2): 'VISITED', (1, 2): 'VISITED', (2, 1): 'VISITED', (3, 2): 'VISITED'},
        "expected": ["RIGHT"]
    },
    {
        "id": 7, "name": "Prioritize Goal over Exploration",
        "agent_pos": (3, 3),
        "vision": "(3, 4): TARGET, (4, 4): EMPTY",
        "memory": {(3, 3): 'VISITED', (3, 4): 'TARGET'},
        "expected": ["RIGHT"]
    },
    {
        "id": 8, "name": "Trap Corridor",
        "agent_pos": (1, 1),
        "vision": "(0, 1): DANGER, (1, 2): WALL",
        "memory": {(1, 1): 'VISITED', (0, 1): 'DANGER', (1, 2): 'WALL'},
        "expected": ["DOWN", "LEFT"]
    },
    {
        "id": 9, "name": "Ignore Irrelevant History",
        "agent_pos": (0, 0),
        "vision": "(0, 1): TARGET",
        "memory": {(0, 0): 'VISITED', (4, 4): 'DANGER', (1, 1): 'VISITED'},
        "expected": ["RIGHT"]
    },
    {
        "id": 10, "name": "Complex Bypass 2",
        "agent_pos": (2, 2),
        "vision": "(2, 3): WALL, (1, 2): DANGER",
        "memory": {(2, 2): 'VISITED', (2, 3): 'WALL', (1, 2): 'DANGER', (2, 4): 'TARGET'},
        "expected": ["UP", "DOWN"]
    },
    {
        "id": 11, "name": "mik",
        "agent_pos": (2, 2),
        "vision": "(2, 3): DANGER, (2, 4): TARGET",
        "memory": {(2, 2): 'VISITED'},
        "expected": ["UP", "DOWN"]
    },

    # --- ПРОМЕЖУТОЧНЫЕ СЦЕНАРИИ (12-21) ---
    {
        "id": 12, "name": "The Tunnel (Narrow Path)",
        "agent_pos": (2, 2),
        "vision": "(2, 1): WALL, (2, 3): WALL, (0, 2): TARGET",
        "memory": {(2, 2): 'VISITED', (2, 1): 'WALL', (2, 3): 'WALL', (0, 2): 'TARGET'},
        "expected": ["UP"]
    },
    {
        "id": 13, "name": "Panic Mode (Surrounded)",
        "agent_pos": (2, 2),
        "vision": "(1, 2): DANGER, (2, 3): DANGER, (3, 2): DANGER",
        "memory": {(2, 2): 'VISITED', (1, 2): 'DANGER', (2, 3): 'DANGER', (3, 2): 'DANGER', (0, 0): 'TARGET'},
        "expected": ["LEFT"]
    },
    {
        "id": 14, "name": "Retreat to Advance",
        "agent_pos": (0, 0),
        "vision": "(-1, 0): WALL, (0, -1): WALL, (0, 1): DANGER",
        "memory": {(0, 0): 'VISITED', (-1, 0): 'WALL', (0, -1): 'WALL', (0, 1): 'DANGER', (4, 4): 'TARGET'},
        "expected": ["DOWN"]
    },
    {
        "id": 15, "name": "Memory vs Wall (Blind Bypass)",
        "agent_pos": (0, 2),
        "vision": "(0, 3): WALL",
        "memory": {(0, 2): 'VISITED', (0, 3): 'WALL', (0, 4): 'TARGET'},
        "expected": ["DOWN"]
    },
    {
        "id": 16, "name": "Diagonal Block",
        "agent_pos": (2, 2),
        "vision": "(1, 2): DANGER, (2, 3): DANGER",
        "memory": {(2, 2): 'VISITED', (1, 2): 'DANGER', (2, 3): 'DANGER', (0, 4): 'TARGET'},
        "expected": ["DOWN", "LEFT"]
    },
    {
        "id": 17, "name": "Efficient Bypass (Short Path)",
        "agent_pos": (2, 2),
        "vision": "(2, 3): DANGER, (2, 4): TARGET",
        "memory": {(2, 2): 'VISITED', (2, 1): 'VISITED'},
        "expected": ["UP", "DOWN"]
    },
    {
        "id": 18, "name": "U-Turn Logic",
        "agent_pos": (1, 2),
        "vision": "(2, 2): WALL",
        "memory": {(1, 2): 'VISITED', (0, 2): 'VISITED', (2, 2): 'WALL', (4, 2): 'TARGET'},
        "expected": ["LEFT", "RIGHT"]
    },
    {
        "id": 19, "name": "Prioritize Unexplored",
        "agent_pos": (2, 2),
        "vision": "Open space around",
        "memory": {(2, 2): 'VISITED', (2, 1): 'VISITED', (1, 2): 'VISITED'},
        "expected": ["RIGHT", "DOWN"]
    },
    {
        "id": 20, "name": "Look Behind (Memory Check)",
        "agent_pos": (2, 2),
        "vision": "Open space around",
        "memory": {(2, 2): 'VISITED', (2, 0): 'TARGET'},
        "expected": ["LEFT"]
    },
    {
        "id": 21, "name": "Dead End Escape",
        "agent_pos": (2, 2),
        "vision": "(1, 2): WALL, (2, 3): WALL, (3, 2): WALL",
        "memory": {(2, 2): 'VISITED', (1, 2): 'WALL', (2, 3): 'WALL', (3, 2): 'WALL', (0, 0): 'TARGET'},
        "expected": ["LEFT"]
    },

    # --- НОВЫЕ СЛОЖНЕЙШИЕ СЦЕНАРИИ (22-31) ---

    {
        "id": 22, "name": "Zig-Zag Logic (Blocked Axis)",
        # Цель снизу (4,2). Путь вниз заблокирован стеной (3,2).
        # Слева (2,1) посещено. Справа (2,3) посещено.
        # Надо идти ВПРАВО или ВЛЕВО, чтобы обойти стену снизу.
        # Sidestep logic: Blocked Vertically -> Move Horizontal.
        "agent_pos": (2, 2),
        "vision": "(3, 2): WALL",
        "memory": {(2, 2): 'VISITED', (3, 2): 'WALL', (4, 2): 'TARGET', (2, 1): 'VISITED', (2, 3): 'VISITED'},
        "expected": ["LEFT", "RIGHT"]
    },
    {
        "id": 23, "name": "Snake Move (Double Danger)",
        # Цель справа (2,4). Путь прямо (2,3) - DANGER.
        # Обход сверху (1,2) - тоже DANGER!
        # Остается только идти ВНИЗ (3,2).
        "agent_pos": (2, 2),
        "vision": "(2, 3): DANGER, (1, 2): DANGER",
        "memory": {(2, 2): 'VISITED', (2, 3): 'DANGER', (1, 2): 'DANGER', (2, 4): 'TARGET'},
        "expected": ["DOWN"]
    },
    {
        "id": 24, "name": "Edge Bypass (Corner Trap)",
        # Агент в углу (0,4). Цель внизу слева (4,0).
        # Снизу (1,4) - DANGER. Слева (0,3) - DANGER.
        # Некуда идти! Но если есть выход?
        # Давай упростим: Слева (0,3) DANGER. Снизу (1,4) EMPTY.
        # Надо идти вниз.
        "agent_pos": (0, 4),
        "vision": "(0, 3): DANGER, (1, 4): EMPTY",
        "memory": {(0, 4): 'VISITED', (0, 3): 'DANGER', (4, 0): 'TARGET'},
        "expected": ["DOWN"]
    },
    {
        "id": 25, "name": "Long Distance Bypass (Blind)",
        # Агент в (2,2). Цель в (2,4).
        # Напрямку (2,3) - DANGER.
        # В памяти: (1,2) VISITED, (3,2) UNEXPLORED.
        # Логика: Sidestep. Идем вниз (UNEXPLORED), а не вверх (VISITED).
        "agent_pos": (2, 2),
        "vision": "(2, 3): DANGER",
        "memory": {(2, 2): 'VISITED', (2, 3): 'DANGER', (2, 4): 'TARGET', (1, 2): 'VISITED'},
        "expected": ["DOWN"]
    },
    {
        "id": 26, "name": "Corridor of Death",
        # Агент в коридоре (2,2). Стены слева и справа.
        # Сверху (1,2) - DANGER. Цель сверху (0,2).
        # Надо идти ВНИЗ (назад), чтобы обойти.
        "agent_pos": (2, 2),
        "vision": "(2, 1): WALL, (2, 3): WALL, (1, 2): DANGER",
        "memory": {(2, 2): 'VISITED', (2, 1): 'WALL', (2, 3): 'WALL', (1, 2): 'DANGER', (0, 2): 'TARGET'},
        "expected": ["DOWN"]
    },
    {
        "id": 27, "name": "Target Behind Wall",
        # Цель справа за стеной. Агент (2,2). Стена (2,3). Цель (2,4).
        # Сверху (1,2) VISITED. Снизу (3,2) UNEXPLORED.
        # Идем вниз обходить.
        "agent_pos": (2, 2),
        "vision": "(2, 3): WALL",
        "memory": {(2, 2): 'VISITED', (2, 3): 'WALL', (2, 4): 'TARGET', (1, 2): 'VISITED'},
        "expected": ["DOWN"]
    },
    {
        "id": 28, "name": "Ant Colony Logic",
        # Агент (2,2). Цель (2,0).
        # Слева (2,1) VISITED.
        # Надо идти влево, даже если посещено (кратчайший путь).
        # Проверка приоритета цели над исследованием.
        "agent_pos": (2, 2),
        "vision": "Open space around",
        "memory": {(2, 2): 'VISITED', (2, 1): 'VISITED', (2, 0): 'TARGET'},
        "expected": ["LEFT"]
    },
    {
        "id": 29, "name": "Maze Entry Decision",
        # Агент перед входом в лабиринт (2,2).
        # Цель (0,4).
        # Путь ВПРАВО (2,3) DANGER.
        # Путь ВВЕРХ (1,2) VISITED.
        # Путь ВНИЗ (3,2) UNEXPLORED.
        # Sidestep logic: Blocked horizontally -> Vertical.
        # Prefer UNEXPLORED.
        "agent_pos": (2, 2),
        "vision": "(2, 3): DANGER",
        "memory": {(2, 2): 'VISITED', (2, 3): 'DANGER', (0, 4): 'TARGET', (1, 2): 'VISITED'},
        "expected": ["DOWN"]
    },
    {
        "id": 30, "name": "Double Blockade",
        # Цель слева (2,0).
        # Слева (2,1) WALL.
        # Сверху (1,2) DANGER.
        # Надо идти ВНИЗ.
        "agent_pos": (2, 2),
        "vision": "(2, 1): WALL, (1, 2): DANGER",
        "memory": {(2, 2): 'VISITED', (2, 1): 'WALL', (1, 2): 'DANGER', (2, 0): 'TARGET'},
        "expected": ["DOWN"]
    },
    {
        "id": 31, "name": "Safe Backtrack",
        # Агент зашел в тупик (2,2).
        # Сверху (1,2) VISITED.
        # Справа (2,3) WALL.
        # Снизу (3,2) DANGER.
        # Цель была слева (2,0).
        # Надо идти ВЛЕВО (VISITED), другой дороги нет.
        "agent_pos": (2, 2),
        "vision": "(2, 3): WALL, (3, 2): DANGER",
        "memory": {(2, 2): 'VISITED', (2, 3): 'WALL', (3, 2): 'DANGER', (1, 2): 'VISITED', (2, 0): 'TARGET'},
        "expected": ["LEFT"]
    }
]


# --- ФУНКЦИИ ---

def build_user_prompt(strategy_name, scenario):
    """Собирает User Prompt в зависимости от стратегии."""

    base = (
        f"Current Position: {scenario['agent_pos']}\n"
        f"Current Vision: {scenario['vision']}\n"
        f"Explored Map Summary: {json.dumps(list(scenario['memory'].items()))}\n"
    )

    # Стратегия: Добавляем программные предупреждения
    if PROMPT_STRATEGIES[strategy_name]["use_warnings"]:
        x, y = scenario['agent_pos']
        moves = {
            "UP": (x - 1, y), "DOWN": (x + 1, y),
            "LEFT": (x, y - 1), "RIGHT": (x, y + 1)
        }
        warnings = []
        for move, coord in moves.items():
            if coord in scenario['memory']:
                cell_type = scenario['memory'][coord]
                if cell_type in ['DANGER', 'WALL']:
                    warnings.append(f"WARNING: {move} leads to {coord} ({cell_type})!")

        if warnings:
            base += "\n".join(warnings) + "\n"

    base += "Analyze and choose the best action."
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
            temperature=0.0,  # Максимальная точность
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        # Обработка ответа
        action = parsed.get("action", "UNKNOWN").upper()

        # Если модель решала reasoning, выводим его для дебага
        reasoning = parsed.get("reasoning", "")

        return action, reasoning, user_prompt

    except Exception as e:
        return f"ERROR: {str(e)}", "", user_prompt


# --- ЗАПУСК ТЕСТОВ ---

results = []
print(f"Starting tests for {len(PROMPT_STRATEGIES)} strategies and {len(TEST_SCENARIOS)} scenarios...\n")

for strat_name in PROMPT_STRATEGIES:
    print(f"Testing Strategy: {strat_name}...")
    for scenario in TEST_SCENARIOS:
        action, reasoning, prompt_used = get_model_response(strat_name, scenario)

        # Проверка результата
        is_correct = action in scenario['expected']

        results.append({
            "Strategy": strat_name,
            "Scenario ID": scenario['id'],
            "Scenario Name": scenario['name'],
            "Expected": "/".join(scenario['expected']),
            "Model Action": action,
            "Correct": "YES" if is_correct else "NO",
            "Reasoning": reasoning,
            "Prompt Used": prompt_used  # Для анализа, почему модель ошиблась
        })

        # Небольшая пауза, чтобы не спамить API
        time.sleep(0.5)

# --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---

df = pd.DataFrame(results)

# Выводим сводку в консоль
print("\n=== SUMMARY ===")
summary = df.groupby("Strategy")['Correct'].apply(lambda x: (x == 'YES').sum()).reset_index()
summary.columns = ["Strategy", "Correct Answers (out of 10)"]
print(summary)

# Сохраняем детальный отчет
filename = "llm_prompt_test_results.csv"
df.to_csv(filename, index=False)
print(f"\nDetailed results saved to {filename}")

# Выводим ошибки для анализа
failures = df[df['Correct'] == 'NO']
if not failures.empty:
    print("\n--- FAILED CASES ---")
    for idx, row in failures.iterrows():
        print(f"Strat: {row['Strategy']} | Scenario: {row['Scenario Name']}")
        print(f"  Expected: {row['Expected']}, Got: {row['Model Action']}")
        print(f"  Reasoning: {row['Reasoning']}")
        print("-" * 30)