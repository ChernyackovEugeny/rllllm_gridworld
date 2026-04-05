import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from src.environment.environment import GridWorldEnv, Places
from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper

# --- Конфигурация ---
N_EPISODES = 500
MAP_SIZE = 10
DATASET_PATH = Path(__file__).parent / f"dataset_{MAP_SIZE}size.pt"
MAX_STEPS_PER_EPISODE = 50  # Лимит шагов

# --- Энкодер состояния ---
def encode_state(env_wrapper):
    env = env_wrapper.unwrapped
    known_world = env_wrapper.known_world
    agent_pos = tuple(map(int, env._agent_location))

    state = np.zeros((5, env.size, env.size), dtype=np.float32)

    for (x, y), cell_type in known_world.items():
        if 0 <= x < env.size and 0 <= y < env.size:
            if cell_type == 'WALL':
                state[1, x, y] = 1.0
            elif cell_type == 'DANGER':
                state[2, x, y] = 1.0
            elif cell_type == 'TARGET':
                state[3, x, y] = 1.0
            elif cell_type in ['VISITED', 'SAFE']:
                state[4, x, y] = 1.0

    state[0, agent_pos[0], agent_pos[1]] = 1.0

    # Стены по краям
    state[1, 0, :] = 1.0
    state[1, -1, :] = 1.0
    state[1, :, 0] = 1.0
    state[1, :, -1] = 1.0

    return state


# --- Сбор данных ---
data_states = []
data_actions = []

print("Сбор данных с LLM эксперта...")
best_strategy = (
    "You are a strategic exploration AI. "
    "Your task is to choose a coordinate (x, y) to explore next. "
    "Return ONLY a JSON object with 'target_coordinate' as [x, y]. "
    "Prioritize unexplored areas."
)

for episode in tqdm(range(N_EPISODES)):
    env = GridWorldEnv(size=MAP_SIZE, num_bombs=10)
    env = HighLevelPlannerWrapper(env, strategy=best_strategy)
    obs, info = env.reset(seed=episode)

    # --- ИЗМЕНЕНИЕ 1: Временный буфер для текущего эпизода ---
    episode_states = []
    episode_actions = []

    step_count = 0
    done = False
    is_truncated = False  # Флаг, что мы прервали эпизод вручную

    while not done:
        # --- ЗАЩИТА ОТ ЦИКЛА ---
        if step_count >= MAX_STEPS_PER_EPISODE:
            # Мы прервали эпизод, помечаем это
            is_truncated = True
            # print(f"Episode {episode} truncated due to step limit.") # Можно раскомментировать для логов
            break

        # 1. Запоминаем состояние ДО шага
        state_tensor = encode_state(env)

        # 2. Делаем шаг экспертом
        prev_pos = tuple(map(int, env.unwrapped._agent_location))
        obs, reward, terminated, truncated, info = env.step(None)
        next_pos = tuple(map(int, env.unwrapped._agent_location))

        step_count += 1

        # 3. Вычисляем действие (0-3)
        dx = next_pos[0] - prev_pos[0]
        dy = next_pos[1] - prev_pos[1]

        if dy == 1:
            action = 1
        elif dy == -1:
            action = 0
        elif dx == 1:
            action = 3
        elif dx == -1:
            action = 2
        else:
            action = -1

        # --- ИЗМЕНЕНИЕ 2: Сохраняем во временный буфер ---
        if action != -1:
            episode_states.append(state_tensor)
            episode_actions.append(action)

        done = terminated or truncated

    env.close()

    # --- ИЗМЕНЕНИЕ 3: Сохраняем данные только если не было ручного прерывания ---
    # Мы сохраняем данные, если эпизод закончился нормально (terminated=True или truncated от среды),
    # но исключаем случай, когда мы вышли по `break` из-за лимита шагов (is_truncated=True).
    if not is_truncated:
        data_states.extend(episode_states)
        data_actions.extend(episode_actions)

# Сохраняем в формате PyTorch
torch.save({
    'states': torch.tensor(np.array(data_states), dtype=torch.float32),
    'actions': torch.tensor(data_actions, dtype=torch.long)
}, DATASET_PATH)

print(f"Собрано {len(data_actions)} примеров. Сохранено в {DATASET_PATH}")