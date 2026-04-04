import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.environment.environment import Places
from src.ppo_llmhint_conv_distilation.ml_model.ExplorerCNN import ExplorerCNN

_DIR = Path(__file__).parent

# --- 2. Fast Wrapper ---
class FastLLMHintWrapper(gym.ObservationWrapper):
    def __init__(self, env, map_size=5, cnn_model_path=None):
        super().__init__(env)

        self.device = "cpu"
        self.model = ExplorerCNN(map_size=map_size).to(self.device)
        path = Path(cnn_model_path) if cnn_model_path else _DIR / "ml_model" / f"student_cnn_{map_size}size.pth"
        if not path.exists():
            raise FileNotFoundError(f"CNN model not found: {path}")
        self.model.load_state_dict(torch.load(str(path), map_location=self.device))
        self.model.eval()

        # Память карты
        self.known_world = {}

        # Обновляем пространство
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            'llm_hint': gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        })

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.known_world = {}
        start_pos = tuple(map(int, self.env.unwrapped._agent_location))
        self.known_world[start_pos] = 'VISITED'
        return self.observation(obs), info

    def observation(self, obs):
        # 1. Обновляем память
        self._update_memory()

        # 2. Кодируем состояние (как при сборе данных)
        state_tensor = self._encode_state()

        # 3. Получаем предсказание
        with torch.no_grad():
            # Добавляем размерность батча [1, 5, 5, 5]
            input_batch = state_tensor.unsqueeze(0).to(self.device)
            logits = self.model(input_batch)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        obs['llm_hint'] = probs.astype(np.float32)
        return obs

    def _update_memory(self):
        """Обновляет known_world на основе текущего обзора."""
        _, current_view_objects = self._get_local_view_description()
        agent_pos = tuple(map(int, self.env.unwrapped._agent_location))

        for coord, obj_type in current_view_objects.items():
            if obj_type in ['TARGET', 'DANGER', 'WALL']:
                self.known_world[coord] = obj_type
            elif obj_type == 'EMPTY':
                # ВАЖНО: Для BFS нам нужно знать, где проходимо
                self.known_world[coord] = 'SAFE'

        self.known_world[agent_pos] = 'VISITED'

    def _encode_state(self):
        # Полная копия функции из сбора данных
        env = self.env.unwrapped
        state = np.zeros((5, env.size, env.size), dtype=np.float32)
        for (x, y), cell_type in self.known_world.items():
            if 0 <= x < env.size and 0 <= y < env.size:
                if cell_type == 'WALL':
                    state[1, x, y] = 1.0
                elif cell_type == 'DANGER':
                    state[2, x, y] = 1.0
                elif cell_type == 'TARGET':
                    state[3, x, y] = 1.0
                elif cell_type in ['VISITED', 'SAFE']:
                    state[4, x, y] = 1.0

        agent_pos = tuple(map(int, env._agent_location))
        state[0, agent_pos[0], agent_pos[1]] = 1.0

        # Стены по краям
        state[1, 0, :] = 1.0;
        state[1, -1, :] = 1.0
        state[1, :, 0] = 1.0;
        state[1, :, -1] = 1.0

        return torch.tensor(state, dtype=torch.float32)

    def _get_local_view_description(self):
        """Парсит обзор для LLM и памяти."""
        env = self.env.unwrapped
        agent_x, agent_y = map(int, env._agent_location)
        interesting_objects = []
        objects_dict = {}

        view_range = range(-2, 3)

        for dx in view_range:
            for dy in view_range:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = agent_x + dx, agent_y + dy

                if 0 <= nx < env.size and 0 <= ny < env.size:
                    cell_value = env.grid[nx, ny]

                    if cell_value == Places.NOTHING.value:
                        objects_dict[(nx, ny)] = 'EMPTY'
                        continue

                    if cell_value == Places.TARGET.value:
                        obj_type = "TARGET"
                    elif cell_value == Places.BOMB.value:
                        obj_type = "DANGER"
                    else:
                        continue

                    interesting_objects.append(f"({nx}, {ny}): {obj_type}")
                    objects_dict[(nx, ny)] = obj_type
                else:
                    # Стены по краям
                    is_honest_wall = (nx == -1 or nx == env.size or ny == -1 or ny == env.size)
                    if is_honest_wall:
                        interesting_objects.append(f"({nx}, {ny}): WALL")
                        objects_dict[(nx, ny)] = "WALL"

        if not interesting_objects:
            return "Open space around", objects_dict

        return ", ".join(interesting_objects), objects_dict