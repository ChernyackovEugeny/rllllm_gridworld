import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from collections import deque
import pygame

def dfs(vector, visited, grid):
    x, y = vector

    if (x, y) in visited:
        return False
    if grid[x][y] == Places.TARGET.value:
        return True
    if grid[x][y] == Places.BOMB.value:
        return False

    visited.add((x, y))
    for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        neigh_x, neigh_y = x + dx, y + dy
        if neigh_x < 0 or neigh_y < 0 or neigh_y >= len(grid) or neigh_x >= len(grid[0]):
            continue
        if dfs((neigh_x, neigh_y), visited, grid):
            return True
    return False

def bfs(vector, grid):
    queue = deque([tuple(vector)])
    visited = set([tuple(vector)])

    while queue:
        current = queue.popleft()
        x, y = current[0], current[1]

        if grid[x][y] == Places.TARGET.value:
            return True
        if grid[x][y] == Places.BOMB.value:
            continue

        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False


class Actions(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class Places(Enum):
    NOTHING = 0
    TARGET = 1
    WALL = 2
    BOMB = 3
    AGENT = 4
    DONT_KNOW = 5

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_bombs=5, max_steps=200, observation_size=5, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = max_steps
        self.current_step = 0

        assert observation_size % 2 == 1
        self.observation_size = observation_size
        self.cell_type_count = len(Places)
        self.num_bombs = num_bombs

        self.observation_space = spaces.Dict({
            'agent_position': spaces.Box(
                low=0, high=self.size-1,
                shape=(2,),
                dtype=np.int32
            ),
            'agent_observations': spaces.Box(
                low=0,
                high=self.cell_type_count-1,
                shape=(observation_size, observation_size),
                dtype=np.int32
            )
        })

        self.action_space = spaces.Discrete(4)

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._agent_observations = np.ones(shape=(observation_size, observation_size), dtype=np.int32) * -1
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._bombs_location = np.ones(shape=(num_bombs, 2), dtype=np.int32) * -1

        self.grid = np.ones(shape=(size, size), dtype=np.int32) * -1

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        Uses NumPy [row, col] convention where row 0 is at the top.
        """
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([0, 1]),
            Actions.UP.value: np.array([-1, 0]),
            Actions.LEFT.value: np.array([0, -1]),
            Actions.DOWN.value: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)

        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)

        self._bombs_location = self._set_bombs()
        self.grid = self._generate_grid()
        while not bfs(self._agent_location, self.grid):
            self._bombs_location = self._set_bombs()
            self.grid = self._generate_grid()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _set_bombs(self):
        bombs = set()
        while len(bombs) < self.num_bombs:
            b = tuple(self.np_random.integers(0, self.size, size=2))
            while b == tuple(self._agent_location) or b == tuple(self._target_location) or b in bombs:
                b = tuple(self.np_random.integers(0, self.size, size=2))
            bombs.add(b)

        return np.array(list(bombs), dtype=np.int32)

    def _generate_grid(self):
        grid = np.zeros(shape=(self.size, self.size), dtype=np.int32)
        grid[tuple(self._agent_location)] = Places.AGENT.value
        grid[tuple(self._target_location)] = Places.TARGET.value
        for bomb_loc in self._bombs_location:
            grid[tuple(bomb_loc)] = Places.BOMB.value
        return grid

    def _get_obs(self):
        x, y = self._agent_location

        pad = self.observation_size // 2
        padded_grid = np.full((self.size + pad * 2, self.size + pad * 2), Places.WALL.value, dtype=np.int32)
        padded_grid[pad:pad + self.size, pad:pad + self.size] = self.grid

        # координаты агента с учётом паддинга
        px, py = x + pad, y + pad
        # локальный патч вокруг агента
        local_patch = padded_grid[px - pad:px + pad + 1, py - pad:py + pad + 1]

        observation = {
            'agent_position': self._agent_location.astype(np.float32),
            'agent_observations': local_patch.astype(np.float32)
        }

        # Проверяем, что наблюдение соответствует observation_space
        # if not self.observation_space.contains(observation):
        #     raise ValueError("Наблюдение не соответствует observation_space")

        return observation

    def _get_info(self):
        return {
            'grid': self.grid.copy()
        }

    def step(self, action):
        direction = self._action_to_direction[action]
        new_pos = self._agent_location + direction

        terminated = False
        truncated = False
        reward = -0.01  # базовый штраф за шаг

        # --- Проверка границ ---
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            reward = -0.1  # штраф за попытку выйти за границу
        else:
            # обновляем позицию агента
            self._agent_location = new_pos

            # смотрим, что было в клетке ДО перегенерации grid
            current_cell = self.grid[tuple(self._agent_location)]

            if current_cell == Places.BOMB.value:
                reward = -1.0
                terminated = True

            elif current_cell == Places.TARGET.value:
                reward = 1.0
                terminated = True

        # увеличиваем шаг
        self.current_step += 1

        # проверка лимита шагов
        if self.current_step >= self.max_steps and not terminated:
            truncated = True

        # обновляем grid (чтобы агент отрисовывался в новой позиции)
        self.grid = self._generate_grid()

        # формируем observation
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = self.window_size / self.size

        # ---------- Рисуем бомбы ----------
        for bomb in self._bombs_location:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),  # чёрные бомбы
                pygame.Rect(
                    pix_square_size * bomb[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )

        # ---------- Рисуем цель ----------
        pygame.draw.rect(
            canvas,
            (255, 0, 0),  # красная цель
            pygame.Rect(
                pix_square_size * self._target_location[::-1],
                (pix_square_size, pix_square_size),
            ),
        )

        # ---------- Рисуем агента ----------
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # синий агент
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # ---------- Сетка ----------
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        # ---------- Отображение ----------
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2),
            )
