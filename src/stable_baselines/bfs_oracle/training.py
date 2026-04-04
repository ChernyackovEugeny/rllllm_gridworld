"""
BFS Oracle baseline — no training.
Agent has full grid knowledge and uses BFS to find the shortest path to target.
Represents an upper-bound on navigational performance (ignores partial observability).
"""
import os
import sys
import csv
from collections import deque

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.environment.environment import GridWorldEnv, Places, Actions

# --- Config ---
SIZE = 5           # Grid size: 5 or 10
NUM_BOMBS = 3 if SIZE == 5 else 10
N_EPISODES = 100
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'evaluation', 'results')


def bfs_next_action(grid, agent_pos, size):
    """
    Returns the action index that moves the agent one step closer to the target
    via the shortest bomb-free path. Returns None if no path exists.
    """
    start = tuple(agent_pos)

    # Find target
    target = None
    for x in range(size):
        for y in range(size):
            if grid[x, y] == Places.TARGET.value:
                target = (x, y)
                break
        if target:
            break

    if target is None:
        return None

    # BFS: store (position, first_action_taken)
    queue = deque()
    visited = {start}

    action_to_delta = {
        Actions.LEFT.value:  (0, -1),
        Actions.RIGHT.value: (0, 1),
        Actions.UP.value:    (-1, 0),
        Actions.DOWN.value:  (1, 0),
    }

    for action, (dx, dy) in action_to_delta.items():
        nx, ny = start[0] + dx, start[1] + dy
        if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
            if grid[nx, ny] != Places.BOMB.value:
                if (nx, ny) == target:
                    return action
                visited.add((nx, ny))
                queue.append(((nx, ny), action))

    while queue:
        (cx, cy), first_action = queue.popleft()
        for dx, dy in action_to_delta.values():
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                if grid[nx, ny] != Places.BOMB.value:
                    if (nx, ny) == target:
                        return first_action
                    visited.add((nx, ny))
                    queue.append(((nx, ny), first_action))

    return None  # No path found


env = GridWorldEnv(render_mode=None, size=SIZE, num_bombs=NUM_BOMBS)

results = []
for ep in range(N_EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    terminated = False

    while not done:
        grid = info['grid']
        agent_pos = env.unwrapped._agent_location

        action = bfs_next_action(grid, agent_pos, SIZE)
        if action is None:
            action = env.action_space.sample()  # fallback (should not happen)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    # success: total_reward > 0 (worst case: 1.0 - 0.9 penalties = 0.1 > 0)
    # bomb:    total_reward < 0 (at minimum -1.0 + tiny step penalties)
    success = int(terminated and total_reward > 0)
    bomb_hit = int(terminated and total_reward < 0)
    results.append({
        'episode': ep,
        'total_reward': total_reward,
        'steps': steps,
        'success': success,
        'bomb_hit': bomb_hit,
    })
    print(f"ep {ep+1}/{N_EPISODES}  reward={total_reward:.3f}  steps={steps}  "
          f"success={success}  bomb={bomb_hit}")

env.close()

success_rate = sum(r['success'] for r in results) / N_EPISODES * 100
mean_reward = sum(r['total_reward'] for r in results) / N_EPISODES
bomb_rate = sum(r['bomb_hit'] for r in results) / N_EPISODES * 100

print(f"\n--- BFS Oracle {SIZE}x{SIZE} ({N_EPISODES} episodes) ---")
print(f"Success rate : {success_rate:.1f}%")
print(f"Mean reward  : {mean_reward:.3f}")
print(f"Bomb hit rate: {bomb_rate:.1f}%")

os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, f'bfs_oracle_size{SIZE}.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'success', 'bomb_hit'])
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved to {out_path}")
