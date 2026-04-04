"""
Random Agent baseline — no training.
Runs N episodes with uniform random actions and reports performance metrics.
"""
import os
import sys
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.environment import GridWorldEnv

# --- Config ---
SIZE = 5           # Grid size: 5 or 10
NUM_BOMBS = 3 if SIZE == 5 else 10
N_EPISODES = 100
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'evaluation', 'results')

env = GridWorldEnv(render_mode=None, size=SIZE, num_bombs=NUM_BOMBS)

results = []
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    terminated = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
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

print(f"\n--- Random Agent {SIZE}x{SIZE} ({N_EPISODES} episodes) ---")
print(f"Success rate : {success_rate:.1f}%")
print(f"Mean reward  : {mean_reward:.3f}")
print(f"Bomb hit rate: {bomb_rate:.1f}%")

os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, f'random_size{SIZE}.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'success', 'bomb_hit'])
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved to {out_path}")
