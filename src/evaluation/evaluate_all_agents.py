"""
Benchmark all agent architectures on 5×5 and 10×10 grids.
Run from project root:
    python src/evaluation/evaluate_all_agents.py
"""

import os
import sys
import csv
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from stable_baselines3 import PPO, A2C, DQN

from src.environment.environment import GridWorldEnv, Places, Actions
from src.ppo_llmhint_conv_distilation.FastLLMHintWrapper import FastLLMHintWrapper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS = {
    "PPO Baseline": {
        5:  "src/best_models/ppo_size5/best_model.zip",
        10: "src/best_models/ppo_size10/best_model.zip",
    },
    "A2C": {
        5:  "src/best_models/a2c_size5/best_model.zip",
        10: "src/best_models/a2c_size10/best_model.zip",
    },
    "DQN": {
        5:  "src/best_models/dqn_size5/best_model.zip",
        10: "src/best_models/dqn_size10/best_model.zip",
    },
    "PPO+CNN Distil": {
        5:  "src/models/ppo_llmhint_student_600k_5size.zip",
        10: "src/models/ppo_llmhint_student_1200k_10size.zip",
    },
}

CNN_PATHS = {
    5:  "src/ppo_llmhint_conv_distilation/ml_model/student_cnn_5size.pth",
    10: "src/ppo_llmhint_conv_distilation/ml_model/student_cnn_10size.pth",
}

SKILLS_CODEGEN = "src/agent/skills/skills.json"

RESULTS_DIR = "src/evaluation/results"
PLOTS_DIR   = "src/evaluation/plots"

BOMBS = {5: 3, 10: 10}
PPO_EPISODES = 100
LLM_EPISODES = 20

AGENT_NAMES = [
    "Random",
    "PPO Baseline",
    "A2C",
    "DQN",
    "PPO+CNN Distil",
    "BFS Oracle",
    "LLM Code-Gen",
    "LLM Planner",
]
GRID_SIZES = [5, 10]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_ppo_episodes(model, env, n_episodes):
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        terminated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated
        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "success": int(terminated and total_reward > 0),
            "bomb_hit": int(terminated and total_reward < 0),
        })
        print(f"  ep {ep+1}/{n_episodes}  reward={total_reward:.3f}  steps={steps}  "
              f"success={results[-1]['success']}  bomb={results[-1]['bomb_hit']}")
    return results


def run_wrapper_episodes(wrapper, n_episodes):
    """Works for both CodeGeneratorWrapper and HighLevelPlannerWrapper."""
    results = []
    for ep in range(n_episodes):
        obs, info = wrapper.reset()
        done = False
        total_reward = 0.0
        steps = 0
        terminated = False
        while not done:
            obs, reward, terminated, truncated, info = wrapper.step(0)  # action ignored
            total_reward += reward
            steps += 1
            done = terminated or truncated
        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "success": int(terminated and total_reward > 0),
            "bomb_hit": int(terminated and total_reward < 0),
        })
        print(f"  ep {ep+1}/{n_episodes}  reward={total_reward:.3f}  steps={steps}  "
              f"success={results[-1]['success']}  bomb={results[-1]['bomb_hit']}")
    return results


# ---------------------------------------------------------------------------
# BFS Oracle helper (full-grid knowledge, no model)
# ---------------------------------------------------------------------------

def _bfs_next_action(grid, agent_pos, size):
    """Returns action index for one step toward the target via BFS on the full grid."""
    start = tuple(agent_pos)
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

    action_to_delta = {
        Actions.LEFT.value:  (0, -1),
        Actions.RIGHT.value: (0,  1),
        Actions.UP.value:    (-1, 0),
        Actions.DOWN.value:  ( 1, 0),
    }

    visited = {start}
    queue = deque()
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
    return None


# ---------------------------------------------------------------------------
# Per-agent runners
# ---------------------------------------------------------------------------

def eval_random(size):
    env = GridWorldEnv(render_mode=None, num_bombs=BOMBS[size], size=size)
    n = PPO_EPISODES
    results = []
    print(f"[Random {size}x{size}] running {n} episodes")
    for ep in range(n):
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
        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "success": int(terminated and total_reward > 0),
            "bomb_hit": int(terminated and total_reward < 0),
        })
        print(f"  ep {ep+1}/{n}  reward={total_reward:.3f}  steps={steps}  "
              f"success={results[-1]['success']}  bomb={results[-1]['bomb_hit']}")
    env.close()
    return results


def eval_ppo_baseline(size):
    model_path = os.path.join(PROJECT_ROOT, MODELS["PPO Baseline"][size])
    env = GridWorldEnv(render_mode=None, num_bombs=BOMBS[size], size=size)
    model = PPO.load(model_path, env=env)
    print(f"[PPO Baseline {size}x{size}] loaded {model_path}")
    results = run_ppo_episodes(model, env, PPO_EPISODES)
    env.close()
    return results


def eval_a2c(size):
    model_path = os.path.join(PROJECT_ROOT, MODELS["A2C"][size])
    env = GridWorldEnv(render_mode=None, num_bombs=BOMBS[size], size=size)
    model = A2C.load(model_path, env=env)
    print(f"[A2C {size}x{size}] loaded {model_path}")
    results = run_ppo_episodes(model, env, PPO_EPISODES)
    env.close()
    return results


def eval_dqn(size):
    model_path = os.path.join(PROJECT_ROOT, MODELS["DQN"][size])
    env = GridWorldEnv(render_mode=None, num_bombs=BOMBS[size], size=size)
    model = DQN.load(model_path, env=env)
    print(f"[DQN {size}x{size}] loaded {model_path}")
    results = run_ppo_episodes(model, env, PPO_EPISODES)
    env.close()
    return results


def eval_ppo_cnn(size):
    base_env  = GridWorldEnv(render_mode=None, num_bombs=BOMBS[size], size=size)
    cnn_path  = os.path.join(PROJECT_ROOT, CNN_PATHS[size])
    env = FastLLMHintWrapper(base_env, map_size=size, cnn_model_path=cnn_path)
    model_path = os.path.join(PROJECT_ROOT, MODELS["PPO+CNN Distil"][size])
    model = PPO.load(model_path, env=env,
                     custom_objects={"observation_space": env.observation_space})
    print(f"[PPO+CNN Distil {size}x{size}] loaded {model_path}")
    results = run_ppo_episodes(model, env, PPO_EPISODES)
    env.close()
    return results


def eval_bfs_oracle(size):
    env = GridWorldEnv(render_mode=None, num_bombs=BOMBS[size], size=size)
    n = PPO_EPISODES
    results = []
    print(f"[BFS Oracle {size}x{size}] running {n} episodes")
    for ep in range(n):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        terminated = False
        while not done:
            action = _bfs_next_action(info["grid"], env.unwrapped._agent_location, size)
            if action is None:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "success": int(terminated and total_reward > 0),
            "bomb_hit": int(terminated and total_reward < 0),
        })
        print(f"  ep {ep+1}/{n}  reward={total_reward:.3f}  steps={steps}  "
              f"success={results[-1]['success']}  bomb={results[-1]['bomb_hit']}")
    env.close()
    return results


def eval_codegen(size):
    from src.agent.CodeGeneratorWrapper import CodeGeneratorWrapper
    skills_path = os.path.join(PROJECT_ROOT, SKILLS_CODEGEN)
    env = GridWorldEnv(render_mode=None, num_bombs=BOMBS[size], size=size)
    wrapper = CodeGeneratorWrapper(env, skills_path=skills_path)
    print(f"[LLM Code-Gen {size}x{size}] running {LLM_EPISODES} episodes")
    results = run_wrapper_episodes(wrapper, LLM_EPISODES)
    env.close()
    return results


def eval_planner(size):
    from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper
    env = GridWorldEnv(render_mode=None, num_bombs=BOMBS[size], size=size)
    wrapper = HighLevelPlannerWrapper(env)
    print(f"[LLM Planner {size}x{size}] running {LLM_EPISODES} episodes")
    results = run_wrapper_episodes(wrapper, LLM_EPISODES)
    env.close()
    return results


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_csv(all_rows):
    os.makedirs(os.path.join(PROJECT_ROOT, RESULTS_DIR), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, RESULTS_DIR, "all_results.csv")
    fieldnames = ["agent", "grid_size", "episode", "total_reward", "steps", "success", "bomb_hit"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved CSV -> {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "Random":        "#B0B0B0",
    "PPO Baseline":  "#4C72B0",
    "A2C":           "#8172B2",
    "DQN":           "#937860",
    "PPO+CNN Distil":"#DD8452",
    "BFS Oracle":    "#2CA02C",
    "LLM Code-Gen":  "#55A868",
    "LLM Planner":   "#C44E52",
}


def _agg(rows, key):
    vals = [r[key] for r in rows]
    return np.mean(vals), np.std(vals) / max(np.sqrt(len(vals)), 1)


def plot_size(size, data_by_agent):
    """data_by_agent: dict[agent_name -> list of episode dicts]"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"Agent Benchmark — {size}×{size} Grid", fontsize=15, fontweight="bold")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    agents = AGENT_NAMES
    colors = [COLORS[a] for a in agents]
    x = np.arange(len(agents))
    bar_w = 0.6

    # 1. Success Rate
    ax = axes[0, 0]
    sr = [_agg(data_by_agent[a], "success")[0] * 100 for a in agents]
    bars = ax.bar(x, sr, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Success Rate (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    for bar, val in zip(bars, sr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    # 2. Mean Total Reward
    ax = axes[0, 1]
    means = [_agg(data_by_agent[a], "total_reward")[0] for a in agents]
    errs  = [_agg(data_by_agent[a], "total_reward")[1] for a in agents]
    ax.bar(x, means, width=bar_w, color=colors, edgecolor="white", linewidth=0.8,
           yerr=errs, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_title("Mean Total Reward", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)

    # 3. Mean Steps
    ax = axes[1, 0]
    means = [_agg(data_by_agent[a], "steps")[0] for a in agents]
    errs  = [_agg(data_by_agent[a], "steps")[1] for a in agents]
    ax.bar(x, means, width=bar_w, color=colors, edgecolor="white", linewidth=0.8,
           yerr=errs, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_title("Mean Steps", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)

    # 4. Bomb Hit Rate
    ax = axes[1, 1]
    bhr = [_agg(data_by_agent[a], "bomb_hit")[0] * 100 for a in agents]
    bars = ax.bar(x, bhr, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Bomb Hit Rate (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    for bar, val in zip(bars, bhr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, PLOTS_DIR), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, PLOTS_DIR, f"results_{size}x{size}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_rows = []

    RUNNERS = {
        "Random":        eval_random,
        "PPO Baseline":  eval_ppo_baseline,
        "A2C":           eval_a2c,
        "DQN":           eval_dqn,
        "PPO+CNN Distil": eval_ppo_cnn,
        "BFS Oracle":    eval_bfs_oracle,
        "LLM Code-Gen":  eval_codegen,
        "LLM Planner":   eval_planner,
    }

    # Collect results keyed by agent name and grid size
    collected = {a: {} for a in AGENT_NAMES}

    for size in GRID_SIZES:
        for agent_name in AGENT_NAMES:
            print(f"\n{'='*60}")
            print(f"  {agent_name}  |  {size}x{size}")
            print(f"{'='*60}")
            try:
                rows = RUNNERS[agent_name](size)
                collected[agent_name][size] = rows
                for r in rows:
                    all_rows.append({
                        "agent": agent_name,
                        "grid_size": size,
                        **r,
                    })
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                # Placeholder so plots still render
                collected[agent_name][size] = [
                    {"episode": 0, "total_reward": 0.0, "steps": 0,
                     "success": 0, "bomb_hit": 0}
                ]

    save_csv(all_rows)

    for size in GRID_SIZES:
        data_by_agent = {a: collected[a][size] for a in AGENT_NAMES}
        plot_size(size, data_by_agent)

    print("\nDone.")


if __name__ == "__main__":
    main()
