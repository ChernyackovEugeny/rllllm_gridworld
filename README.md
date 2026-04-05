# rlllm_gridworld

Research project comparing RL and LLM-based agents on a Gymnasium grid-world navigation task. The agent must reach a target while avoiding bombs, across grid sizes from 5×5 to 10×10. Eight agent architectures are implemented and benchmarked.

---

## Benchmark

Evaluation: 100 episodes for RL agents, 20 episodes for LLM agents. Bombs: 3 on 5×5, 10 on 10×10.

### 5×5 Grid

| Agent | Success Rate | Mean Reward | Mean Steps | Bomb Hit Rate |
|---|---|---|---|---|
| Random | 22% | -0.945 | 13.7 | 73% |
| PPO Baseline | 85% | 0.606 | 11.9 | 0% |
| A2C | 10% | -0.713 | 44.2 | 2% |
| DQN | 92% | 0.810 | 7.0 | 0% |
| PPO + CNN Distillation | 88% | 0.724 | 9.6 | 0% |
| BFS Oracle | **100%** | **0.954** | **3.6** | 0% |
| LLM Code-Gen | **100%** | 0.947 | 4.0 | 0% |
| LLM Planner | **100%** | 0.932 | 4.8 | 0% |

![5x5 results](src/evaluation/plots/results_5x5.png)

### 10×10 Grid

| Agent | Success Rate | Mean Reward | Mean Steps | Bomb Hit Rate |
|---|---|---|---|---|
| Random | 8% | -0.978 | 20.8 | 91% |
| PPO Baseline | 2% | -0.870 | 182.3 | 7% |
| A2C | 5% | -0.822 | 158.5 | 16% |
| DQN | 19% | -0.856 | 162.9 | 0% |
| PPO + CNN Distillation | 42% | -0.162 | 119.8 | 0% |
| BFS Oracle | **100%** | **0.974** | **6.8** | 0% |
| LLM Code-Gen | **100%** | 0.947 | 12.8 | 0% |
| LLM Planner | 90% | 0.735 | 37.6 | 0% |

![10x10 results](src/evaluation/plots/results_10x10.png)

To reproduce:

```bash
python src/evaluation/evaluate_all_agents.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file at the project root:

```
DEEPSEEK_API_KEY=sk-...
HUGGINGFACE_HUB_TOKEN=hf_...
```

---

## Agents

### 1. LLM Code-Generation Agent (`src/agent/`)

The LLM generates executable Python code at each step. A `MetaController` decides whether to retrieve a previously stored skill from the skill library (via RAG similarity search) or generate new code with DeepSeek (up to 3 error-recovery retries). Skills accumulate in a JSON library, scored by `mean_reward × √usage_count`; garbage collection keeps the top-k.

Generated code receives `agent_pos`, `known_world`, `go_to(coord)`, and `get_nearest_unknown()`, and must return `(action, is_done)` from a `decide_action()` function.

Skills are learned online during inference — no separate training step.

**Inference**

```bash
# Live demo (random scenario)
python src/agent/inference/inference1.py

# 3 hardcoded scenarios: Great Wall, Maze, Snake
python src/agent/inference/inference2.py
```

---

### 2. LLM High-Level Planner (`src/llm_high_level_planning/`)

`DeepSeekPlanner` asks the LLM for a goal coordinate every N steps. `HighLevelPlannerWrapper` then uses BFS to navigate toward that goal. Falls back to random frontier exploration when the LLM fails or returns an invalid coordinate.

**Inference**

```bash
# Basic run
python src/llm_high_level_planning/inference/inference1.py

# Interactive step-by-step mode
python src/llm_high_level_planning/inference/inference2.py

# 5 hardcoded scenarios
python src/llm_high_level_planning/inference/inference3.py
```

**Prompt testing**

```bash
# Single prompt quality test
python src/llm_high_level_planning/testing/promt_testing.py

# A/B comparison between two prompt strategies
python src/llm_high_level_planning/testing/promt_AB_testing.py
```

Results are written to `src/llm_high_level_planning/testing/*.csv`.

---

### 3. PPO Baseline (`src/ppo_baseline/`)

Pure RL — no LLM. Uses `MultiInputPolicy` with 4 parallel environments and trains for 600k steps. Provides a performance baseline for comparing the LLM-augmented approaches.

**Training**

```bash
# 5×5 grid
python src/ppo_baseline/training/ppo_agent_size5.py

# 10×10 grid
python src/ppo_baseline/training/ppo_agent_size10.py
```

**Inference**

```bash
# 5×5 grid
python src/ppo_baseline/inference/inference1.py

# 10×10 grid with action probability display
python src/ppo_baseline/inference/inference2.py

# Multi-scenario evaluation
python src/ppo_baseline/inference/inference3.py
```

---

### 4. PPO + LLM Hints (`src/ppo_llmhint/`)

A PPO policy augmented with LLM exploration hints. The LLM advisor produces a suggested action that is appended to the observation vector, giving the policy a learned signal to follow or override.

**Training**

```bash
# 5×5 grid, 20k steps
python src/ppo_llmhint/training/ppo_llm_agent1.py
```

**Inference**

```bash
# Standard run
python src/ppo_llmhint/inference/ppo_llm_inference1.py

# Step-by-step with logit visualization
python src/ppo_llmhint/inference/ppo_llm_inference1.2.py
```

---

### 5. PPO + CNN Knowledge Distillation (`src/ppo_llmhint_conv_distilation/`)

A two-stage pipeline: an LLM teacher generates `(state, action)` pairs offline; a CNN student (`ExplorerCNN`) is trained on those pairs via supervised learning; the trained CNN then provides hints to the PPO policy during training. No LLM is required at runtime.

**Training**

```bash
# 5×5 grid, 600k PPO steps
python src/ppo_llmhint_conv_distilation/train/ppo_llmhint_train_5size.py

# 10×10 grid, 1.2M PPO steps
python src/ppo_llmhint_conv_distilation/train/ppo_llmhint_train_10size.py
```

