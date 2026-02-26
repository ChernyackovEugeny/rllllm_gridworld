from src.environment.environment import GridWorldEnv
from src.llm_high_level_planning.HighLevelPlannerWrapper import HighLevelPlannerWrapper
import pygame
import time


def wait_for_key(env):
    print("\nНажмите Пробел/Enter в окне Pygame для следующего шага...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    waiting = False
        time.sleep(0.05)
    return True


# --- Основной код ---

# 1. Инициализация
base_env = GridWorldEnv(render_mode='human', size=10, num_bombs=10)
env = HighLevelPlannerWrapper(base_env, plan_frequency=10)

obs, info = env.reset()
done = False
total_reward = 0

print("Start. LLM Planner mode active.")

while not done:
    env.render()

    print("\n" + "=" * 40)
    print(f"Agent Pos: {tuple(env.unwrapped._agent_location)}")

    # Вывод информации о текущем плане LLM
    if env.current_goal:
        print(f"LLM Goal: {env.current_goal} (Path left: {len(env.current_path)})")
    else:
        print("LLM Goal: Calculating...")

    # Ждем нажатия
    should_continue = wait_for_key(env)
    if not should_continue:
        break

    # Выполняем шаг (действие заглушка)
    obs, reward, terminated, truncated, info = env.step(0)
    done = terminated or truncated
    total_reward += reward

print(f"\nFinished. Total Reward: {total_reward}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
env.close()