from src.environment.environment import GridWorldEnv
from src.code_as_policy.CodeGeneratorWrapper import CodeGeneratorWrapper
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
base_env = GridWorldEnv(render_mode='human', size=5, num_bombs=3)
env = CodeGeneratorWrapper(base_env, skills_path='../skills/skills.json')

obs, info = env.reset()
done = False
total_reward = 0

print("Start. LLM Generator mode active.")

while not done:
    env.render()

    print("\n" + "=" * 40)
    print(f"Agent Pos: {tuple(env.unwrapped._agent_location)}")

    # Ждем нажатия
    should_continue = wait_for_key(env)
    if not should_continue:
        break

    # Выполняем шаг (действие заглушка)
    obs, reward, terminated, truncated, info = env.step(0)
    done = terminated or truncated
    total_reward += reward

env.render()
print(f"\nFinished. Total Reward: {total_reward}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
env.close()