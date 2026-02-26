import torch
from stable_baselines3 import PPO
from src.environment.environment import GridWorldEnv
from src.ppo_llmhint.wrappers import LLMExplorerWrapper
import pygame
import time


# Функция для ожидания нажатия клавиши без зависания окна
def wait_for_key(env):
    print("\nНажмите Enter в консоли ИЛИ Пробел/Enter в окне Pygame для следующего шага...")
    waiting = True
    while waiting:
        # 1. Обрабатываем события Pygame, чтобы окно не зависло
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Если закрыли окно, выходим из программы
                return False  # Сигнал к выходу
            if event.type == pygame.KEYDOWN:
                # Если нажали Enter или Пробел в окне Pygame
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    waiting = False

        # 2. Проверяем ввод в консоли (небольшой трюк для Windows)
        # Внимание: это работает только если консоль активна.
        # Для надежности лучше нажимать клавиши в окне Pygame.
        # Если вы работаете в IDE, input() все равно заблокирует GUI,
        # поэтому ниже мы используем цикл событий Pygame как основной способ.

        # Небольшая задержка, чтобы не грузить процессор на 100%
        time.sleep(0.05)

        # Если нужно именно через консоль:
        # К сожалению, input() блокирует GUI.
        # Поэтому лучше нажимать "Пробел" на открытом окне игры.

    return True


# --- Основной код ---

# Создаем среду
env = LLMExplorerWrapper(GridWorldEnv(render_mode='human', size=5, num_bombs=3))
model = PPO.load('./models/ppo_llm_test', env=env)

obs, info = env.reset()
done = False
total_reward = 0

print("Начинаем эпизод. Управление: Нажмите 'Enter' или 'Пробел' в окне игры для шага.")

# Основной цикл
while not done:
    # 1. Отрисовка
    env.render()

    # 2. Получаем логиты и вероятности
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        logits = distribution.distribution.logits
        probs = distribution.distribution.probs

    # Вывод в консоль
    print("\n" + "=" * 30)
    print(f"Логиты: {logits.cpu().numpy()[0]}")
    print(f"Вероятности: {probs.cpu().numpy()[0]}")
    print("=" * 30)

    action_names = ["Влево", "Вправо", "Вверх", "Вниз"]
    for i, p in enumerate(probs.cpu().numpy()[0]):
        print(f"Действие {i} ({action_names[i]}): {p * 100:.2f}%")

    # 3. Ждем нажатия (теперь окно не виснет!)
    # Обратите внимание: фокус должен быть на окне Pygame, чтобы перехватить нажатие
    should_continue = wait_for_key(env)
    if not should_continue:
        break  # Выход, если закрыли окно

    # 4. Получаем действие
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)

    print(f"--> Агент выбрал действие: {action} ({action_names[action]})")

    # 5. Шаг среды
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    print(f"Награда за шаг: {reward}")

print(f"\nЭпизод завершен. Total Reward: {total_reward}")

# Ждем закрытия окна пользователем
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()