import torch.nn as nn
import torch.nn.functional as F


# --- Модель Студента (CNN) ---
class ExplorerCNN(nn.Module):
    def __init__(self, map_size=5):
        super().__init__()
        # Вход: 5 каналов (Agent, Wall, Bomb, Target, Visited)
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)  # 3x3 свертка
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # ВАЖНО: Считаем размер входа для полносвязного слоя динамически
        # После двух сверток с padding=1 размер карты остается map_size x map_size
        conv_output_size = 32 * (map_size * map_size)

        self.fc1 = nn.Linear(conv_output_size, 128)  # Для карты 5x5
        self.fc2 = nn.Linear(128, 4)  # 4 действия

    def forward(self, x):
        # x shape: (Batch, 5, 5, 5)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)