import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.ppo_llmhint_conv_distilation.data.ExplorationDataset import ExplorationDataset
from src.ppo_llmhint_conv_distilation.ml_model.ExplorerCNN import ExplorerCNN

SIZE = 10
_DIR = Path(__file__).parent

# --- Обучение ---
print("Начинаем обучение студента...")
dataset = ExplorationDataset(str(_DIR.parent / "data" / f"dataset_{SIZE}size.pt"))
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = ExplorerCNN(map_size=SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    total_loss = 0
    correct = 0
    total = 0

    for states, labels in loader:
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch + 1}/200 | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

save_path = _DIR / f"student_cnn_{SIZE}size.pth"
torch.save(model.state_dict(), str(save_path))
print(f"Модель сохранена как {save_path}")