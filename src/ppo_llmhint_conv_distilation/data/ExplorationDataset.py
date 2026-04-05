import torch
from torch.utils.data import Dataset

# --- Загрузка данных ---
class ExplorationDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.states = data['states']
        self.actions = data['actions']

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]