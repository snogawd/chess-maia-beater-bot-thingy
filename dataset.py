import torch
import random
import numpy as np


class RandomChessDataset(torch.utils.data.Dataset):
def __init__(self, size=50000):
self.size = size


def __len__(self):
return self.size


def __getitem__(self, idx):
x = np.random.rand(773).astype(np.float32)
y = random.randint(0, 4671)
return torch.tensor(x), torch.tensor(y)
