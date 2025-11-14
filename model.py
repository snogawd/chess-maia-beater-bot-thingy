import torch
import torch.nn as nn


class ChessNet(nn.Module):
def __init__(self):
super().__init__()
self.net = nn.Sequential(
nn.Linear(773, 512),
nn.ReLU(),
nn.Linear(512, 512),
nn.ReLU(),
nn.Linear(512, 4672) # all legal moves
)


def forward(self, x):
return self.net(x)
