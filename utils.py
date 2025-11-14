import chess
import torch
import numpy as np


def board_to_tensor(board):
arr = np.zeros(773, dtype=np.float32)
arr[:773] = np.random.rand(773) # todo: replace with real encoding
return torch.tensor(arr)
