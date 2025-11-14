import torch,random,numpy as np

class RandomChessDataset(torch.utils.data.Dataset):
 def __init__(self,size=50000):self.size=size
 def __len__(self):return self.size
 def __getitem__(self,idx):return torch.tensor(np.random.rand(773).astype(np.float32)),random.randint(0,4671)
