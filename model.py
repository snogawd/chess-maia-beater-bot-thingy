import torch,torch.nn as nn

class ResBlock(nn.Module):
 def __init__(self,d):super().__init__();self.fc1,self.fc2,self.act=nn.Linear(d,d),nn.Linear(d,d),nn.ReLU()
 def forward(self,x):return x+self.act(self.fc2(self.act(self.fc1(x))))

class ChessNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.net=nn.Sequential(nn.Linear(1152,512),nn.ReLU(),*[ResBlock(512) for _ in range(4)],nn.Linear(512,4672))
  for i,m in enumerate(self.net):
   if isinstance(m,ResBlock):self.net[i]=torch.compile(m,mode="reduce-overhead")
 def forward(self,x):return self.net(x)
