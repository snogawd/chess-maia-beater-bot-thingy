import torch
from torch import nn, optim
from model import ChessNet
from dataset import RandomChessDataset
from torch.utils.data import DataLoader


model = ChessNet()
loader = DataLoader(RandomChessDataset(), batch_size=64, shuffle=True)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


epochs = 3
for epoch in range(epochs):
for x, y in loader:
optimizer.zero_grad()
preds = model(x)
loss = criterion(preds, y)
loss.backward()
optimizer.step()
torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")
print(f"Epoch {epoch+1} complete")
