import torch,torch.nn as nn,torch.optim as optim
from model import ChessNet
from dataset import RandomChessDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler

d,b,e,lr=torch.device("cuda" if torch.cuda.is_available() else "cpu"),64,10,1e-3
cuda_avail=torch.cuda.is_available()
if cuda_avail:
 torch.backends.cudnn.enabled=True
 props=torch.cuda.get_device_properties(d)
 has_tf32=hasattr(props,"compute_capability") and props.compute_capability>=(7,5)
 torch.backends.cuda.matmul.allow_tf32=has_tf32
 torch.backends.cudnn.allow_tf32=has_tf32
m=ChessNet().to(d)
pc=sum(p.numel() for p in m.parameters())
print(f"Params: {pc:,}|Device: {d}|bfloat16: {cuda_avail}")
dl=DataLoader(RandomChessDataset(),batch_size=b,shuffle=True,num_workers=2,pin_memory=cuda_avail)
c=nn.CrossEntropyLoss()
o=optim.AdamW(m.parameters(),lr=lr,fused=cuda_avail)
s=GradScaler(enabled=cuda_avail)
for ep in range(e):
 l_t=0
 for x,y in dl:
  x,y=x.to(d),y.to(d)
  with autocast(device_type="cuda" if cuda_avail else "cpu",dtype=torch.bfloat16 if cuda_avail else torch.float32):p=m(x);l=c(p,y)
  o.zero_grad()
  s.scale(l).backward()
  s.unscale_(o)
  nn.utils.clip_grad_norm_(m.parameters(),1.0)
  s.step()
  l_t+=l.item()
 print(f"Ep {ep+1}/{e} L: {l_t/len(dl):.4f}")
 torch.save(m.state_dict(),f"checkpoint_e{ep+1}.pth")
