import torch,torch.nn as nn,torch.optim as optim
from model import ChessNet
from utils import iter_pgn_positions
import torch.amp

d,b,e,lr,pgn=torch.device("cuda" if torch.cuda.is_available() else "cpu"),64,10,1e-3,"data/elite_games_2020.pgn"
cuda_avail=torch.cuda.is_available()
if cuda_avail:
 torch.backends.cudnn.enabled=True
 props=torch.cuda.get_device_properties(d)
 has_tf32=props.compute_capability>=(7,5) if hasattr(props,"compute_capability") else False
 if has_tf32:torch.backends.cuda.matmul.fp32_precision="tf32";torch.backends.cudnn.conv.fp32_precision="tf32"
m=ChessNet().to(d)
pc=sum(p.numel() for p in m.parameters())
print(f"Params: {pc:,}|Device: {d}|bfloat16: {cuda_avail}")
c=nn.CrossEntropyLoss()
o=optim.AdamW(m.parameters(),lr=lr,fused=cuda_avail)
s=torch.amp.GradScaler(enabled=cuda_avail)
for ep in range(e):
 l_t=0;i=0
 for bt,p,v,il in iter_pgn_positions(pgn):
  bt,p=bt.flatten().to(d),p.to(d)
  with torch.amp.autocast("cuda" if cuda_avail else "cpu",dtype=torch.bfloat16 if cuda_avail else torch.float32):pred=m(bt.unsqueeze(0));l=c(pred,p.unsqueeze(0))
  o.zero_grad()
  s.scale(l).backward()
  s.unscale_(o)
  nn.utils.clip_grad_norm_(m.parameters(),1.0)
  s.step(o)
  s.update()
  l_t+=l.item();i+=1
  if i%1000==0:print(f"Ep {ep+1}/{e} Step {i} L: {l_t/i:.4f}")
  if i>=50000:break
 print(f"Ep {ep+1}/{e} Final L: {l_t/i:.4f}")
 torch.save(m.state_dict(),f"checkpoint_e{ep+1}.pth")
