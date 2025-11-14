import pexpect,chess,torch
from model import ChessNet
from utils import encode_board,move_to_index

d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda_avail=torch.cuda.is_available()
m=ChessNet().to(d);m.load_state_dict(torch.load("checkpoint_e1.pth",map_location=d));m.eval()
mai=pexpect.spawn("maia/lc0.exe --model maia/maia-1100.pb.gz");mai.expect("uci");mai.sendline("isready");mai.expect("readyok")
b=chess.Board()
while not b.is_game_over():
 if b.turn:
  x=encode_board(b).unsqueeze(0).to(d)
  with torch.no_grad():
   with torch.autocast(device_type="cuda" if cuda_avail else "cpu",dtype=torch.bfloat16 if cuda_avail else torch.float32):mv_idx=m(x).argmax().item()
  lm=list(b.legal_moves);mv=lm[mv_idx%len(lm)];b.push(mv);print(f"Our: {mv}")
 else:
  mai.sendline(f"position fen {b.fen()}");mai.sendline("go movetime 200");mai.expect("bestmove (.*)");mv=mai.match.group(1).decode();b.push_uci(mv);print(f"Maia: {mv}")
print(f"Result: {b.result()}")
