import pexpect
import chess
import torch
from model import ChessNet
from utils import board_to_tensor


MAIA_PATH = "maia/maia_uci.exe"
MAIA_MODEL = "maia/maia-1100.pb.gz"


model = ChessNet()
model.load_state_dict(torch.load("checkpoint_epoch1.pth"))
model.eval()


maia = pexpect.spawn(f"{MAIA_PATH} --model {MAIA_MODEL}")
maia.expect("uci")
maia.sendline("isready")
maia.expect("readyok")


board = chess.Board()


while not board.is_game_over():
if board.turn: # our engine
x = board_to_tensor(board)
with torch.no_grad():
move_index = model(x).argmax().item()
move = list(board.legal_moves)[move_index % len(list(board.legal_moves))]
board.push(move)
print("Our move:", move)
else: # Maia
maia.sendline(f"position fen {board.fen()}")
maia.sendline("go movetime 200")
maia.expect("bestmove (.*)")
move = maia.match.group(1).decode()
board.push_uci(move)
print("Maia move:", move)


print("Game over! Result:", board.result())
