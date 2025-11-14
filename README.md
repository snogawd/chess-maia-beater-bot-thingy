# Chess ResNet to Beat Maia 1100

Train a neural network without search to defeat Maia 1100 ELO.

### Architecture

- **ResNet with 4 residual blocks** (512 hidden dim) for improved gradient flow and pattern learning
- **Torch.compile on residual blocks** with `reduce-overhead` mode to fuse computations and reduce kernel launch overhead

### Training Optimizations

- **CUDA autodetection** with automatic device selection (falls back to CPU)
- **TF32 precision** auto-enabled on Ampere+ GPUs (compute capability ≥7.5) for ~3-5x speedup
- **bfloat16 mixed precision** via `GradScaler` for memory efficiency and faster convergence
- **AdamW optimizer** with gradient clipping (norm=1.0) for stable training - also fused for speed
- **Real PGN data** loaded directly from elite games via `iter_pgn_positions()` in utils
- **Deleted dataset.py** - was just random dummy data; replaced with actual Lichess elite games

### Evaluation

- **BF16 inference** via `torch.autocast` for faster evaluation
- **Direct UCI evaluation** against Maia 1100 engine

### Setup

1. Download `maia-1100.pb.gz` and place in `maia/`
2. Download the Maia UCI engine and place in `maia/`
3. Install dependencies: `pip install -r requirements.txt`
4. Train: `python train.py`
5. Evaluate: `python evaluate_vs_maia.py`

### Data

- Use `python download-data.py --year YYYY` to download elite games from Lichess
- `utils.py` contains board encoding, move indexing, and PGN parsing for dataset creation

### You need to download lc0 which runs the maia weights 

- https://lczero.org/play/download/