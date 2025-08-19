# Chess Move Predictor (PGN → FEN → Move)
Minimal PyTorch project that trains a *policy head* to predict the next move
from chess **positions extracted from PGN files** (e.g., Lichess/TWIC).

**Pipeline**
1) `pgn_to_csv.py` — read one or more `.pgn` (or `.pgn.zst`) files and write a CSV: `fen,uci`
2) `train.py` — loads that CSV, encodes FEN to 25×8×8 planes, trains 3-head classifier:
   - head 1: from-square (64 classes)
   - head 2: to-square (64 classes)
   - head 3: promotion (5 classes: none,q,r,b,n)
3) `predict_fen.py` — given a FEN (and optionally enforce legal moves) output top‑K UCIs.

> This baseline does **not** use images. Inputs are board planes created from FENs.

---

## 1) Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Get PGNs (examples)
- Lichess monthly PGNs (`.pgn.zst`): database.lichess.org
- Lichess **Elite** Database (filtered to high‑rated games)
- TWIC (The Week in Chess) weekly PGNs

> You can stream `.zst` files without fully decompressing: `zstdcat file.pgn.zst | python -m src.pgn_to_csv --stdin --out data/dataset.csv`

## 3) Build dataset (CSV of positions)
```bash
# Read one or more PGN files or an entire folder
python -m src.pgn_to_csv   --inputs data/pgn/Jan24.pgn.zst data/pgn/Feb24.pgn.zst   --out data/dataset.csv   --min-elo 2400   --exclude-bullet   --max-games 200000    # optional cap for a first run
```

The output CSV has two columns: `fen,uci`. Each row is a single training example
(position BEFORE the move).

## 4) Train
```bash
python -m src.train   --csv data/dataset.csv   --out runs/run1   --epochs 5   --batch-size 256
```

Tips:
- On CPU, start with `--epochs 1 --batch-size 64` to smoke test.
- You can limit rows used with `--limit 200000` for quick iterations.

## 5) Evaluate
```bash
python -m src.eval --csv data/dataset.csv --ckpt runs/run1/best.pt
```

## 6) Predict from a FEN
```bash
python -m src.predict_fen   --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"   --ckpt runs/run1/best.pt   --topk 5   --legal          # restrict to legal moves from that position
```

---

## Encoder (25 planes)
- 12 piece planes: white {P,N,B,R,Q,K} + black {p,n,b,r,q,k}
- 1 side-to-move plane
- 4 castling-rights planes: WK, WQ, BK, BQ
- 8 en-passant file planes (one-hot by file when EP square exists)

---

## Notes
- Factorized heads scale better than a single 4672-class softmax.
- Training uses simple CNN with global average pooling; easy to swap for a ResNet or transformer.
- This baseline predicts **policy** only, not value / legality / move ordering beyond optional mask.
