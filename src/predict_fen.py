import argparse, math, heapq
import torch
import torch.nn.functional as F
import chess

from .model import SmallConvPolicy
from .encoder import board_to_planes
from .move_vocab import compose_uci

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fen', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--legal', action='store_true', help='Mask to legal moves from given FEN.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    board = chess.Board(args.fen)
    x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(device)

    model = SmallConvPolicy().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    out = model(x)
    lpf = F.log_softmax(out['from'], dim=1).squeeze(0)
    lpt = F.log_softmax(out['to'],   dim=1).squeeze(0)
    lpp = F.log_softmax(out['promo'],dim=1).squeeze(0)

    n_from = 20; n_to = 20; n_p = 5
    topf = torch.topk(lpf, k=min(n_from, lpf.numel())).indices.tolist()
    topt = torch.topk(lpt, k=min(n_to,   lpt.numel())).indices.tolist()
    topp = torch.topk(lpp, k=min(n_p,    lpp.numel())).indices.tolist()

    legal = None
    if args.legal:
        legal = {m.uci() for m in board.legal_moves}

    candidates = []
    for f in topf:
        for t in topt:
            for p in topp:
                u = compose_uci(f,t,p)
                if legal is not None and u not in legal:
                    continue
                score = (lpf[f] + lpt[t] + lpp[p]).item()
                candidates.append((score, u))

    best = heapq.nlargest(args.topk, candidates, key=lambda tup: tup[0])

    print("Top predictions:")
    for score, u in best:
        print(f"{u}\t{math.exp(score):.4f}")

if __name__ == '__main__':
    main()
