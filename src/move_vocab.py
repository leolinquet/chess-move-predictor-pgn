from typing import Tuple

FILES = "abcdefgh"
RANKS = "12345678"
PROMO_TO_IDX = {None:0, 'q':1, 'r':2, 'b':3, 'n':4}
IDX_TO_PROMO = {v:k for k,v in PROMO_TO_IDX.items()}

def square_to_idx(sq: str) -> int:
    file = FILES.index(sq[0])
    rank = RANKS.index(sq[1])
    return rank * 8 + file

def idx_to_square(idx: int) -> str:
    file = idx % 8
    rank = idx // 8
    return f"{FILES[file]}{RANKS[rank]}"

def parse_uci(uci: str) -> Tuple[int, int, int]:
    u = uci.strip().lower()
    if len(u) < 4: raise ValueError(f"Bad UCI: {uci}")
    f, t = u[:2], u[2:4]
    p = u[4] if len(u) >= 5 else None
    if p not in (None,'q','r','b','n'):
        raise ValueError(f"Bad promotion in UCI: {uci}")
    return square_to_idx(f), square_to_idx(t), PROMO_TO_IDX[p]

def compose_uci(from_idx: int, to_idx: int, promo_idx: int) -> str:
    f = idx_to_square(from_idx)
    t = idx_to_square(to_idx)
    p = {0:'',1:'q',2:'r',3:'b',4:'n'}[promo_idx]
    return f + t + p
