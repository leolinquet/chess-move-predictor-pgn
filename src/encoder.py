from typing import Tuple
import numpy as np
import chess

# Planes: 12 piece planes + 1 side-to-move + 4 castling + 8 en-passant file = 25
PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
]

def _square_to_coords(sq: int) -> Tuple[int,int]:
    # python-chess: a1=0 -> (rank=0,file=0); h8=63 -> (7,7)
    r = sq // 8
    f = sq % 8
    return r, f

def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((25, 8, 8), dtype=np.float32)

    # 12 piece planes
    for i, p in enumerate(PIECE_ORDER):
        bb_w = board.pieces(p, chess.WHITE)
        bb_b = board.pieces(p, chess.BLACK)
        for sq in bb_w:
            r, f = _square_to_coords(sq)
            planes[i, r, f] = 1.0
        for sq in bb_b:
            r, f = _square_to_coords(sq)
            planes[6+i, r, f] = 1.0

    # side to move (all ones if white to move, zeros otherwise)
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # castling rights
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # en-passant file one-hot (8 planes)
    if board.ep_square is not None:
        file_idx = board.ep_square % 8
        planes[17 + file_idx, :, :] = 1.0

    return planes
