import pandas as pd
import torch
from torch.utils.data import Dataset
import chess
from .encoder import board_to_planes
from .move_vocab import parse_uci

class FENDataset(Dataset):
    def __init__(self, csv_path: str, limit: int = 0):
        self.df = pd.read_csv(csv_path)
        if limit > 0:
            self.df = self.df.iloc[:limit].copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fen = row["fen"]
        uci = row["uci"]
        board = chess.Board(fen)
        x = torch.from_numpy(board_to_planes(board))  # [25,8,8], float32
        f_idx, t_idx, p_idx = parse_uci(uci)
        y_from  = torch.tensor(f_idx, dtype=torch.long)
        y_to    = torch.tensor(t_idx, dtype=torch.long)
        y_promo = torch.tensor(p_idx, dtype=torch.long)
        return x, {"from": y_from, "to": y_to, "promo": y_promo}
