import os, random
import numpy as np
import torch
import yaml

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_config(out_dir: str, cfg: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)

@torch.no_grad()
def head_acc(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

@torch.no_grad()
def triplet_acc(logits_from, logits_to, logits_promo, targets_from, targets_to, targets_promo):
    p_from  = logits_from.argmax(dim=1)
    p_to    = logits_to.argmax(dim=1)
    p_promo = logits_promo.argmax(dim=1)
    ok = (p_from==targets_from) & (p_to==targets_to) & (p_promo==targets_promo)
    return ok.float().mean().item()
