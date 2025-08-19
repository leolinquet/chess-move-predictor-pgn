import argparse, os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .dataset import FENDataset
from .model import SmallConvPolicy
from .utils import get_device, head_acc, triplet_acc

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    ds = FENDataset(args.csv, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SmallConvPolicy().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    total = 0
    meters = {'from':0.0, 'to':0.0, 'promo':0.0, 'triplet':0.0}
    for x, targets in tqdm(loader, desc='eval'):
        x = x.to(device)
        tf = targets['from'].to(device)
        tt = targets['to'].to(device)
        tp = targets['promo'].to(device)
        out = model(x)
        bs = x.size(0)
        meters['from'] += head_acc(out['from'], tf) * bs
        meters['to'] += head_acc(out['to'], tt) * bs
        meters['promo'] += head_acc(out['promo'], tp) * bs
        meters['triplet'] += triplet_acc(out['from'], out['to'], out['promo'], tf, tt, tp) * bs
        total += bs

    for k in meters: meters[k] /= max(1,total)
    print(meters)

if __name__ == '__main__':
    main()
