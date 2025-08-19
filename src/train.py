import argparse, os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import FENDataset
from .model import SmallConvPolicy
from .utils import set_seed, get_device, save_config, head_acc, triplet_acc

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0
    meters = {'loss':0.0, 'from':0.0, 'to':0.0, 'promo':0.0, 'triplet':0.0}
    pbar = tqdm(loader, desc='train', leave=False)
    for x, targets in pbar:
        x = x.to(device)
        tf = targets['from'].to(device)
        tt = targets['to'].to(device)
        tp = targets['promo'].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(x)
            loss = ce(out['from'], tf) + ce(out['to'], tt) + 0.5*ce(out['promo'], tp)
        if scaler is not None:
            scaler.scale(loss).step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        with torch.no_grad():
            bs = x.size(0)
            meters['loss'] += loss.item() * bs
            meters['from'] += head_acc(out['from'], tf) * bs
            meters['to'] += head_acc(out['to'], tt) * bs
            meters['promo'] += head_acc(out['promo'], tp) * bs
            meters['triplet'] += triplet_acc(out['from'], out['to'], out['promo'], tf, tt, tp) * bs
            total += bs
            pbar.set_postfix({k: f"{meters[k]/max(1,total):.3f}" for k in meters})
    for k in meters: meters[k] /= max(1,total)
    return meters

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0
    meters = {'loss':0.0, 'from':0.0, 'to':0.0, 'promo':0.0, 'triplet':0.0}
    for x, targets in tqdm(loader, desc='val', leave=False):
        x = x.to(device)
        tf = targets['from'].to(device)
        tt = targets['to'].to(device)
        tp = targets['promo'].to(device)
        out = model(x)
        loss = ce(out['from'], tf) + ce(out['to'], tt) + 0.5*ce(out['promo'], tp)
        bs = x.size(0)
        meters['loss'] += loss.item() * bs
        meters['from'] += head_acc(out['from'], tf) * bs
        meters['to'] += head_acc(out['to'], tt) * bs
        meters['promo'] += head_acc(out['promo'], tp) * bs
        meters['triplet'] += triplet_acc(out['from'], out['to'], out['promo'], tf, tt, tp) * bs
        total += bs
    for k in meters: meters[k] /= max(1,total)
    return meters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV with columns fen,uci')
    ap.add_argument('--out', required=True, help='Output dir')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-split', type=float, default=0.1)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='Optional cap on number of rows used. 0 = all.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    save_config(args.out, vars(args))
    set_seed(args.seed)
    device = get_device()

    fullset = FENDataset(args.csv, limit=args.limit)

    # split
    n_total = len(fullset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    trainset, valset = random_split(fullset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SmallConvPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best = {'triplet': -1.0}
    for epoch in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, optimizer, device, scaler)
        va = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train={tr} val={va}")

        torch.save({'model':model.state_dict(), 'epoch':epoch}, os.path.join(args.out, 'latest.pt'))
        if va['triplet'] > best['triplet']:
            best = va
            torch.save({'model':model.state_dict(), 'epoch':epoch}, os.path.join(args.out, 'best.pt'))
            print(f"[+] Saved new best.pt (triplet_acc={va['triplet']:.4f})")

if __name__ == '__main__':
    main()
