"""Bidirectional LSTM sequence model, masked L1 on inspiratory steps.

Usage: python train.py --fold 0 [--epochs 250] [--resume]
Saves checkpoints/fold{k}_{last,best}.pt, oof_fold{k}.npy, test_fold{k}.npy.
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from folds import get_folds

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, required=True)
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--bs", type=int, default=512)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="lstm")
p.add_argument("--resume", action="store_true")
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"
CK = "checkpoints"; os.makedirs(CK, exist_ok=True)
pref = f"{CK}/{a.tag}_fold{a.fold}"

X = np.load("cache/Xtr.npy"); y = np.load("cache/ytr.npy"); m = np.load("cache/mtr.npy")
Xte = np.load("cache/Xte.npy")
fold = get_folds(np.load("cache/rc_tr.npy"))
tr, va = fold != a.fold, fold == a.fold
Xtr_t = torch.tensor(X[tr], device=dev); ytr_t = torch.tensor(y[tr], device=dev); mtr_t = torch.tensor(m[tr], device=dev)
Xva_t = torch.tensor(X[va], device=dev); yva = y[va]; mva = m[va]
Xte_t = torch.tensor(Xte, device=dev)


class Net(nn.Module):
    def __init__(self, nf, h, L):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, h), nn.LayerNorm(h), nn.GELU())
        self.lstms = nn.ModuleList()
        d = h
        for i in range(L):
            self.lstms.append(nn.LSTM(d, h // (2 ** min(i, 1)), batch_first=True, bidirectional=True))
            d = 2 * (h // (2 ** min(i, 1)))
        self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.inp(x)
        for l in self.lstms:
            x, _ = l(x)
        return self.head(x).squeeze(-1)


net = Net(X.shape[-1], a.hidden, a.layers).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
steps_per_epoch = int(np.ceil(tr.sum() / a.bs))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.05)
start, best = 0, 1e9
if a.resume and os.path.exists(pref + "_last.pt"):
    s = torch.load(pref + "_last.pt")
    net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    start, best = s["epoch"] + 1, s["best"]
    print(f"resumed at epoch {start}, best {best:.4f}", flush=True)


@torch.no_grad()
def predict(Xt):
    net.eval()
    return torch.cat([net(Xt[i:i + 2048]) for i in range(0, len(Xt), 2048)]).cpu().numpy()


n = len(Xtr_t)
for ep in range(start, a.epochs):
    t0 = time.time(); net.train(); perm = torch.randperm(n, device=dev); tot = 0.0
    for i in range(0, n, a.bs):
        idx = perm[i:i + a.bs]
        out = net(Xtr_t[idx]); mk = mtr_t[idx]
        loss = (out - ytr_t[idx]).abs()[mk].mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step(); tot += loss.item() * len(idx)
    pv = predict(Xva_t); vmae = np.abs(pv - yva)[mva].mean()
    if vmae < best:
        best = vmae
        torch.save(net.state_dict(), pref + "_best.pt")
        np.save(pref + "_oof.npy", pv); np.save(pref + "_test.npy", predict(Xte_t))
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep, "best": best}, pref + "_last.pt")
    print(f"ep {ep:3d} train {tot / n:.4f} val {vmae:.4f} best {best:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.1f}s", flush=True)
print(f"DONE fold {a.fold} best val MAE {best:.4f}", flush=True)
