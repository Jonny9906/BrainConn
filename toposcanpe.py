import argparse
import hashlib
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from ALTER (https://github.com/yushuowiki/ALTER)
class ClusterAssignment(nn.Module):
    def __init__(self, k, dim):
        super().__init__()
        c = torch.zeros(k, dim, dtype=torch.float)
        nn.init.xavier_uniform_(c)
        o = torch.zeros(k, dim, dtype=torch.float)
        o[0] = c[0]
        for i in range(1, k):
            project = 0
            for j in range(i):
                project += (torch.dot(c[j], c[i]) / torch.dot(c[j], c[j])) * c[j]
            c[i] -= project
            o[i] = c[i] / torch.norm(c[i], p=2)
        self.centers = nn.Parameter(o, requires_grad=False)

    def forward(self, x):
        a = torch.pow(x @ self.centers.T, 2) / torch.norm(self.centers, p=2, dim=-1)
        return F.softmax(a, dim=-1)


class DEC(nn.Module):
    def __init__(self, k, dim, encoder):
        super().__init__()
        self.encoder = encoder
        self.assignment = ClusterAssignment(k, dim)

    def forward(self, x):
        b, n = x.size(0), x.size(1)
        h = self.encoder(x.view(b, -1)).view(b * n, -1)
        a = self.assignment(h).view(b, n, -1)
        return torch.bmm(a.transpose(1, 2), h.view(b, n, -1))


class EncoderLayer(nn.TransformerEncoderLayer):
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)


class TransPool(nn.Module):
    def __init__(self, d, n_in, n_out, pooling):
        super().__init__()
        self.transformer = EncoderLayer(d, 4, dim_feedforward=1024, batch_first=True, bias=False)
        self.pooling = pooling
        if pooling:
            enc = nn.Sequential(nn.Linear(d * n_in, 32), nn.LeakyReLU(), nn.Linear(32, 32),
                                nn.LeakyReLU(), nn.Linear(32, d * n_in))
            self.dec = DEC(n_out, d, enc)

    def forward(self, x):
        x = self.transformer(x)
        return self.dec(x) if self.pooling else x


class ROIScanEncoder(nn.Module):
    def __init__(self, n_slices, dropout, d=64):
        super().__init__()
        self.embed = nn.Linear(4, d)
        self.slice_emb = nn.Parameter(torch.zeros(1, n_slices, d))
        self.filt_emb = nn.Parameter(torch.zeros(1, 2, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        for p in (self.slice_emb, self.filt_emb, self.cls):
            nn.init.normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=256, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(d, d)

    def forward(self, s):
        M, S, _ = s.shape
        h = self.embed(torch.stack([s[..., :4], s[..., 4:]], 1))
        h = (h + self.slice_emb[:, :S].unsqueeze(1) + self.filt_emb.unsqueeze(2)).reshape(M, 2 * S, -1)
        h = torch.cat([self.cls.expand(M, -1, -1), h], 1)
        return self.out(self.tr(h)[:, 0])


class TopoScanPE(nn.Module):
    def __init__(self, n, n_slices, dropout, k=16, d=64):
        super().__init__()
        w = n + k
        while w % 4:
            w += 1
        self.pad = w - n - k
        self.roi = ROIScanEncoder(n_slices, dropout, d)
        self.pe_proj = nn.Linear(d, k)
        self.layers = nn.ModuleList([TransPool(w, n, n, False), TransPool(w, n, 100, True)])
        self.res_proj = nn.Linear(n, w)
        self.dim_reduction = nn.Sequential(nn.Linear(w, 8), nn.LeakyReLU())
        self.fc = nn.Sequential(nn.Linear(800, 256), nn.LeakyReLU(), nn.Linear(256, 32),
                                nn.LeakyReLU(), nn.Linear(32, 2))

    def forward(self, corr, scan, mix=None):
        B, N = corr.shape[0], corr.shape[1]
        if mix is not None:
            lam, idx = mix
            scan = lam * scan + (1 - lam) * scan[idx]
        pe = self.pe_proj(self.roi(scan.reshape(B * N, scan.shape[2], scan.shape[3])).reshape(B, N, -1))
        pe = pe * (corr.std(dim=(-1, -2), keepdim=True) / pe.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6))
        z = torch.cat([corr, pe] + ([corr.new_zeros(B, N, self.pad)] if self.pad else []), -1)
        r = self.res_proj(corr)
        for layer in self.layers:
            z = layer(z)
            if z.shape[1] == r.shape[1]:
                z = z + r
        return self.fc(self.dim_reduction(z).reshape(B, -1))


def betti01(A):
    A = np.array(A, dtype=bool, copy=True)
    n = A.shape[0]
    if n == 0:
        return 0, 0
    np.fill_diagonal(A, False)
    b0 = int(connected_components(csr_matrix(A), directed=False)[0])
    iu, ju = np.nonzero(np.triu(A, 1))
    if len(iu) == 0:
        return b0, 0
    eid = {(u, v): k for k, (u, v) in enumerate(zip(iu.tolist(), ju.tolist()))}
    pivots, rank = {}, 0
    for u, v in zip(iu.tolist(), ju.tolist()):
        for w in np.flatnonzero(A[u] & A[v]):
            w = int(w)
            if w <= v:
                continue
            r = (1 << eid[(u, v)]) | (1 << eid[(u, w)]) | (1 << eid[(v, w)])
            while r:
                p = r.bit_length() - 1
                if p in pivots:
                    r ^= pivots[p]
                else:
                    pivots[p] = r
                    rank += 1
                    break
    return b0, len(iu) - n + b0 - rank


def knn_graph(corr, k=8):
    N = corr.shape[0]
    W = np.array(corr, dtype=np.float64, copy=True)
    np.fill_diagonal(W, -np.inf)
    idx = np.argpartition(-W, k, axis=1)[:, :k]
    A = np.zeros((N, N), bool)
    rows = np.repeat(np.arange(N), k)
    A[rows, idx.ravel()] = W[rows, idx.ravel()] > 0
    return A | A.T


def two_hop(A, u, cap):
    one = np.flatnonzero(A[u])
    one = one[one != u]
    two = np.flatnonzero(A[one].any(0)) if len(one) else np.array([], int)
    ball = np.unique(np.concatenate([[u], one, two]))

    def top(cand, room):
        cand = cand[cand != u]
        if room <= 0 or len(cand) == 0:
            return np.array([], int)
        return cand[np.argsort(-A[np.ix_(cand, ball)].sum(1), kind="stable")[:room]]

    if len(one) >= cap:
        keep = top(one, cap)
    else:
        keep = np.concatenate([one, top(np.setdiff1d(ball, np.concatenate([[u], one])), cap - len(one))])
    return np.unique(np.concatenate([[u], keep]))


def forman(A):
    deg = A.sum(1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        nbr = (A.astype(np.float64) @ deg) / np.maximum(deg, 1)
    return np.where(deg > 0, 4.0 - deg - nbr, 0.0)


def hks(A, t=1.0):
    Af = A.astype(np.float64)
    d = Af.sum(1)
    dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    lam, phi = np.linalg.eigh(np.eye(A.shape[0]) - dinv[:, None] * Af * dinv[None, :])
    return (phi ** 2 * np.exp(-t * lam)[None, :]).sum(1)


def scan(A, f, n_slices, m=2):
    grid = np.unique(np.quantile(f, np.linspace(0, 1, n_slices + m)))
    out = np.zeros((n_slices, 4), np.float32)
    for i in range(min(max(len(grid) - m, 0), n_slices)):
        idx = np.where((f >= grid[i]) & (f <= grid[i + m]))[0]
        if len(idx) == 0:
            continue
        S = A[np.ix_(idx, idx)]
        out[i] = (*betti01(S), len(idx), int(S.sum() // 2))
    return out


def toposcan_subject(corr, node_frac, n_slices):
    A = knn_graph(corr)
    N = A.shape[0]
    cap = max(int(round(node_frac * N)), 3)
    out = np.zeros((N, n_slices, 8), np.float32)
    for u in range(N):
        nodes = two_hop(A, u, cap)
        S = A[np.ix_(nodes, nodes)]
        if S.sum() == 0:
            continue
        out[u, :, :4] = scan(S, forman(S), n_slices)
        out[u, :, 4:] = scan(S, hks(S), n_slices)
    return out


def toposcan_features(corr_np, node_frac, n_slices, cache_dir):
    fp = hashlib.md5(np.ascontiguousarray(corr_np, np.float32).tobytes()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"toposcan_{fp}_f{node_frac}_s{n_slices}.npy")
    if os.path.exists(path):
        return np.load(path)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        res = list(ex.map(partial(toposcan_subject, node_frac=node_frac, n_slices=n_slices),
                          [c.astype(np.float64) for c in corr_np], chunksize=4))
    out = np.stack(res).astype(np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, out)
    return out


def scale(X, train):
    Z = np.log1p(np.clip(np.asarray(X, np.float64), 0, None))
    mu, sd = Z[train].mean((0, 1)), Z[train].std((0, 1))
    keep = sd > 1e-6 * np.maximum(1.0, np.abs(mu))
    return (((Z - mu) / np.where(keep, sd, 1.0)) * keep).astype(np.float32)


# from BQN (https://github.com/LYWJUN/BQN-demo)
def split(labels, site, run, split_dir):
    path = os.path.join(split_dir, f"split_r{run}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["train"], z["val"], z["test"]
    n = len(labels)
    n_train, n_val = int(n * 0.7), int(n * 0.1)
    tr, rest = next(StratifiedShuffleSplit(1, train_size=n_train, test_size=n - n_train,
                                           random_state=42).split(np.zeros(n), site))
    va, te = next(StratifiedShuffleSplit(1, test_size=n - n_train - n_val).split(rest, site[rest]))
    os.makedirs(split_dir, exist_ok=True)
    np.savez(path, train=tr, val=rest[va], test=rest[te])
    return tr, rest[va], rest[te]


# from ALTER (https://github.com/yushuowiki/ALTER)
def evaluate(model, loader):
    model.eval()
    loss, acc, n, ys, ps = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        for corr, scan_, y in loader:
            corr, scan_, y = corr.to(DEVICE), scan_.to(DEVICE), y.to(DEVICE).float()
            out = model(corr, scan_)
            loss += F.cross_entropy(out, y, reduction="sum").item() * len(y)
            acc += out.topk(1, 1)[1].view(-1).eq(y[:, 1]).float().sum().mul_(100.0 / len(y)).item() * len(y)
            n += len(y)
            ps += F.softmax(out, 1)[:, 1].tolist()
            ys += y[:, 1].tolist()
    ys, ps = np.array(ys), np.array(ps)
    hard = ps > 0.5
    return {"loss": loss / n, "AUC": roc_auc_score(ys, ps), "ACC": acc / n,
            "SEN": (hard & (ys == 1)).sum() / (ys == 1).sum(),
            "SPE": (~hard & (ys == 0)).sum() / (ys == 0).sum()}


def fit(model, loaders, epochs):
    train, val, test = loaders
    no_wd = [p for n, p in model.named_parameters()
             if any(k in n for k in ("cls", "slice_emb", "filt_emb")) and p.requires_grad]
    ids = {id(p) for p in no_wd}
    opt = torch.optim.Adam([{"params": [p for p in model.parameters() if id(p) not in ids]},
                            {"params": no_wd, "weight_decay": 0.0}], lr=0.0, weight_decay=1e-4)
    total = ((len(train.dataset) - 1) // train.batch_size + 1) * epochs
    step, hist = 0, []
    for epoch in range(epochs):
        model.train()
        for corr, scan_, y in train:
            corr, scan_, y = corr.to(DEVICE), scan_.to(DEVICE), y.to(DEVICE).float()
            step += 1
            for g in opt.param_groups:
                g["lr"] = 1e-5 + (1e-4 - 1e-5) * (1 + math.cos(math.pi * (step / total))) / 2
            lam = float(np.random.beta(1.0, 1.0))
            idx = torch.randperm(len(y)).to(DEVICE)
            corr = lam * corr + (1 - lam) * corr[idx, :]
            y = lam * y + (1 - lam) * y[idx]
            loss = F.cross_entropy(model(corr, scan_, (lam, idx)), y, reduction="sum")
            opt.zero_grad()
            loss.backward()
            opt.step()
        v, t = evaluate(model, val), evaluate(model, test)
        hist.append({"vloss": v["loss"], **t})
        print(f"epoch {epoch}  val loss {v['loss']:.3f}  test AUC {t['AUC']:.4f}", flush=True)
    return hist


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run(r, args, corr, labels, site, X):
    seed_all(r)
    sp = split(labels, site, r, os.path.join(args.out, "splits"))
    T = torch.from_numpy(scale(X, sp[0]))
    y = F.one_hot(labels.to(torch.int64))
    ds = lambda ix: utils.TensorDataset(*[t[torch.as_tensor(ix).long()] for t in (corr, T, y)])
    loaders = [utils.DataLoader(ds(sp[0]), batch_size=16, shuffle=True, drop_last=True),
               utils.DataLoader(ds(sp[1]), batch_size=16),
               utils.DataLoader(ds(sp[2]), batch_size=16)]
    model = TopoScanPE(corr.shape[1], T.shape[2], args.dropout).to(DEVICE)
    seed_all(r)
    hist = fit(model, loaders, args.epochs)
    # BQN: test metrics at the epoch with the lowest validation loss
    best = hist[int(np.argmin([h["vloss"] for h in hist]))]
    return {"split": r, **{m: best[m] * (1 if m == "ACC" else 100) for m in ("AUC", "ACC", "SEN", "SPE")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="abide.npy")
    ap.add_argument("--out", default="results")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--node_frac", type=float, default=0.4)
    ap.add_argument("--slices", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True).item()
    corr, labels = torch.from_numpy(d["corr"]).float(), torch.from_numpy(d["label"]).float()
    X = toposcan_features(corr.numpy(), args.node_frac, args.slices, os.path.join(args.out, "cache"))
    rows = [run(r, args, corr, labels, np.asarray(d["site"]), X) for r in range(args.splits)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    print("  ".join(f"{m} {df[m].mean():.2f}±{df[m].std(ddof=0):.2f}" for m in ("AUC", "ACC", "SEN", "SPE")))


if __name__ == "__main__":
    main()
