import argparse
import hashlib
import logging
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch.nn import Parameter, TransformerEncoderLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KNN = 8
WIDTH_M = 2
HKS_T = 1.0


# ======================================================================================
# Taken from ALTER (https://github.com/yushuowiki/ALTER), alter/models/LRBGT/.
# ClusterAssignment and DEC were taken by ALTER from https://github.com/vlukiyanov/pt-dec.
# Unused branches and methods removed.
# ======================================================================================

# ALTER: ptdec/cluster.py
class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha=1.0, cluster_centers=None,
                 orthogonal=True, freeze_center=True, project_assignment=True):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.project_assignment = project_assignment
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers

        if orthogonal:
            orthogonal_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float)
            orthogonal_cluster_centers[0] = initial_cluster_centers[0]
            for i in range(1, cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(initial_cluster_centers[j], initial_cluster_centers[i])
                initial_cluster_centers[i] -= project
                orthogonal_cluster_centers[i] = initial_cluster_centers[i] / \
                    torch.norm(initial_cluster_centers[i], p=2)
            initial_cluster_centers = orthogonal_cluster_centers

        self.cluster_centers = Parameter(initial_cluster_centers, requires_grad=(not freeze_center))

    @staticmethod
    def project(u, v):
        return (torch.dot(u, v) / torch.dot(u, u)) * u

    def forward(self, batch):
        assignment = batch @ self.cluster_centers.T
        assignment = torch.pow(assignment, 2)
        norm = torch.norm(self.cluster_centers, p=2, dim=-1)
        soft_assign = assignment / norm
        return F.softmax(soft_assign, dim=-1)


# ALTER: ptdec/dec.py
class DEC(nn.Module):
    def __init__(self, cluster_number, hidden_dimension, encoder, alpha=1.0, orthogonal=True,
                 freeze_center=True, project_assignment=True):
        super().__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha, orthogonal=orthogonal,
            freeze_center=freeze_center, project_assignment=project_assignment)

    def forward(self, batch):
        node_num = batch.size(1)
        batch_size = batch.size(0)
        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        encoded = encoded.view(batch_size * node_num, -1)
        assignment = self.assignment(encoded)
        assignment = assignment.view(batch_size, node_num, -1)
        encoded = encoded.view(batch_size, node_num, -1)
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment


# ALTER: components/transformer_encoder.py (is_causal added for PyTorch >= 2.0)
class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, weights = self.self_attn(x, x, x, attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask, need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)


# ALTER: lrbgt.py
class TransPoolingEncoder(nn.Module):
    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num,
                 pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size,
                           encoder=self.encoder, orthogonal=orthogonal,
                           freeze_center=freeze_center, project_assignment=project_assignment)

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None


# ======================================================================================
# Taken from ALTER (https://github.com/yushuowiki/ALTER):
#   Train             <- alter/training/training.py  (wandb logging and learnable-matrix export removed;
#                                                      train_per_epoch / test_per_epoch are in TSPETrain)
#   LRScheduler       <- alter/components/lr_scheduler.py  (cosine mode only)
#   logger_factory    <- alter/components/logger.py
#   TotalMeter        <- alter/utils/meter.py
#   accuracy, isfloat <- alter/utils/accuracy.py
#   count_params      <- alter/utils/count_params.py
# ======================================================================================

class TotalMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def update_with_weight(self, val, count):
        self.sum += val * count
        self.count += count

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        if self.count == 0:
            return -1
        return self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)
    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))
    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class LRScheduler:
    def __init__(self, base_lr, target_lr, total_steps):
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.total_steps = total_steps
        self.lr = base_lr

    def update(self, optimizer, step):
        assert 0 <= step <= self.total_steps
        current_ratio = step / self.total_steps
        cosine = math.cos(math.pi * current_ratio)
        self.lr = self.target_lr + (self.base_lr - self.target_lr) * (1 + cosine) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


def get_formatter():
    return logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')


def initialize_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(get_formatter())
    logger.addHandler(stream_handler)
    return logger


def logger_factory(config):
    log_path = Path(config.log_path) / config.unique_id
    log_path.mkdir(exist_ok=True, parents=True)
    logger = initialize_logger()
    file_handler = logging.FileHandler(str(log_path / config.unique_id))
    file_handler.setFormatter(get_formatter())
    logger.addHandler(file_handler)
    return logger


class Train:

    def __init__(self, cfg, model, optimizers, lr_schedulers, dataloaders, logger):
        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy = [TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss]:
            meter.reset()

    def save_result(self, results):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "training_process.npy", results, allow_pickle=True)
        torch.save(self.model.state_dict(), self.save_path / "model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_per_epoch(self.val_dataloader, self.val_loss, self.val_accuracy)
            test_result = self.test_per_epoch(self.test_dataloader, self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })

        self.save_result(training_process)


# ======================================================================================
# Data
# ======================================================================================

# Adapted from ALTER (https://github.com/yushuowiki/ALTER), alter/dataset/abide.py.
# Only the correlation matrices, labels and sites are loaded; time series are not used.
def load_abide(path):
    data = np.load(path, allow_pickle=True).item()
    corr, labels = [torch.from_numpy(d).float() for d in (data["corr"], data["label"])]
    return corr, labels, np.asarray(data["site"])


# Taken from BQN (https://github.com/LYWJUN/BQN-demo), data_utils.py: init_stratified_dataloader.
# 7:1:2 split stratified by site; the training set is fixed by seed 42 and the
# validation/test sets are redrawn for every run (validation 100, test 203 on ABIDE).
def stratified_split(labels, site, seed=42, train_prop=0.7, val_prop=0.1):
    length = len(labels)
    train_length = int(length * train_prop)
    val_length = int(length * val_prop)
    test_length = length - train_length - val_length
    split1 = StratifiedShuffleSplit(n_splits=1, train_size=train_length,
                                    test_size=length - train_length, random_state=seed)
    train_index, rest_index = next(split1.split(np.zeros(length), site))
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length)
    val_pos, test_pos = next(split2.split(rest_index, site[rest_index]))
    return train_index, rest_index[val_pos], rest_index[test_pos]


def get_split(labels, site, run, split_dir):
    path = os.path.join(split_dir, f"split_r{run}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["train"], z["val"], z["test"]
    train_index, val_index, test_index = stratified_split(labels, site)
    os.makedirs(split_dir, exist_ok=True)
    np.savez(path, train=train_index, val=val_index, test=test_index)
    return train_index, val_index, test_index


def make_loaders(corr, scan, labels, split, batch_size=16):
    train_index, val_index, test_index = split
    y = F.one_hot(labels.to(torch.int64))
    ds = lambda ix: utils.TensorDataset(*[t[torch.as_tensor(ix).long()] for t in (corr, scan, y)])
    return [utils.DataLoader(ds(train_index), batch_size=batch_size, shuffle=True, drop_last=True),
            utils.DataLoader(ds(val_index), batch_size=batch_size, shuffle=False),
            utils.DataLoader(ds(test_index), batch_size=batch_size, shuffle=False)]


# ======================================================================================
# Topo-Scan features
# ======================================================================================

# (beta_0, beta_1) of the clique complex over Z/2
def betti01(A):
    A = np.array(A, dtype=bool, copy=True)
    n = A.shape[0]
    if n == 0:
        return 0, 0
    np.fill_diagonal(A, False)
    b0 = int(connected_components(csr_matrix(A), directed=False)[0])
    iu, ju = np.nonzero(np.triu(A, 1))
    n_edges = len(iu)
    if n_edges == 0:
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
    return b0, n_edges - n + b0 - rank


def knn_graph(corr, k):
    N = corr.shape[0]
    W = np.array(corr, dtype=np.float64, copy=True)
    np.fill_diagonal(W, -np.inf)
    idx = np.argpartition(-W, k, axis=1)[:, :k]
    A = np.zeros((N, N), bool)
    rows = np.repeat(np.arange(N), k)
    A[rows, idx.ravel()] = W[rows, idx.ravel()] > 0
    return A | A.T


def two_hop(A, u, max_nbrs):
    one = np.flatnonzero(A[u])
    one = one[one != u]
    two = np.flatnonzero(A[one].any(0)) if len(one) else np.array([], int)
    ball = np.unique(np.concatenate([[u], one, two]))

    def by_ball_degree(cand, room):
        cand = cand[cand != u]
        if room <= 0 or len(cand) == 0:
            return np.array([], int)
        deg = A[np.ix_(cand, ball)].sum(1)
        return cand[np.argsort(-deg, kind="stable")[:room]]

    if len(one) >= max_nbrs:
        keep = by_ball_degree(one, max_nbrs)
    else:
        rest = np.setdiff1d(ball, np.concatenate([[u], one]))
        keep = np.concatenate([one, by_ball_degree(rest, max_nbrs - len(one))])
    return np.unique(np.concatenate([[u], keep]))


def forman_node(A):
    deg = A.sum(1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        nbr_deg = (A.astype(np.float64) @ deg) / np.maximum(deg, 1)
    return np.where(deg > 0, 4.0 - deg - nbr_deg, 0.0)


def hks_node(A, t=1.0):
    n = A.shape[0]
    Af = A.astype(np.float64)
    d = Af.sum(1)
    dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    L = np.eye(n) - dinv[:, None] * Af * dinv[None, :]
    lam, phi = np.linalg.eigh(L)
    return (phi ** 2 * np.exp(-t * lam)[None, :]).sum(1)


def scan(A_sub, f, n_slices, m):
    grid = np.unique(np.quantile(f, np.linspace(0, 1, n_slices + m)))
    out = np.zeros((n_slices, 4), np.float32)
    for i in range(min(max(len(grid) - m, 0), n_slices)):
        idx = np.where((f >= grid[i]) & (f <= grid[i + m]))[0]
        if len(idx) == 0:
            continue
        S = A_sub[np.ix_(idx, idx)]
        b0, b1 = betti01(S)
        out[i] = (b0, b1, len(idx), int(S.sum() // 2))
    return out


# (N, N) correlation -> (N, n_slices, 8): Forman (b0, b1, |V|, |E|) | HKS (b0, b1, |V|, |E|)
def toposcan_subject(corr, node_frac, n_slices):
    A = knn_graph(corr, KNN)
    N = A.shape[0]
    cap = max(int(round(node_frac * N)), 3)
    out = np.zeros((N, n_slices, 8), np.float32)
    for u in range(N):
        nodes = two_hop(A, u, cap)
        S = A[np.ix_(nodes, nodes)]
        if S.sum() == 0:
            continue
        out[u, :, :4] = scan(S, forman_node(S), n_slices, WIDTH_M)
        out[u, :, 4:] = scan(S, hks_node(S, HKS_T), n_slices, WIDTH_M)
    return out


def toposcan_features(corr_np, node_frac, n_slices, cache_dir, workers=None):
    fp = hashlib.md5(np.ascontiguousarray(corr_np, np.float32).tobytes()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"toposcan_{fp}_f{node_frac}_s{n_slices}.npy")
    if os.path.exists(path):
        return np.load(path)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    batch = [corr_np[s].astype(np.float64) for s in range(len(corr_np))]
    fn = partial(toposcan_subject, node_frac=node_frac, n_slices=n_slices)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(fn, batch, chunksize=4))
    else:
        res = [fn(c) for c in batch]
    out = np.stack(res).astype(np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, out)
    return out


# log1p + z-score per (slice, feature), fitted on training subjects only
class ScanScaler:
    def fit(self, X, train_idx, rel_eps=1e-6):
        Z = np.log1p(np.clip(np.asarray(X, np.float64)[train_idx], 0, None))
        self.mu, self.sd = Z.mean((0, 1)), Z.std((0, 1))
        self.keep = self.sd > rel_eps * np.maximum(1.0, np.abs(self.mu))
        return self

    def transform(self, X):
        Z = np.log1p(np.clip(np.asarray(X, np.float64), 0, None))
        return (((Z - self.mu) / np.where(self.keep, self.sd, 1.0)) * self.keep).astype(np.float32)


# ======================================================================================
# Model
# ======================================================================================

class ROIScanEncoder(nn.Module):
    def __init__(self, n_slices, d=64, nhead=4, layers=2, ff=256, dropout=0.1, d_tok=64):
        super().__init__()
        self.embed = nn.Linear(4, d)
        self.slice_emb = nn.Parameter(torch.zeros(1, n_slices, d))
        self.filt_emb = nn.Parameter(torch.zeros(1, 2, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        for p in (self.slice_emb, self.filt_emb, self.cls):
            nn.init.normal_(p, std=0.02)
        lyr = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=ff, dropout=dropout,
                                         batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(lyr, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, d_tok)

    def forward(self, s):
        M, S, _ = s.shape
        both = torch.stack([s[..., :4], s[..., 4:]], 1)
        h = self.embed(both) + self.slice_emb[:, :S].unsqueeze(1) + self.filt_emb.unsqueeze(2)
        h = h.reshape(M, 2 * S, -1)
        h = torch.cat([self.cls.expand(M, -1, -1), h], 1)
        return self.out(self.tr(h)[:, 0])


# Backbone (TransPoolingEncoder layers, dim_reduction, fc) follows ALTER's BrainNetworkTransformer
# (https://github.com/yushuowiki/ALTER, alter/models/LRBGT/lrbgt.py), with the Topo-Scan
# positional encoding in place of ALTER's random-walk encoding.
class TopoScanPE(nn.Module):
    def __init__(self, node_sz, n_slices, d_pe=16, d_tok=64, dropout=0.1,
                 sizes=(200, 100), pooling=(False, True), hidden=1024, n_classes=2):
        super().__init__()
        self.K = d_pe
        nhead = 4
        width = node_sz + self.K
        while width % nhead:
            width += 1
        self.pad, self.width = width - (node_sz + self.K), width

        self.roi = ROIScanEncoder(n_slices, dropout=dropout, d_tok=d_tok)
        self.pe_proj = nn.Linear(d_tok, self.K)
        sizes = list(sizes)
        sizes[0] = node_sz
        self.attention_list = nn.ModuleList([
            TransPoolingEncoder(input_feature_size=width, input_node_num=n, hidden_size=hidden,
                                output_node_num=s, pooling=p, orthogonal=True,
                                freeze_center=True, project_assignment=True)
            for n, s, p in zip([node_sz, sizes[0]], sizes, pooling)])
        self.res_proj = nn.Linear(node_sz, width)
        self.dim_reduction = nn.Sequential(nn.Linear(width, 8), nn.LeakyReLU())
        self.fc = nn.Sequential(nn.Linear(8 * sizes[-1], 256), nn.LeakyReLU(),
                                nn.Linear(256, 32), nn.LeakyReLU(), nn.Linear(32, n_classes))

    def forward(self, corr, scan, mix=None):
        B, N = corr.shape[0], corr.shape[1]
        if mix is not None:
            lam, index = mix
            scan = lam * scan + (1 - lam) * scan[index]
        tok = self.roi(scan.reshape(B * N, scan.shape[2], scan.shape[3])).reshape(B, N, -1)
        pe = self.pe_proj(tok)
        t = corr.std(dim=(-1, -2), keepdim=True)
        pe = pe * (t / pe.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6))
        parts = [corr, pe]
        if self.pad:
            parts.append(corr.new_zeros(B, N, self.pad))
        z = torch.cat(parts, dim=-1)
        r = self.res_proj(corr)
        for layer in self.attention_list:
            z, _ = layer(z)
            if z.shape[1] == r.shape[1]:
                z = z + r
        return self.fc(self.dim_reduction(z).reshape(B, -1))


# ======================================================================================
# Training
# ======================================================================================

# Adapted from ALTER (https://github.com/yushuowiki/ALTER), alter/utils/prepossess.py:
# continus_mixup_data, also returning (lam, index) so the Topo-Scan input is mixed with the same pairs.
def mixup_with_index(*xs, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(y.shape[0]).to(y.device)
    new = [lam * x + (1 - lam) * x[index, :] for x in xs]
    return (*new, lam * y + (1 - lam) * y[index], (lam, index))


# Adam with ALTER's settings (alter/conf/optimizer/adam.yaml); embeddings excluded from weight decay
def make_optimizer(model, weight_decay=1e-4, no_wd=("cls", "slice_emb", "filt_emb")):
    sel = [p for n, p in model.named_parameters() if any(k in n for k in no_wd) and p.requires_grad]
    ids = {id(p) for p in sel}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.Adam([{"params": rest}, {"params": sel, "weight_decay": 0.0}],
                            lr=0.0, weight_decay=weight_decay)


# train_per_epoch / test_per_epoch adapted from ALTER, alter/training/training.py (Train)
class TSPETrain(Train):

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        for corr, scan, label in self.train_dataloader:
            corr, scan, label = corr.to(DEVICE), scan.to(DEVICE), label.to(DEVICE).float()
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            mix = None
            if self.config.preprocess.continus:
                corr, label, mix = mixup_with_index(corr, y=label)
            predict = self.model(corr, scan, mix=mix)
            loss = self.loss_fn(predict, label)
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.train_accuracy.update_with_weight(accuracy(predict, label[:, 1])[0], label.shape[0])

    @torch.no_grad()
    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        self.model.eval()
        labels, result = [], []
        for corr, scan, label in dataloader:
            corr, scan, label = corr.to(DEVICE), scan.to(DEVICE), label.to(DEVICE).float()
            output = self.model(corr, scan)
            loss_meter.update_with_weight(self.loss_fn(output, label).item(), label.shape[0])
            acc_meter.update_with_weight(accuracy(output, label[:, 1])[0], label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()
        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(labels, result, average='micro')
        report = classification_report(labels, result, output_dict=True, zero_division=0)
        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def save_result(self, results):
        self.history = results
        super().save_result(results)


# ======================================================================================
# Main
# ======================================================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(r, args, corr, labels, site, X):
    seed_all(r)
    split = get_split(labels, site, r, os.path.join(args.out, "splits"))
    T = torch.from_numpy(ScanScaler().fit(X, split[0]).transform(X))
    loaders = make_loaders(corr, T, labels, split, args.batch_size)
    steps_per_epoch = (len(split[0]) - 1) // args.batch_size + 1
    cfg = SimpleNamespace(
        training=SimpleNamespace(epochs=args.epochs),
        preprocess=SimpleNamespace(continus=True),
        total_steps=steps_per_epoch * args.epochs,
        log_path=os.path.join(args.out, "runs"),
        unique_id=f"split{r}_{datetime.now().strftime('%m%d-%H%M%S%f')}")
    model = TopoScanPE(corr.shape[1], int(T.shape[2]), dropout=args.dropout).to(DEVICE)
    seed_all(r)
    trainer = TSPETrain(cfg, model, [make_optimizer(model)],
                        [LRScheduler(1e-4, 1e-5, cfg.total_steps)], loaders, logger_factory(cfg))
    trainer.train()

    # BQN protocol (https://github.com/LYWJUN/BQN-demo, main.py): test metrics at the lowest-val-loss epoch
    hist = trainer.history
    e = int(np.argmin([h["Val Loss"] for h in hist]))
    best, last = hist[e], hist[-1]
    return {"split": r, "epoch_vloss": e, "smallest_vloss": best["Val Loss"],
            "AUC_at_vloss": 100 * best["Test AUC"], "ACC_at_vloss": best["Test Accuracy"],
            "SEN_at_vloss": 100 * best["Test Sensitivity"], "SPE_at_vloss": 100 * best["Test Specificity"],
            "AUC_final": 100 * last["Test AUC"], "ACC_final": last["Test Accuracy"],
            "SEN_final": 100 * last["Test Sensitivity"], "SPE_final": 100 * last["Test Specificity"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="abide.npy")
    ap.add_argument("--out", default="results")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--node_frac", type=float, default=0.4)
    ap.add_argument("--slices", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    corr, labels, site = load_abide(args.data)
    X = toposcan_features(corr.numpy().astype(np.float32), args.node_frac, args.slices,
                          os.path.join(args.out, "cache"), args.workers)

    csv = os.path.join(args.out, f"tspe_f{args.node_frac:g}_s{args.slices}_d{args.dropout:g}.csv")
    rows = []
    for r in range(args.splits):
        rows.append(run(r, args, corr, labels, site, X))
        pd.DataFrame(rows).to_csv(csv, index=False, float_format="%.4f")

    # mean ± std over splits, np.std (ddof=0) as in BQN
    df = pd.DataFrame(rows)
    for sfx in ("at_vloss", "final"):
        print(sfx, "  ".join(f"{m} {df[f'{m}_{sfx}'].mean():.2f}±{df[f'{m}_{sfx}'].std(ddof=0):.2f}"
                             for m in ("AUC", "ACC", "SEN", "SPE")))
    print("results ->", csv)


if __name__ == "__main__":
    main()
