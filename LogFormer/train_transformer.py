#!/usr/bin/env python3
# train_transformer.py — Stage-1 classifier with CLI seed; early-stop without fixed θ.

import argparse
import os
import random
import time
import warnings

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

matplotlib.use("Agg")

from dataloader import DataGenerator
from model import Model

P = argparse.ArgumentParser()
P.add_argument("--log_name", default="alllogs")
P.add_argument("--window_size", type=int, default=20)
P.add_argument("--mode", default="classifier", choices=["classifier", "adapter"])
P.add_argument("--num_layers", type=int, default=2)
P.add_argument("--adapter_size", type=int, default=64)
P.add_argument("--dropout", type=float, default=0.20)
P.add_argument("--lr", type=float, default=1e-5)
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--patience", type=int, default=8)
P.add_argument("--batch", type=int, default=64)
P.add_argument(
    "--threshold",
    type=float,
    default=None,
    help="Deprecated for early-stop; if set, only used for printed val metrics",
)
P.add_argument("--seed", type=int, default=123)
P.add_argument(
    "--npz_dir",
    default=None,
    help="default: <this_dir>/npz",
)
P.add_argument("--ckpt_dir", default=None, help="default: <this_dir>/checkpoints")
args = P.parse_args()

_HERE = os.path.dirname(os.path.abspath(__file__))
NPZ_DIR = args.npz_dir or os.path.join(_HERE, "npz")
CKPT_DIR = args.ckpt_dir or os.path.join(_HERE, "checkpoints")
os.makedirs(os.path.join(_HERE, "result"), exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

suffix = (
    f"{args.log_name}_{args.mode}_{args.num_layers}L_"
    f"do{args.dropout}_{args.lr}_seed{args.seed}"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

tr = np.load(f"{NPZ_DIR}/train.npz", allow_pickle=True)
va = np.load(f"{NPZ_DIR}/val.npz", allow_pickle=True)
te = np.load(f"{NPZ_DIR}/test.npz", allow_pickle=True)


def _stack(a):
    return np.stack(a.tolist()).astype(np.float32) if a.dtype == object else a.astype(np.float32)


x_tr, y_tr = _stack(tr["x"]), _stack(tr["y"])
x_va, y_va = _stack(va["x"]), _stack(va["y"])
x_te, y_te = _stack(te["x"]), _stack(te["y"])

train_loader = torch.utils.data.DataLoader(
    DataGenerator(x_tr, y_tr, args.window_size, return_mask=True),
    batch_size=args.batch,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    DataGenerator(x_va, y_va, args.window_size, return_mask=True),
    batch_size=args.batch,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    DataGenerator(x_te, y_te, args.window_size, return_mask=True),
    batch_size=args.batch,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

model = Model(
    mode=args.mode,
    num_layers=args.num_layers,
    adapter_size=args.adapter_size,
    dim=768,
    window_size=args.window_size,
    nhead=8,
    dim_feedforward=4 * 768,
    dropout=args.dropout,
).to(device)

opt = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
criterion = nn.CrossEntropyLoss()

best_f1, patience_cnt = 0.0, 0
print(f"[i] seed={args.seed} npz_dir={NPZ_DIR} early_stop=argmax_F1 (not fixed θ)")


def evaluate(loader, use_threshold=None):
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, pad_mask in loader:
            xb = xb.to(device)
            pad_mask = pad_mask.to(device).bool()
            out = model(xb, src_key_padding_mask=pad_mask)
            if use_threshold is None:
                pred = out.argmax(1).cpu().tolist()
            else:
                ps = torch.softmax(out, dim=1)[:, 1]
                pred = (ps >= use_threshold).int().cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.argmax(1).tolist())
    P, R, F, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return P, R, F, len(y_true), cm


for epoch in range(args.epochs):
    model.train()
    losses = []
    t0 = time.time()
    for xb, yb, pad_mask in tqdm(
        train_loader, desc=f"Epoch {epoch}", leave=False, ncols=90
    ):
        xb, yb = xb.to(device), yb.to(device)
        pad_mask = pad_mask.to(device).bool()
        y_idx = yb.argmax(1).long()
        loss = criterion(model(xb, src_key_padding_mask=pad_mask), y_idx)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        losses.append(loss.item())

    avg_loss = float(np.mean(losses))
    elapsed = time.time() - t0
    P, R, F, n_win, CM = evaluate(val_loader, use_threshold=None)
    TN, FP, FN, TP = int(CM[0, 0]), int(CM[0, 1]), int(CM[1, 0]), int(CM[1, 1])
    print(
        f"Epoch: {epoch}/{args.epochs - 1}  |  Avg Train Loss = {avg_loss:.6f}  |  Time = {elapsed:4.1f}s"
    )
    print(f"Number of validation windows: {n_win}")
    print(f"  Precision (val,argmax): {P:.4f}")
    print(f"  Recall   (val,argmax): {R:.4f}")
    print(f"  F1 score (val,argmax): {F:.4f}")
    print(f"  [[ {TN} {FP}]\n   [ {FN} {TP}]]")

    if F > best_f1 + 1e-4:
        best_f1, patience_cnt = F, 0
        ckpt_path = os.path.join(CKPT_DIR, f"best_{suffix}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[✓] saved {ckpt_path}")
    else:
        patience_cnt += 1
        if patience_cnt >= args.patience:
            print(f"> Early stopping (patience {args.patience}) at epoch {epoch}")
            break
    scheduler.step()

ckpt_path = os.path.join(CKPT_DIR, f"best_{suffix}.pt")
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

# Optional: print test with argmax (do NOT use for tuning)
P, R, F, n_win, CM = evaluate(test_loader, use_threshold=None)
TN, FP, FN, TP = int(CM[0, 0]), int(CM[0, 1]), int(CM[1, 0]), int(CM[1, 1])
print("\n[note] Final argmax metrics on test set (DO NOT use for tuning)")
print(f"Number of testing windows: {n_win}")
print(f"  Precision: {P:.4f}  Recall: {R:.4f}  F1: {F:.4f}")
print(f"  [[ {TN} {FP}]\n   [ {FN} {TP}]]")
print(f"Training finished. best_val_argmax_f1={best_f1:.4f} ckpt={ckpt_path}")
