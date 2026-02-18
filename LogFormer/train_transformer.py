#!/usr/bin/env python3
# train_transformer_fmt.py (prints confusion matrix to console)

import argparse, os, random, time, warnings, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401

from dataloader import DataGenerator
from model import Model

# ---------------- CLI ----------------
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
P.add_argument("--threshold", type=float, default=0.67)
args = P.parse_args()

suffix = f"{args.log_name}_{args.mode}_{args.num_layers}L_do{args.dropout}_{args.lr}"
os.makedirs("result", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ---------------- Device & Seed ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

# ---------------- Load data ----------------
# Use local npz folder relative to this script for portability
NPZ_DIR = os.path.join(os.path.dirname(__file__), "npz")
tr = np.load(f"{NPZ_DIR}/train.npz", allow_pickle=True)
va = np.load(f"{NPZ_DIR}/val.npz", allow_pickle=True)
te = np.load(f"{NPZ_DIR}/test.npz", allow_pickle=True)

x_tr = np.stack(tr["x"].tolist()).astype(np.float32) if tr["x"].dtype == object else tr["x"].astype(np.float32)
y_tr = np.stack(tr["y"].tolist()).astype(np.float32) if tr["y"].dtype == object else tr["y"].astype(np.float32)
x_va = np.stack(va["x"].tolist()).astype(np.float32) if va["x"].dtype == object else va["x"].astype(np.float32)
y_va = np.stack(va["y"].tolist()).astype(np.float32) if va["y"].dtype == object else va["y"].astype(np.float32)
x_te = np.stack(te["x"].tolist()).astype(np.float32) if te["x"].dtype == object else te["x"].astype(np.float32)
y_te = np.stack(te["y"].tolist()).astype(np.float32) if te["y"].dtype == object else te["y"].astype(np.float32)

train_loader = torch.utils.data.DataLoader(
    DataGenerator(x_tr, y_tr, args.window_size, return_mask=True),
    batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    DataGenerator(x_te, y_te, args.window_size, return_mask=True),
    batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    DataGenerator(x_va, y_va, args.window_size, return_mask=True),
    batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True
)

# ---------------- Model ----------------
model = Model(
    mode=args.mode,
    num_layers=args.num_layers,
    adapter_size=args.adapter_size,
    dim=768,
    window_size=args.window_size,
    nhead=8,
    dim_feedforward=4 * 768,
    dropout=args.dropout
).to(device)

opt = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
criterion = nn.CrossEntropyLoss()

best_f1, patience_cnt = 0.0, 0
train_loss_hist, val_f1_hist = [], []
THRESH = args.threshold

def evaluate(loader):
    """Return Precision/Recall/F1 and confusion matrix ([[TN, FP], [FN, TP]])."""
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, pad_mask in loader:
            xb = xb.to(device)
            pad_mask = pad_mask.to(device).bool()
            out = model(xb, src_key_padding_mask=pad_mask)  # (B, 2) logits
            ps = torch.softmax(out, dim=1)[:, 1]  # anomaly probability
            y_pred.extend((ps >= THRESH).int().cpu().tolist())
            y_true.extend(yb.argmax(1).tolist())

    P, R, F, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN, FP], [FN, TP]]
    return P, R, F, len(y_true), cm

# ---------------- Training loop ----------------
for epoch in range(args.epochs):
    model.train()
    losses = []
    t0 = time.time()

    for xb, yb, pad_mask in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=90):
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
    train_loss_hist.append(avg_loss)
    elapsed = time.time() - t0

    # --- Evaluate on validation set ---
    P, R, F, n_win, CM = evaluate(val_loader)
    val_f1_hist.append(F)
    TN, FP, FN, TP = int(CM[0, 0]), int(CM[0, 1]), int(CM[1, 0]), int(CM[1, 1])

    print(f"Epoch: {epoch}/{args.epochs-1}  |  Avg Train Loss = {avg_loss:.6f}  |  Time = {elapsed:4.1f}s")
    print(f"Number of validation windows: {n_win}")
    print(f"  Precision (val): {P:.4f}")
    print(f"  Recall   (val): {R:.4f}")
    print(f"  F1 score (val): {F:.4f}")
    print("  Confusion matrix  [ [TN FP] [FN TP] ]")
    print(f"  [[ {TN} {FP}]\n   [ {FN} {TP}]]")

    # --- Early stopping (validation) ---
    if F > best_f1 + 1e-4:
        best_f1, patience_cnt = F, 0
        torch.save(model.state_dict(), f"checkpoints/best_{suffix}.pt")
    else:
        patience_cnt += 1
        if patience_cnt >= args.patience:
            print(f"> Early stopping (patience {args.patience}) at epoch {epoch}")
            break

    scheduler.step()

# ---------------- Final test evaluation ----------------
ckpt_path = f"checkpoints/best_{suffix}.pt"
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

P, R, F, n_win, CM = evaluate(test_loader)
TN, FP, FN, TP = int(CM[0, 0]), int(CM[0, 1]), int(CM[1, 0]), int(CM[1, 1])
print("\nFinal evaluation on test set")
print(f"Number of testing windows: {n_win}")
print(f"  Precision: {P:.4f}")
print(f"  Recall   : {R:.4f}")
print(f"  F1 score : {F:.4f}")
print("  Confusion matrix  [ [TN FP] [FN TP] ]")
print(f"  [[ {TN} {FP}]\n   [ {FN} {TP}]]")

print("Training finished.")
