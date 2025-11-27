#!/usr/bin/env python3
# train_transformer_fmt.py  (console confusion-matrix output added)

import argparse, os, random, time, warnings, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix  # added
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

from dataloader import DataGenerator
from model      import Model

# ────────── CLI ──────────
P = argparse.ArgumentParser()
P.add_argument("--log_name", default="alllogs")
P.add_argument("--window_size", type=int, default=20)
P.add_argument("--mode", default="classifier", choices=["classifier","adapter"])
P.add_argument("--num_layers", type=int, default=2)
P.add_argument("--adapter_size", type=int, default=64)
P.add_argument("--dropout", type=float, default=0.20)
P.add_argument("--lr", type=float, default=1e-5)
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--patience", type=int, default=8)
P.add_argument("--batch", type=int, default=64)
args = P.parse_args()

suffix = f"{args.log_name}_{args.mode}_{args.num_layers}L_do{args.dropout}_{args.lr}"
os.makedirs("result", exist_ok=True); os.makedirs("checkpoints", exist_ok=True)

# ────────── Device & Seed ──────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
torch.manual_seed(123); np.random.seed(123); random.seed(123)

# ────────── Data load ──────────
NPZ_DIR = "/home/irv4/PycharmProjects/LG_Project/LogRAG/LogFormer/npz"
tr = np.load(f"{NPZ_DIR}/train.npz", allow_pickle=True)
te = np.load(f"{NPZ_DIR}/test.npz",  allow_pickle=True)
x_tr = np.stack(tr["x"].tolist()).astype(np.float32) if tr["x"].dtype==object else tr["x"].astype(np.float32)
y_tr = np.stack(tr["y"].tolist()).astype(np.float32) if tr["y"].dtype==object else tr["y"].astype(np.float32)
x_te = np.stack(te["x"].tolist()).astype(np.float32) if te["x"].dtype==object else te["x"].astype(np.float32)
y_te = np.stack(te["y"].tolist()).astype(np.float32) if te["y"].dtype==object else te["y"].astype(np.float32)

train_loader = torch.utils.data.DataLoader(
    DataGenerator(x_tr,y_tr,args.window_size),
    batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    DataGenerator(x_te,y_te,args.window_size),
    batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

# ────────── Model ──────────
model = Model(mode=args.mode, num_layers=args.num_layers, adapter_size=args.adapter_size,
              dim=768, window_size=args.window_size, nhead=8, dim_feedforward=4*768,
              dropout=args.dropout).to(device)
opt       = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
criterion  = nn.BCEWithLogitsLoss()

best_f1, patience_cnt = 0.0, 0
train_loss_hist, test_f1_hist = [], []

def evaluate(loader):
    """Return Precision/Recall/F1 and confusion matrix ([[TN FP],[FN TP]])."""
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))           # (B,2) logits
            y_pred.extend(out.argmax(1).cpu().tolist())
            y_true.extend(yb.argmax(1).tolist())
    P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN, FP],[FN, TP]]
    return P, R, F, len(y_true), cm

# ────────── Training loop ──────────
for epoch in range(args.epochs):
    model.train(); losses=[]
    t0 = time.time()

    for xb,yb in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=90):
        xb,yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
        losses.append(loss.item())
    avg_loss = np.mean(losses); train_loss_hist.append(avg_loss)
    elapsed  = time.time() - t0

    # --- Evaluate on test set ---
    P,R,F,n_win,CM = evaluate(test_loader); test_f1_hist.append(F)
    TN, FP, FN, TP = int(CM[0,0]), int(CM[0,1]), int(CM[1,0]), int(CM[1,1])

    print(f"Epoch: {epoch}/{args.epochs-1}  |  Avg Train Loss = {avg_loss:.6f}  |  Time = {elapsed:4.1f}s")
    print(f"Number of testing windows: {n_win}")
    print(f"  Precision: {P:.4f}")
    print(f"  Recall   : {R:.4f}")
    print(f"  F1 score : {F:.4f}")
    print("  Confusion-matrix  [ [TN FP] [FN TP] ]")
    print(f"  [[ {TN} {FP}]\n   [ {FN} {TP}]]")

    # --- Early-Stop condition ---
    if F > best_f1 + 1e-4:
        best_f1, patience_cnt = F, 0
        torch.save(model.state_dict(), f"checkpoints/best_{suffix}.pt")
    else:
        patience_cnt += 1
        if patience_cnt >= args.patience:
            print(f"> Early stopping (patience {args.patience}) at epoch {epoch}")
            break
    scheduler.step()

print("Training finished.")
