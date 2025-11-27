#!/usr/bin/env python3
# train_transformer_fmt_with_val_curve.py

import argparse, os, random, time, warnings, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
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
val_loader = torch.utils.data.DataLoader(
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
best_epoch, early_stop_epoch = 0, None
train_loss_hist, val_loss_hist, val_f1_hist = [], [], []

def evaluate_metrics_and_loss(loader):
    """Compute average loss + P/R/F1 for validation."""
    preds, trues, losses = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            preds.extend(logits.argmax(1).cpu().numpy())   # ✅ GPU → CPU → NumPy
            trues.extend(yb.argmax(1).cpu().numpy())       # ✅ GPU → CPU → NumPy
    avg_loss = float(np.mean(losses)) if losses else 0.0
    P, R, F = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)[:3]
    return avg_loss, P, R, F, len(trues)

# ────────── Training loop ──────────
for epoch in range(args.epochs):
    model.train(); batch_losses=[]
    t0 = time.time()

    for xb,yb in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=90):
        xb,yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
        batch_losses.append(loss.item())

    avg_train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    train_loss_hist.append(avg_train_loss)

    # --- Validation (loss + F1) ---
    avg_val_loss, P, R, F, n_win = evaluate_metrics_and_loss(val_loader)
    val_loss_hist.append(avg_val_loss)
    val_f1_hist.append(F)

    elapsed = time.time() - t0
    print(f"Epoch {epoch:02d}/{args.epochs-1}  |  TrainLoss {avg_train_loss:.4f}  |  ValLoss {avg_val_loss:.4f}  |  ValF1 {F:.4f}  |  {elapsed:4.1f}s")

    # --- Early-Stop (based on F1) ---
    if F > best_f1 + 1e-4:
        best_f1, patience_cnt = F, 0
        best_epoch = epoch
        torch.save(model.state_dict(), f"checkpoints/best_{suffix}.pt")
    else:
        patience_cnt += 1
        if patience_cnt >= args.patience:
            early_stop_epoch = epoch  # stop here
            print(f"> Early stopping (patience={args.patience}) at epoch {epoch}")
            break

    scheduler.step()

print("Training finished.")
print(f"Best Val F1 = {best_f1:.4f} @ epoch {best_epoch}")

# ────────── Plots: Train vs Validation Loss (+ Early-Stop marker) ──────────
# 1️⃣ Train Loss vs Test F1
import matplotlib.pyplot as plt

# 1️⃣ Train Loss vs Validation F1
plt.figure(figsize=(5,3))
plt.plot(train_loss_hist, label="Train Loss", color="#1f77b4")  # blue
plt.plot(val_f1_hist, label="Val F1", color="#ff7f0e")         # orange
plt.xlabel("Epoch"); plt.ylabel("Loss / F1")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("result/train_vs_valF1.png", dpi=300)

# 2️⃣ Train / Validation Loss + Early Stop
plt.figure(figsize=(5,3))
plt.plot(train_loss_hist, label="Train Loss", color="#2ca02c")  # green
plt.plot(val_loss_hist, label="Val Loss", color="#d62728")      # red
plt.axvline(early_stop_epoch, color="#9467bd", linestyle="--", 
            label=f"Early stop @ {early_stop_epoch}")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("result/train_val_loss_with_earlystop.png", dpi=300)

# 3️⃣ Validation F1
plt.figure(figsize=(5,3))
plt.plot(val_f1_hist, color="#17becf", label="Validation F1")  # teal
plt.xlabel("Epoch"); plt.ylabel("F1")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("result/valF1_curve.png", dpi=300)
