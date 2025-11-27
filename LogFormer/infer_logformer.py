#!/usr/bin/env python3
# infer_logformer.py

import argparse, os, csv, numpy as np, torch
from tqdm import tqdm
from dataloader import DataGenerator
from model      import Model

# Use script-relative path for defaults.
THIS_DIR = os.path.dirname(__file__)
DEFAULT_CKPT = os.path.join(THIS_DIR, "checkpoints", "best_alllogs_classifier_2L_do0.2_1e-05.pt")
DEFAULT_OUT  = os.path.join(os.path.dirname(THIS_DIR), "output", "anomaly_logs_detected_by_logformer.csv")

P = argparse.ArgumentParser()
P.add_argument("--ckpt", default=DEFAULT_CKPT)
P.add_argument("--window_size", type=int, default=20)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--out",  default=DEFAULT_OUT)
args = P.parse_args()

# Merge all windows (train+test) using script-relative path.
NPZ = os.path.join(os.path.dirname(__file__), "npz")
tr = np.load(f"{NPZ}/train.npz", allow_pickle=True)
te = np.load(f"{NPZ}/test.npz",  allow_pickle=True)

def to_f32(a): return np.stack(a.tolist()).astype(np.float32) if a.dtype==object else a.astype(np.float32)
x_all = np.concatenate([to_f32(tr["x"]), to_f32(te["x"])], axis=0)
dummy_y = np.zeros((len(x_all), 2), dtype=np.float32)          # placeholder for DataGenerator

loader = torch.utils.data.DataLoader(
    DataGenerator(x_all, dummy_y, args.window_size),
    batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

# ─── load model ───
state = torch.load(args.ckpt, map_location="cpu")  # state_dict only
model = Model(mode="classifier", num_layers=2, adapter_size=64,
              dim=768, window_size=args.window_size, nhead=8,
              dim_feedforward=4*768, dropout=0.20)
model.load_state_dict(state); model.eval()

# ─── inference ───
preds, probs = [], []
with torch.no_grad():
    for xb,_ in tqdm(loader, desc="Infer"):
        out = model(xb)                      # (B,2)
        ps  = torch.sigmoid(out[:,1])
        preds.extend((ps>=0.7).int().tolist())
        probs.extend(ps.tolist())

# ─── save ───
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out,"w",newline='',encoding="utf-8") as fw:
    wr = csv.writer(fw); wr.writerow(["window_id","is_anomaly_pred","prob"])
    for i,(p,s) in enumerate(zip(preds,probs)):
        wr.writerow([i,p,s])

print(f"[✓] Saved predictions → {args.out}")
