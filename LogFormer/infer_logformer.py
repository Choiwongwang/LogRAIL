#!/usr/bin/env python3
# infer_logformer.py

import argparse, os, csv, numpy as np, torch
from tqdm import tqdm
from dataloader import DataGenerator
from model import Model

P = argparse.ArgumentParser()
P.add_argument("--ckpt", required=True)
P.add_argument("--window_size", type=int, default=20)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--out", required=True)
args = P.parse_args()

# --- Concatenate all windows from NPZ (train + test) ---
NPZ = "/home/irv4/PycharmProjects/LG_Project/LogRAG/LogFormer/npz"
tr = np.load(f"{NPZ}/train.npz", allow_pickle=True)
te = np.load(f"{NPZ}/test.npz", allow_pickle=True)

def to_f32(a):
    return np.stack(a.tolist()).astype(np.float32) if a.dtype == object else a.astype(np.float32)

x_all = np.concatenate([to_f32(tr["x"]), to_f32(te["x"])], axis=0)
dummy_y = np.zeros((len(x_all), 2), dtype=np.float32)  # placeholder for DataGenerator

loader = torch.utils.data.DataLoader(
    DataGenerator(x_all, dummy_y, args.window_size, return_mask=True),
    batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True
)

# --- Load model checkpoint ---
state = torch.load(args.ckpt, map_location="cpu")  # state_dict-only file
model = Model(
    mode="classifier",
    num_layers=2,
    adapter_size=64,
    dim=768,
    window_size=args.window_size,
    nhead=8,
    dim_feedforward=4 * 768,
    dropout=0.20
)
model.load_state_dict(state)
model.eval()

# --- Inference ---
preds, probs = [], []
with torch.no_grad():
    for xb, _, pad_mask in tqdm(loader, desc="Infer"):
        out = model(xb, src_key_padding_mask=pad_mask)  # (B, 2)
        ps = torch.sigmoid(out[:, 1])  # anomaly probability
        preds.extend((ps >= 0.7).int().tolist())
        probs.extend(ps.tolist())

# --- Save predictions ---
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="", encoding="utf-8") as fw:
    wr = csv.writer(fw)
    wr.writerow(["window_id", "is_anomaly_pred", "prob"])
    for i, (p, s) in enumerate(zip(preds, probs)):
        wr.writerow([i, p, s])

print(f"[✓] Saved predictions → {args.out}")
