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
P.add_argument("--threshold", type=float, default=0.67)
# keep model shape consistent with training
P.add_argument("--mode", default="classifier", choices=["classifier", "adapter"])
P.add_argument("--num_layers", type=int, default=2)
P.add_argument("--adapter_size", type=int, default=64)
P.add_argument("--dropout", type=float, default=0.20)
P.add_argument("--npz_dir", default=os.path.join(os.path.dirname(__file__), "npz"))
P.add_argument("--splits", default="train,val,test",
               help="Comma-separated splits to infer on (e.g., train,val,test or test)")
args = P.parse_args()

# --- Concatenate windows from selected NPZ splits ---
NPZ = args.npz_dir
tr = np.load(f"{NPZ}/train.npz", allow_pickle=True)
va_path = f"{NPZ}/val.npz"
te = np.load(f"{NPZ}/test.npz", allow_pickle=True)

def to_f32(a):
    return np.stack(a.tolist()).astype(np.float32) if a.dtype == object else a.astype(np.float32)

split_map = {
    "train": to_f32(tr["x"]),
    "val": to_f32(np.load(va_path, allow_pickle=True)["x"]) if os.path.exists(va_path) else None,
    "test": to_f32(te["x"]),
}
requested = [s.strip() for s in args.splits.split(",") if s.strip()]
xs = []
for s in requested:
    if s not in split_map:
        raise ValueError(f"Unknown split: {s}")
    if split_map[s] is None:
        raise FileNotFoundError(f"Split '{s}' requested but {va_path} not found")
    xs.append(split_map[s])
if not xs:
    raise ValueError("No valid splits provided")
x_all = np.concatenate(xs, axis=0)
dummy_y = np.zeros((len(x_all), 2), dtype=np.float32)  # placeholder for DataGenerator

loader = torch.utils.data.DataLoader(
    DataGenerator(x_all, dummy_y, args.window_size, return_mask=True),
    batch_size=args.batch, shuffle=False, num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# --- Load model checkpoint ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(args.ckpt, map_location=device)  # state_dict-only file
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
model.load_state_dict(state)
model.eval()

# --- Inference ---
preds, probs = [], []
with torch.no_grad():
    for xb, _, pad_mask in tqdm(loader, desc="Infer"):
        xb = xb.to(device)
        pad_mask = pad_mask.to(device).bool()
        out = model(xb, src_key_padding_mask=pad_mask)  # (B, 2)
        ps = torch.softmax(out, dim=1)[:, 1]  # anomaly probability
        preds.extend((ps >= args.threshold).int().cpu().tolist())
        probs.extend(ps.cpu().tolist())

# --- Save predictions ---
out_dir = os.path.dirname(args.out)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
with open(args.out, "w", newline="", encoding="utf-8") as fw:
    wr = csv.writer(fw)
    wr.writerow(["window_id", "is_anomaly_pred", "prob"])
    for i, (p, s) in enumerate(zip(preds, probs)):
        wr.writerow([i, p, s])

print(f"[✓] Saved predictions → {args.out}")
