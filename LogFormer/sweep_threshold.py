#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sweep decision thresholds for Stage-1 (LogFormer) and report metrics.

import argparse
import os
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from dataloader import DataGenerator
from model import Model


def to_f32(a: np.ndarray) -> np.ndarray:
    return np.stack(a.tolist()).astype(np.float32) if a.dtype == object else a.astype(np.float32)


def load_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(npz_path, allow_pickle=True)
    x = to_f32(arr["x"])
    y = to_f32(arr["y"])
    return x, y


def eval_at_threshold(probs: np.ndarray, y_true: np.ndarray, thr: float) -> tuple[float, float, float, np.ndarray]:
    y_pred = (probs >= thr).astype(int)
    P, R, F, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return P, R, F, cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to LogFormer checkpoint (state_dict)")
    ap.add_argument("--npz_dir", default=os.path.join(os.path.dirname(__file__), "npz"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mode", default="classifier", choices=["classifier", "adapter"])
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--adapter_size", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--thr_start", type=float, default=0.50)
    ap.add_argument("--thr_end", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.05)
    ap.add_argument("--out_csv", default=None, help="Optional CSV path to save sweep results")
    args = ap.parse_args()

    npz_path = os.path.join(args.npz_dir, f"{args.split}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    x, y = load_npz(npz_path)
    y_true = y.argmax(1).astype(int)

    loader = torch.utils.data.DataLoader(
        DataGenerator(x, y, args.window_size, return_mask=True),
        batch_size=args.batch, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device)
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

    probs = []
    with torch.no_grad():
        for xb, _, pad_mask in loader:
            xb = xb.to(device)
            pad_mask = pad_mask.to(device).bool()
            out = model(xb, src_key_padding_mask=pad_mask)
            ps = torch.softmax(out, dim=1)[:, 1]
            probs.extend(ps.cpu().numpy().tolist())
    probs = np.array(probs, dtype=np.float32)

    thresholds = np.arange(args.thr_start, args.thr_end + 1e-9, args.thr_step)
    best = {"thr": None, "f1": -1.0}
    rows = []
    for thr in thresholds:
        P, R, F, cm = eval_at_threshold(probs, y_true, float(thr))
        rows.append((thr, P, R, F, cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]))
        if F > best["f1"]:
            best = {"thr": float(thr), "f1": float(F)}

    print(f"[✓] Sweep split: {args.split}  (n={len(y_true)})")
    print(f"[✓] Best F1={best['f1']:.4f} at threshold={best['thr']:.2f}")

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall,f1,TN,FP,FN,TP\n")
            for r in rows:
                f.write(f"{r[0]:.2f},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]},{r[5]},{r[6]},{r[7]}\n")
        print(f"[✓] Saved sweep results → {args.out_csv}")


if __name__ == "__main__":
    main()
