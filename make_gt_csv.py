#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate GT CSVs from LogFormer NPZ outputs.

- gt_test.csv: labels for test windows only
  window_id starts at n_train (because infer_logformer concatenates train+test)
- gt_all.csv: labels for all windows (train+test), window_id 0..N-1
"""

import argparse
import os
import numpy as np
import pandas as pd


def _to_arr(a):
    return np.stack(a.tolist()) if getattr(a, "dtype", None) == object else a


def _to_label(y):
    y = _to_arr(y)
    if y.ndim == 2 and y.shape[1] >= 2:
        return y[:, 1].astype(int)
    return y.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="LogFormer/npz", help="Folder with train.npz/test.npz")
    ap.add_argument("--out_test", default="output/gt_test.csv")
    ap.add_argument("--out_all", default="output/gt_all.csv")
    ap.add_argument("--no_all", action="store_true", help="Do not write gt_all.csv")
    args = ap.parse_args()

    tr_path = os.path.join(args.npz_dir, "train.npz")
    te_path = os.path.join(args.npz_dir, "test.npz")
    if not os.path.exists(tr_path) or not os.path.exists(te_path):
        raise SystemExit(f"[x] npz not found: {tr_path} / {te_path}")

    tr = np.load(tr_path, allow_pickle=True)
    te = np.load(te_path, allow_pickle=True)

    y_train = _to_label(tr["y"])
    y_test = _to_label(te["y"])
    n_train = len(y_train)
    n_test = len(y_test)

    # gt_test.csv
    df_test = pd.DataFrame({
        "window_id": np.arange(n_train, n_train + n_test, dtype=int),
        "label": y_test,
    })
    os.makedirs(os.path.dirname(args.out_test), exist_ok=True)
    df_test.to_csv(args.out_test, index=False)
    print(f"[✓] wrote {args.out_test} rows={len(df_test):,}")

    # gt_all.csv (train+test)
    if not args.no_all and args.out_all:
        y_all = np.concatenate([y_train, y_test], axis=0)
        df_all = pd.DataFrame({
            "window_id": np.arange(len(y_all), dtype=int),
            "label": y_all,
        })
        os.makedirs(os.path.dirname(args.out_all), exist_ok=True)
        df_all.to_csv(args.out_all, index=False)
        print(f"[✓] wrote {args.out_all} rows={len(df_all):,}")


if __name__ == "__main__":
    main()
