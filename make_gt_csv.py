#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate GT CSVs from NPZ and/or split_manifest.csv.

Preferred: --manifest (window_id already assigned as train||val||test).
Fallback: NPZ sizes with train||val||test concatenation.
"""

from __future__ import annotations

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
    ap.add_argument("--npz_dir", default="LogFormer/npz")
    ap.add_argument(
        "--manifest",
        default=None,
        help="split_manifest.csv (preferred; uses window_id + label)",
    )
    ap.add_argument("--out_test", default="output/gt_test.csv")
    ap.add_argument("--out_val", default="output/gt_val.csv")
    ap.add_argument("--out_all", default="output/gt_all.csv")
    ap.add_argument("--no_all", action="store_true")
    ap.add_argument("--no_val", action="store_true")
    args = ap.parse_args()

    man_path = args.manifest
    if man_path is None:
        cand = os.path.join(args.npz_dir, "split_manifest.csv")
        if os.path.exists(cand):
            man_path = cand

    if man_path and os.path.exists(man_path):
        man = pd.read_csv(man_path)
        man = man[man["split"] != "gap_drop"].copy()
        for split_name, out_path, skip in (
            ("test", args.out_test, False),
            ("val", args.out_val, args.no_val),
        ):
            if skip:
                continue
            sub = man[man["split"] == split_name][["window_id", "label"]].copy()
            sub["window_id"] = sub["window_id"].astype(int)
            sub["label"] = sub["label"].astype(int)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            sub.to_csv(out_path, index=False)
            print(f"[✓] wrote {out_path} rows={len(sub):,} (from manifest)")

        if not args.no_all and args.out_all:
            all_df = man[man["split"].isin(["train", "val", "test"])][
                ["window_id", "label"]
            ].copy()
            all_df = all_df.sort_values("window_id")
            os.makedirs(os.path.dirname(args.out_all) or ".", exist_ok=True)
            all_df.to_csv(args.out_all, index=False)
            print(f"[✓] wrote {args.out_all} rows={len(all_df):,} (from manifest)")
        return

    # Fallback: NPZ train||val||test
    tr_path = os.path.join(args.npz_dir, "train.npz")
    va_path = os.path.join(args.npz_dir, "val.npz")
    te_path = os.path.join(args.npz_dir, "test.npz")
    if not os.path.exists(tr_path) or not os.path.exists(te_path):
        raise SystemExit(f"[x] npz not found: {tr_path} / {te_path}")

    y_train = _to_label(np.load(tr_path, allow_pickle=True)["y"])
    y_test = _to_label(np.load(te_path, allow_pickle=True)["y"])
    n_train = len(y_train)
    n_val = 0
    y_val = np.array([], dtype=int)
    if os.path.exists(va_path):
        y_val = _to_label(np.load(va_path, allow_pickle=True)["y"])
        n_val = len(y_val)

    os.makedirs(os.path.dirname(args.out_test) or ".", exist_ok=True)
    df_test = pd.DataFrame(
        {
            "window_id": np.arange(n_train + n_val, n_train + n_val + len(y_test), dtype=int),
            "label": y_test,
        }
    )
    df_test.to_csv(args.out_test, index=False)
    print(f"[✓] wrote {args.out_test} rows={len(df_test):,}")

    if not args.no_val and n_val > 0:
        os.makedirs(os.path.dirname(args.out_val) or ".", exist_ok=True)
        df_val = pd.DataFrame(
            {
                "window_id": np.arange(n_train, n_train + n_val, dtype=int),
                "label": y_val,
            }
        )
        df_val.to_csv(args.out_val, index=False)
        print(f"[✓] wrote {args.out_val} rows={len(df_val):,}")

    if not args.no_all and args.out_all:
        parts = [y_train]
        if n_val:
            parts.append(y_val)
        parts.append(y_test)
        y_all = np.concatenate(parts, axis=0)
        df_all = pd.DataFrame(
            {"window_id": np.arange(len(y_all), dtype=int), "label": y_all}
        )
        os.makedirs(os.path.dirname(args.out_all) or ".", exist_ok=True)
        df_all.to_csv(args.out_all, index=False)
        print(f"[✓] wrote {args.out_all} rows={len(df_all):,}")


if __name__ == "__main__":
    main()
