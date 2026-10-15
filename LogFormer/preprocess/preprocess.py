#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window preprocess -> NPZ + split_manifest.csv (time | random)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from postprocess.llm_registry import default_distilbert  # noqa: E402

DEFAULT_CSV = os.path.abspath(os.path.join(_REPO, "dataset", "Android_new_full.csv"))
DEFAULT_OUT = os.path.abspath(os.path.join(_HERE, "..", "npz"))

WINDOW_SIZE = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def normalize_tpl(s: str) -> str:
    return " ".join(str(s).strip().split())


def make_window_label(labels):
    return np.array([0, 1], np.int64) if (labels == 1).any() else np.array([1, 0], np.int64)


def encode_windows(df, tpl2vec, window_size: int):
    num_win = math.ceil(len(df) / window_size)
    if not num_win:
        return (
            np.array([], object),
            np.array([], object),
            [],
        )

    pad_vec = np.zeros_like(next(iter(tpl2vec.values())))
    X_list, Y_list, meta_rows = [], [], []
    for i in tqdm(range(num_win), desc="Building windows", ncols=80):
        start = i * window_size
        end = min(start + window_size, len(df))
        block = df.iloc[start:end]

        vecs = [tpl2vec[t] for t in block["EventTemplate_norm"]]
        if len(vecs) < window_size:
            vecs.extend([pad_vec] * (window_size - len(vecs)))
        vecs = np.stack(vecs, axis=0)

        labels = block["label"].values
        if len(labels) < window_size:
            labels = np.concatenate(
                [labels, np.zeros(window_size - len(labels), dtype=labels.dtype)]
            )

        y = make_window_label(labels)
        X_list.append(vecs)
        Y_list.append(y)
        meta_rows.append(
            {
                "orig_idx": i,
                "start_line": int(start),
                "end_line": int(end - 1),
                "label": int(np.argmax(y)),
            }
        )

    return np.array(X_list, object), np.array(Y_list, object), meta_rows


def save_npz(path, x, y):
    np.savez(path, x=np.array(x, dtype=object), y=np.array(y, dtype=object))
    print(f"[✓] {os.path.basename(path):<25}  x:{len(x):>5}  y:{len(y):>5}")


def stratified_split_indices(y, train_ratio, val_ratio, test_ratio, seed):
    idx0 = [i for i, v in enumerate(y) if v == 0]
    idx1 = [i for i, v in enumerate(y) if v == 1]
    rng = random.Random(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split_indices(idx):
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    t0, v0, te0 = split_indices(idx0)
    t1, v1, te1 = split_indices(idx1)
    train_idx = t0 + t1
    val_idx = v0 + v1
    test_idx = te0 + te1
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def time_split_indices(n: int, train_ratio, val_ratio, test_ratio, gap: int):
    """Sequential cut with gap windows dropped around both cut points."""
    if n <= 0:
        return [], [], [], set()

    # provisional cut sizes on the full sequence (before gap removal)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # ensure non-empty when possible
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(0, n - 2)
    n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train - 1)

    cut1 = n_train  # first index of val region (pre-gap)
    cut2 = n_train + n_val  # first index of test region (pre-gap)

    dropped = set()
    g = max(0, int(gap))
    for c in (cut1, cut2):
        for j in range(c - g, c + g):
            if 0 <= j < n:
                dropped.add(j)

    train_idx, val_idx, test_idx = [], [], []
    for i in range(n):
        if i in dropped:
            continue
        if i < cut1:
            train_idx.append(i)
        elif i < cut2:
            val_idx.append(i)
        else:
            test_idx.append(i)

    return train_idx, val_idx, test_idx, dropped


def main(args):
    os.makedirs(args.out, exist_ok=True)
    window_size = args.window_size
    embed_model = args.embed_model or default_distilbert()

    df = pd.read_csv(args.csv)
    # tolerate alternate template column names
    if "EventTemplate" not in df.columns:
        raise SystemExit("[x] CSV must contain EventTemplate column")
    if "label" not in df.columns:
        raise SystemExit("[x] CSV must contain label column")

    df["EventTemplate_norm"] = df["EventTemplate"].apply(normalize_tpl)
    df["label"] = df["label"].astype(int)
    print(
        f"[INFO] Loaded {len(df):,} rows  "
        f"(normal={sum(df.label == 0):,}, anomaly={sum(df.label == 1):,})"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(embed_model, device=device)
    uniq_tpls = df["EventTemplate_norm"].unique().tolist()
    print(f"[STEP] Encoding {len(uniq_tpls):,} unique templates → {embed_model} ({device})")
    tpl_vecs = model.encode(uniq_tpls, batch_size=256, show_progress_bar=True)
    tpl2vec = {t: v for t, v in zip(uniq_tpls, tpl_vecs)}

    X, Y, win_meta = encode_windows(df, tpl2vec, window_size)
    print(f"[INFO] Total windows built: {len(X):,}")
    y_labels = [int(np.argmax(y)) for y in Y]

    dropped = set()
    if args.split_mode == "random":
        idx_train, idx_val, idx_test = stratified_split_indices(
            y_labels, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    else:
        idx_train, idx_val, idx_test, dropped = time_split_indices(
            len(X), args.train_ratio, args.val_ratio, args.test_ratio, args.gap
        )

    print(
        f"[INFO] split_mode={args.split_mode} gap={args.gap} "
        f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} "
        f"dropped={len(dropped)}"
    )

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    X_test, Y_test = X[idx_test], Y[idx_test]

    save_npz(os.path.join(args.out, "train.npz"), X_train, Y_train)
    save_npz(os.path.join(args.out, "val.npz"), X_val, Y_val)
    save_npz(os.path.join(args.out, "test.npz"), X_test, Y_test)

    # Global window_id order = train || val || test (matches infer default splits)
    manifest_rows = []
    wid = 0
    for split_name, idxs in (
        ("train", idx_train),
        ("val", idx_val),
        ("test", idx_test),
    ):
        for local_i, orig in enumerate(idxs):
            m = win_meta[orig]
            manifest_rows.append(
                {
                    "window_id": wid,
                    "orig_idx": int(orig),
                    "split": split_name,
                    "split_local_idx": local_i,
                    "label": int(m["label"]),
                    "start_line": int(m["start_line"]),
                    "end_line": int(m["end_line"]),
                }
            )
            wid += 1

    # also record dropped windows for audit
    for orig in sorted(dropped):
        m = win_meta[orig]
        manifest_rows.append(
            {
                "window_id": -1,
                "orig_idx": int(orig),
                "split": "gap_drop",
                "split_local_idx": -1,
                "label": int(m["label"]),
                "start_line": int(m["start_line"]),
                "end_line": int(m["end_line"]),
            }
        )

    man_path = os.path.join(args.out, "split_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(man_path, index=False)
    print(f"[✓] saved {man_path}")

    def _anom_rate(idxs):
        if not idxs:
            return 0.0
        return float(np.mean([y_labels[i] for i in idxs]))

    meta = {
        "window_size": window_size,
        "embedding_model": embed_model,
        "split_mode": args.split_mode,
        "gap": int(args.gap),
        "seed": int(args.seed),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train_windows": len(idx_train),
        "val_windows": len(idx_val),
        "test_windows": len(idx_test),
        "dropped_windows": len(dropped),
        "train_anomaly_rate": _anom_rate(idx_train),
        "val_anomaly_rate": _anom_rate(idx_val),
        "test_anomaly_rate": _anom_rate(idx_test),
        "csv": os.path.abspath(args.csv),
        "manifest": man_path,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[✓] saved meta.json")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess logs -> NPZ + split_manifest")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--split_mode", choices=["time", "random"], default="time")
    ap.add_argument("--gap", type=int, default=5, help="gap windows dropped at each time cut")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    ap.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    ap.add_argument(
        "--embed_model",
        default=None,
        help="local path / registry key / HF id (default: local distilbert)",
    )
    main(ap.parse_args())
