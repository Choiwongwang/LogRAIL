#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_window_templates_from_split.py
──────────────────────────────────────────────────────────────
Reproduce NPZ preprocessing window order (non-overlapping + ceil padding +
random shuffled train/test split + concat train then test) and export
window_id-aligned templates.

NEW 목적 (Stage-2용):
- 대표 템플릿 1개를 뽑지 않음
- 각 window_id에 대해 "그 윈도우의 템플릿 20개 전체"를 복원해서 제공
  → 이후 Stage-2에서 VDB 유사도 기반 top/bottom-5 선택에 사용

Why needed?
- prediction CSV의 window_id는 원본 라인 인덱스가 아니라,
  (윈도우 인덱스 리스트를 seed로 shuffle) → train/test split → train 뒤 test concat
  으로 만들어진 재정렬 인덱스임.
- 따라서 원본 CSV에서 loc[window_id] 같은 접근은 틀림.
- 본 스크립트는 preprocessing split 규칙(window_size/train_ratio/seed)을 재현해
  window_id 순서와 1:1로 맞는 window 템플릿 목록을 출력함.

Inputs
- --src_csv      : line-level CSV (must include EventTemplate)
- --window_size  : window size (e.g., 20)
- --train_ratio  : train split ratio (e.g., 0.7)
- --seed         : shuffle seed (e.g., 42)
- --pred_csv     : optional, sanity check (#unique window_id)
- --pad_token    : padding token for last partial window (default: "<PAD>")
- --format       : output format: long / wide / json
                  long:  window_id,pos,EventTemplate
                  wide:  window_id,tpl_0,...,tpl_19
                  json:  window_id,templates_json
- --out          : output CSV path

Notes
- 반드시 src_csv가 NPZ 생성 시 사용한 "동일한 전처리 결과(행 순서/필터링 포함)"여야 함.
- padding은 ceil 기반으로 마지막 윈도우를 유지하며, 부족한 칸은 pad_token으로 채움.
"""

import argparse
import json
import math
import random
from typing import List

import numpy as np
import pandas as pd


def build_window_templates(df: pd.DataFrame, window_size: int, pad_token: str) -> List[List[str]]:
    """Return list of windows, each is a list[str] length=window_size (padded)."""
    if "EventTemplate" not in df.columns:
        raise ValueError("src_csv must contain an EventTemplate column")

    et = df["EventTemplate"].astype(str).tolist()
    N = len(et)
    num_win = math.ceil(N / window_size)
    windows: List[List[str]] = []

    for i in range(num_win):
        s, e = i * window_size, (i + 1) * window_size
        chunk = et[s:min(e, N)]
        if len(chunk) < window_size:
            chunk = chunk + [pad_token] * (window_size - len(chunk))
        windows.append(chunk)

    return windows  # length = num_win


def build_window_labels(df: pd.DataFrame, window_size: int) -> List[int]:
    """
    Build window-level labels using OR rule over line-level labels.
    Requires a 'label' column (0/1).
    """
    if "label" not in df.columns:
        raise ValueError("src_csv must contain a 'label' column for stratified split.")
    labels = df["label"].astype(int).tolist()
    N = len(labels)
    num_win = math.ceil(N / window_size)
    win_labels: List[int] = []
    for i in range(num_win):
        s, e = i * window_size, (i + 1) * window_size
        block = labels[s:min(e, N)]
        if len(block) < window_size:
            block = block + [0] * (window_size - len(block))
        win_labels.append(1 if any(v == 1 for v in block) else 0)
    return win_labels


def _stratified_split_indices(y_labels: List[int], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    idx0 = [i for i, v in enumerate(y_labels) if v == 0]
    idx1 = [i for i, v in enumerate(y_labels) if v == 1]
    rng = random.Random(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split_indices(idx):
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    t0, v0, te0 = split_indices(idx0)
    t1, v1, te1 = split_indices(idx1)

    train_idx = t0 + t1
    val_idx = v0 + v1
    test_idx = te0 + te1
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def reorder_to_window_id(windows: List[List[str]], win_labels: List[int], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """
    Reproduce preprocessing (stratified):
    - split indices by window label (0/1)
    - shuffle within each class with seed
    - concatenate train then val then test as window_id order
    """
    train_idx, val_idx, test_idx = _stratified_split_indices(
        win_labels, train_ratio, val_ratio, test_ratio, seed
    )

    ordered = []
    mapping = []

    wid = 0
    for orig in train_idx:
        ordered.append(windows[orig])
        mapping.append((wid, orig))
        wid += 1
    for orig in val_idx:
        ordered.append(windows[orig])
        mapping.append((wid, orig))
        wid += 1
    for orig in test_idx:
        ordered.append(windows[orig])
        mapping.append((wid, orig))
        wid += 1

    return ordered, mapping, len(train_idx), len(val_idx), len(test_idx)


def export_windows(ordered_windows: List[List[str]], out_path: str, fmt: str):
    rows = []
    if fmt == "long":
        # window_id,pos,EventTemplate
        for wid, w in enumerate(ordered_windows):
            for pos, tpl in enumerate(w):
                rows.append((wid, pos, tpl))
        out = pd.DataFrame(rows, columns=["window_id", "pos", "EventTemplate"])

    elif fmt == "wide":
        # window_id,tpl_0..tpl_{window_size-1}
        window_size = len(ordered_windows[0]) if ordered_windows else 0
        cols = ["window_id"] + [f"tpl_{i}" for i in range(window_size)]
        for wid, w in enumerate(ordered_windows):
            rows.append([wid] + list(w))
        out = pd.DataFrame(rows, columns=cols)

    elif fmt == "json":
        # window_id,templates_json
        for wid, w in enumerate(ordered_windows):
            rows.append((wid, json.dumps(w, ensure_ascii=False)))
        out = pd.DataFrame(rows, columns=["window_id", "templates_json"])

    else:
        raise ValueError(f"Unknown format: {fmt}")

    out.to_csv(out_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True)
    ap.add_argument("--window_size", type=int, required=True)
    ap.add_argument("--train_ratio", type=float, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pred_csv", default=None)
    ap.add_argument("--pad_token", default="<PAD>")
    ap.add_argument("--format", choices=["long", "wide", "json"], default="json")
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all",
                    help="Which split order to export. 'all' matches train+test concat order.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    windows = build_window_templates(df, args.window_size, args.pad_token)
    win_labels = build_window_labels(df, args.window_size)

    ordered_windows, mapping, n_train, n_val, n_test = reorder_to_window_id(
        windows, win_labels, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, seed=args.seed
    )

    # If split is requested, slice and reindex window_id accordingly
    if args.split == "train":
        ordered_windows = ordered_windows[:n_train]
    elif args.split == "val":
        ordered_windows = ordered_windows[n_train:n_train + n_val]
    elif args.split == "test":
        ordered_windows = ordered_windows[n_train + n_val:]

    # sanity check with pred_csv (if given)
    if args.pred_csv:
        pred = pd.read_csv(args.pred_csv)
        uniq = pred["window_id"].nunique()
        if uniq != len(ordered_windows):
            print(f"[!] sanity mismatch: pred unique window_id={uniq} vs built={len(ordered_windows)}")
            print("    → 원인 후보: src_csv 행 수/필터링 불일치, window_size 불일치, ceil padding 규칙 불일치, seed/train_ratio 불일치")
        else:
            print(f"[i] sanity OK: pred unique window_id={uniq} == built={len(ordered_windows)}")

    out_df = export_windows(ordered_windows, args.out, args.format)
    if args.split == "all":
        print(f"[✓] wrote {args.out} rows={len(out_df):,} windows={len(ordered_windows):,} (train={n_train:,} val={n_val:,} test={n_test:,})")
    else:
        print(f"[✓] wrote {args.out} rows={len(out_df):,} windows={len(ordered_windows):,} (split={args.split})")

    # optional: mapping debug
    # print("[debug] first 5 mappings (window_id -> orig_window_idx):", mapping[:5])


if __name__ == "__main__":
    main()
