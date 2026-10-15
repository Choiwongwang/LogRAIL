#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild window templates aligned to window_id using split_manifest.csv."""

from __future__ import annotations

import argparse
import json
import math
from typing import List

import pandas as pd


def build_window_templates(df: pd.DataFrame, window_size: int, pad_token: str) -> List[List[str]]:
    if "EventTemplate" not in df.columns:
        raise ValueError("src_csv must contain an EventTemplate column")
    et = df["EventTemplate"].astype(str).tolist()
    n = len(et)
    num_win = math.ceil(n / window_size)
    windows: List[List[str]] = []
    for i in range(num_win):
        s, e = i * window_size, (i + 1) * window_size
        chunk = et[s : min(e, n)]
        if len(chunk) < window_size:
            chunk = chunk + [pad_token] * (window_size - len(chunk))
        windows.append(chunk)
    return windows


def export_windows(ordered_windows: List[List[str]], window_ids: List[int], out_path: str, fmt: str):
    rows = []
    if fmt == "long":
        for wid, w in zip(window_ids, ordered_windows):
            for pos, tpl in enumerate(w):
                rows.append((wid, pos, tpl))
        out = pd.DataFrame(rows, columns=["window_id", "pos", "EventTemplate"])
    elif fmt == "wide":
        window_size = len(ordered_windows[0]) if ordered_windows else 0
        cols = ["window_id"] + [f"tpl_{i}" for i in range(window_size)]
        for wid, w in zip(window_ids, ordered_windows):
            rows.append([wid] + list(w))
        out = pd.DataFrame(rows, columns=cols)
    elif fmt == "json":
        for wid, w in zip(window_ids, ordered_windows):
            rows.append((wid, json.dumps(w, ensure_ascii=False)))
        out = pd.DataFrame(rows, columns=["window_id", "templates_json"])
    else:
        raise ValueError(f"Unknown format: {fmt}")
    out.to_csv(out_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True)
    ap.add_argument(
        "--manifest",
        required=True,
        help="split_manifest.csv from preprocess (authoritative split order)",
    )
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--pred_csv", default=None)
    ap.add_argument("--pad_token", default="<PAD>")
    ap.add_argument("--format", choices=["long", "wide", "json"], default="json")
    ap.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="all = train||val||test in manifest window_id order",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    man = pd.read_csv(args.manifest)
    man_use = man[man["split"] != "gap_drop"].copy()
    if args.split != "all":
        man_use = man_use[man_use["split"] == args.split].copy()
    man_use = man_use.sort_values("window_id").reset_index(drop=True)

    windows = build_window_templates(df, args.window_size, args.pad_token)
    ordered, wids = [], []
    for _, row in man_use.iterrows():
        orig = int(row["orig_idx"])
        if orig < 0 or orig >= len(windows):
            raise SystemExit(f"[x] orig_idx out of range: {orig}")
        ordered.append(windows[orig])
        wids.append(int(row["window_id"]))

    if args.pred_csv:
        pred = pd.read_csv(args.pred_csv)
        uniq = pred["window_id"].nunique()
        if uniq != len(ordered):
            print(
                f"[!] sanity mismatch: pred unique window_id={uniq} vs built={len(ordered)}"
            )
        else:
            print(f"[i] sanity OK: pred unique window_id={uniq} == built={len(ordered)}")

    out_df = export_windows(ordered, wids, args.out, args.format)
    print(
        f"[✓] wrote {args.out} rows={len(out_df):,} windows={len(ordered):,} "
        f"(split={args.split})"
    )


if __name__ == "__main__":
    main()
