#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_window_repr_from_split.py
──────────────────────────────────────────────────────────────
Reproduces the NPZ preprocessing rules (non-overlapping windows + padding +
random shuffled train/test split) to build window-level representative
EventTemplates that match the **window_id order in the prediction CSV** 1:1.

Why is this needed?
- In the prediction CSV, `window_id` is NOT the original line index.
  It is the index after shuffling and then concatenating train.npz followed by test.npz.
- If the preprocessing keeps the tail window via padding (using ceil),
  this script must also use ceil to include the final padded window,
  otherwise the number of window_ids will not match.
- Therefore, extracting templates via `loc[window_id]` from the original CSV is incorrect.
- This script reproduces the preprocessing split rule (window_size/train_ratio/seed)
  and generates a `[window_id, EventTemplate]` CSV aligned with:
    window_id = [0..n_train-1, n_train..n_train+n_test-1]

Inputs
- --src_csv     : source line-level CSV (must include EventTemplate and label/Level)
- --label_col   : label column name (default: 'label')
- --window_size : window size used in preprocessing (e.g., 20)
- --train_ratio : train split ratio used in preprocessing (e.g., 0.7)
- --seed        : seed used in preprocessing (e.g., 42)
- --pred_csv    : (optional) prediction CSV path for sanity-checking counts
- --out         : output CSV path (single-mode)

Outputs
- CSV with columns `[window_id, EventTemplate]` aligned to the prediction CSV window_id order
"""

import argparse
import math
import random
import numpy as np
import pandas as pd
from collections import Counter


def _info_score(tpl: str) -> float:
    """Heuristic information score for a template.

    Higher when the template contains more meaningful tokens and fewer wildcards.
    """
    if not tpl:
        return 0.0
    tokens = str(tpl).split()
    if not tokens:
        return 0.0
    wildcard_tokens = sum(1 for tok in tokens if "<*>" in tok or tok in {"*", "<num>"})
    alpha_tokens = sum(1 for tok in tokens if any(ch.isalpha() for ch in tok))
    other_tokens = len(tokens) - alpha_tokens - wildcard_tokens
    # Reweighted: alphabetic tokens full weight, numeric/other half, heavy penalty for wildcards
    return alpha_tokens + 0.5 * other_tokens - 2.0 * wildcard_tokens


TRUE_TOKENS = {"1", "true", "y", "yes", "anomaly", "abnormal", "pos", "positive", "e", "f"}
FALSE_TOKENS = {"0", "false", "n", "no", "normal", "neg", "negative", "i", "w", "d"}


def to_bool_anom(v):
    v = str(v).strip().lower()
    if v in TRUE_TOKENS:
        return True
    if v in FALSE_TOKENS:
        return False
    return any(k in v for k in ["anomaly", "error", "fail", "panic", "fatal"])  # heuristic fallback


def build_repr_templates(df: pd.DataFrame, label_col: str, window_size: int, prefer_anomaly: bool):
    N = len(df)
    num_win = math.ceil(N / window_size)
    if num_win == 0:
        return []

    reps = []  # window_index -> representative template
    et = df["EventTemplate"].astype(str).tolist()
    lab = (df[label_col] if label_col in df.columns else df.get("Level", "normal")).tolist()

    for i in range(num_win):
        s, e = i * window_size, (i + 1) * window_size
        slice_templates = et[s:min(e, N)]
        slice_labels = lab[s:min(e, N)]
        if not slice_templates:
            reps.append("")
            continue

        slice_scores = [_info_score(tpl) for tpl in slice_templates]

        # If scores are generally low, fall back to simpler selection logic
        score_threshold = max(0.5, max(slice_scores) * 0.3)

        # Candidate indices ordered by descending information score (tie-breaker: earlier index)
        candidates = sorted(
            range(len(slice_templates)),
            key=lambda idx: (slice_scores[idx], -idx),
            reverse=True,
        )

        # Filter by information score threshold
        filtered_idxs = [idx for idx in candidates if slice_scores[idx] >= score_threshold]

        # 1) Prefer the requested label (anomaly/normal) among informative candidates
        rep = None
        for idx in filtered_idxs:
            is_anom = to_bool_anom(slice_labels[idx])
            if (prefer_anomaly and is_anom) or ((not prefer_anomaly) and (not is_anom)):
                rep = slice_templates[idx]
                break

        # 2) If none match the preferred label, pick the most frequent among informative candidates
        if rep is None and filtered_idxs:
            filtered = [slice_templates[idx] for idx in filtered_idxs]
            c = Counter(filtered)
            rep = c.most_common(1)[0][0]

        # 3) Final fallback: first occurrence of preferred label, else global most frequent
        if rep is None:
            for j in range(len(slice_templates)):
                is_anom = to_bool_anom(slice_labels[j])
                if (prefer_anomaly and is_anom) or ((not prefer_anomaly) and (not is_anom)):
                    rep = slice_templates[j]
                    break
        if rep is None:
            c = Counter(slice_templates)
            rep = c.most_common(1)[0][0]

        reps.append(rep)

    return reps  # length = num_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--window_size", type=int, required=True)
    ap.add_argument("--train_ratio", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pred_csv", default=None)
    ap.add_argument(
        "--mode",
        choices=["anomaly", "normal", "both"],
        default="anomaly",
        help="Representative template preference: anomaly / normal / both (generate two outputs)",
    )
    ap.add_argument("--out", required=False, help="Output path for single-mode (anomaly/normal)")
    ap.add_argument("--out_anomaly", required=False, help="Output path when mode=both (prefer anomaly)")
    ap.add_argument("--out_normal", required=False, help="Output path when mode=both (prefer normal)")
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    if "EventTemplate" not in df.columns:
        raise ValueError("src_csv must contain an EventTemplate column")

    mode = args.mode
    if mode in ("anomaly", "normal") and not args.out:
        raise ValueError("--out is required when mode is anomaly/normal")
    if mode == "both" and (not args.out_anomaly or not args.out_normal):
        raise ValueError("mode=both requires --out_anomaly and --out_normal")

    def build_and_save(prefer_anomaly: bool, out_path: str):
        reps = build_repr_templates(df, args.label_col, args.window_size, prefer_anomaly)

        # Reproduce the same shuffled split as preprocessing (Python random)
        num_win = len(reps)
        idx = list(range(num_win))
        random.seed(args.seed)
        random.shuffle(idx)

        n_train = int(len(idx) * args.train_ratio)
        idx_train, idx_test = idx[:n_train], idx[n_train:]

        rows = []
        for wid, orig in enumerate(idx_train):
            rows.append((wid, reps[orig]))
        for j, orig in enumerate(idx_test):
            rows.append((n_train + j, reps[orig]))

        out = pd.DataFrame(rows, columns=["window_id", "EventTemplate"])

        if args.pred_csv:
            pred = pd.read_csv(args.pred_csv)
            uniq = pred["window_id"].nunique()
            if uniq != len(out):
                print(f"[!] sanity: pred unique window_id={uniq} vs built={len(out)} (mismatch)")
            else:
                print(f"[i] sanity: pred unique window_id={uniq} (OK)")

        out.to_csv(out_path, index=False)
        print(f"[✓] wrote {out_path} rows={len(out):,}  (train={n_train:,} test={len(idx_test):,})")

    if mode in ("anomaly", "normal"):
        build_and_save(prefer_anomaly=(mode == "anomaly"), out_path=args.out)
    else:  # both
        build_and_save(prefer_anomaly=True, out_path=args.out_anomaly)
        build_and_save(prefer_anomaly=False, out_path=args.out_normal)


if __name__ == "__main__":
    main()
