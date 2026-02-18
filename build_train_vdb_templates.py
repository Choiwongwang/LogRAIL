#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build train-only normal/anomaly template corpora for VDB.
This reproduces the LogFormer preprocessing window split (ceil + shuffle),
then extracts templates only from TRAIN windows to avoid data leakage.
"""

import argparse
import math
import random
import re
from typing import List, Tuple

import pandas as pd

TRUE_TOKENS  = {"1","true","y","yes","anomaly","abnormal","pos","positive","e","f"}
FALSE_TOKENS = {"0","false","n","no","normal","neg","negative","i","w","d"}

# Error-like keywords (case-insensitive)
KEYWORD_BAD = re.compile(
    r"(error|fail(?:ed|ure)?|panic|fatal|anr|oom|denied|"
    r"missed|not\s+found|could\s+not\s+find|service\s+died|binder\s+died|"
    r"timeout|unavailable|abort|crash|killed|permission|segfault)",
    re.I
)


def to_bool_anomaly(v) -> bool:
    s = str(v).strip().lower()
    if s in TRUE_TOKENS:
        return True
    if s in FALSE_TOKENS:
        return False
    return False


def normalize_tpl(s: str) -> str:
    t = str(s).strip()
    return " ".join(t.split())


def train_window_indices(n_rows: int, window_size: int, train_ratio: float, seed: int) -> List[int]:
    num_win = math.ceil(n_rows / window_size)
    idx = list(range(num_win))
    random.seed(seed)
    random.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    return idx[:n_train]


def train_row_mask(n_rows: int, window_size: int, train_win_idx: List[int]) -> List[bool]:
    mask = [False] * n_rows
    for w in train_win_idx:
        s = w * window_size
        e = min((w + 1) * window_size, n_rows)
        for i in range(s, e):
            mask[i] = True
    return mask


def extract_templates(
    df: pd.DataFrame,
    label_col: str,
    level_col: str,
    want_anomaly: bool,
    require_error_keywords: bool = False,
    benign_patterns: str | None = None,
) -> Tuple[pd.Series, dict]:
    if "EventTemplate" not in df.columns:
        raise ValueError("src_csv must contain EventTemplate column.")

    # Select anomaly/normal rows
    if label_col in df.columns:
        is_anom = df[label_col].apply(to_bool_anomaly)
    elif level_col in df.columns:
        lv = df[level_col].astype(str).str.strip().str.upper()
        is_anom = lv.isin(["E", "F"])
    else:
        raise ValueError("Either label_col or level_col is required.")

    if want_anomaly:
        sel = is_anom
    else:
        sel = ~is_anom

    df_sel = df.loc[sel, ["EventTemplate"]].copy()
    n_raw = len(df_sel)

    # Normal: remove error-like keywords
    removed_by_keyword = 0
    kept_by_keyword = None
    if not want_anomaly:
        df_sel = df_sel[~df_sel["EventTemplate"].astype(str).str.contains(KEYWORD_BAD)]
        removed_by_keyword = n_raw - len(df_sel)
    else:
        # Anomaly: optional keyword requirement
        if require_error_keywords:
            m = df_sel["EventTemplate"].astype(str).str.contains(KEYWORD_BAD)
            kept_by_keyword = int(m.sum())
            df_sel = df_sel[m]

        # Optional benign patterns removal
        removed_benign = 0
        if benign_patterns:
            regs = []
            with open(benign_patterns, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    regs.append(re.compile(line, re.I))
            if regs:
                def _is_benign(s: str) -> bool:
                    t = str(s) or ""
                    for rx in regs:
                        if rx.search(t):
                            return True
                    return False
                mask_benign = df_sel["EventTemplate"].astype(str).map(_is_benign)
                removed_benign = int(mask_benign.sum())
                df_sel = df_sel[~mask_benign]
        else:
            removed_benign = 0

    # Normalize + deduplicate
    df_sel["EventTemplate"] = df_sel["EventTemplate"].apply(normalize_tpl)
    uniq = df_sel["EventTemplate"].dropna().drop_duplicates().sort_values()

    stats = {
        "raw": n_raw,
        "removed_by_keyword": removed_by_keyword,
        "kept_by_keyword": kept_by_keyword,
        "removed_benign": removed_benign if want_anomaly else 0,
        "uniq_final": len(uniq),
    }
    return uniq, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True, help="Source log CSV (EventTemplate required)")
    ap.add_argument("--label_col", default="label", help="Label column (0/1); if absent, Level is used")
    ap.add_argument("--level_col", default="Level", help="Log level column (e.g., E/F/I/W/D)")
    ap.add_argument("--window_size", type=int, default=20, help="Window size (must match preprocess)")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio (must match preprocess)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (must match preprocess)")
    ap.add_argument("--out_normal", required=True, help="Output CSV for normal templates (train-only)")
    ap.add_argument("--out_anomaly", required=True, help="Output CSV for anomaly templates (train-only)")
    ap.add_argument("--require_error_keywords", action="store_true",
                    help="Anomaly only: keep only templates containing error-like keywords")
    ap.add_argument("--benign_patterns", default=None,
                    help="Anomaly only: benign regex list; matched templates will be removed")
    ap.add_argument("--out_train_csv", default=None,
                    help="Optional: save train-only line-level CSV for inspection")
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    n_rows = len(df)
    train_wins = train_window_indices(n_rows, args.window_size, args.train_ratio, args.seed)
    mask = train_row_mask(n_rows, args.window_size, train_wins)
    df_train = df.loc[mask].copy()

    if args.out_train_csv:
        df_train.to_csv(args.out_train_csv, index=False, encoding="utf-8")

    # Build normal/anomaly template corpora from TRAIN only
    normal_tpls, nstats = extract_templates(
        df_train, args.label_col, args.level_col, want_anomaly=False
    )
    anomaly_tpls, astats = extract_templates(
        df_train, args.label_col, args.level_col, want_anomaly=True,
        require_error_keywords=args.require_error_keywords,
        benign_patterns=args.benign_patterns
    )

    pd.DataFrame({"EventTemplate": normal_tpls}).to_csv(args.out_normal, index=False, encoding="utf-8")
    pd.DataFrame({"EventTemplate": anomaly_tpls}).to_csv(args.out_anomaly, index=False, encoding="utf-8")

    print(f"[✓] train rows: {len(df_train):,} / {n_rows:,}")
    print(f"[✓] normal templates: {len(normal_tpls):,} → {args.out_normal}")
    print(f"[✓] anomaly templates: {len(anomaly_tpls):,} → {args.out_anomaly}")
    print(f"[i] normal stats: raw={nstats['raw']:,} removed_by_keyword={nstats['removed_by_keyword']:,} uniq={nstats['uniq_final']:,}")
    print(f"[i] anomaly stats: raw={astats['raw']:,} removed_benign={astats['removed_benign']:,} kept_by_keyword={astats['kept_by_keyword']} uniq={astats['uniq_final']:,}")


if __name__ == "__main__":
    main()
