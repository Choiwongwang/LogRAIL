#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build train-only normal/anomaly template corpora using split_manifest.csv."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import List, Tuple

import pandas as pd

TRUE_TOKENS = {"1", "true", "y", "yes", "anomaly", "abnormal", "pos", "positive", "e", "f"}
FALSE_TOKENS = {"0", "false", "n", "no", "normal", "neg", "negative", "i", "w", "d"}

KEYWORD_BAD = re.compile(
    r"(error|fail(?:ed|ure)?|panic|fatal|anr|oom|denied|"
    r"missed|not\s+found|could\s+not\s+find|service\s+died|binder\s+died|"
    r"timeout|unavailable|abort|crash|killed|permission|segfault)",
    re.I,
)


def to_bool_anomaly(v) -> bool:
    s = str(v).strip().lower()
    if s in TRUE_TOKENS:
        return True
    if s in FALSE_TOKENS:
        return False
    return False


def normalize_tpl(s: str) -> str:
    return " ".join(str(s).strip().split())


def train_row_mask_from_manifest(
    n_rows: int, window_size: int, train_orig_idx: List[int]
) -> List[bool]:
    mask = [False] * n_rows
    for w in train_orig_idx:
        s = int(w) * window_size
        e = min((int(w) + 1) * window_size, n_rows)
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

    if label_col in df.columns:
        is_anom = df[label_col].apply(to_bool_anomaly)
    elif level_col in df.columns:
        lv = df[level_col].astype(str).str.strip().str.upper()
        is_anom = lv.isin(["E", "F"])
    else:
        raise ValueError("Either label_col or level_col is required.")

    sel = is_anom if want_anomaly else ~is_anom
    df_sel = df.loc[sel, ["EventTemplate"]].copy()
    n_raw = len(df_sel)

    removed_by_keyword = 0
    kept_by_keyword = None
    removed_benign = 0
    if not want_anomaly:
        before = len(df_sel)
        df_sel = df_sel[~df_sel["EventTemplate"].astype(str).str.contains(KEYWORD_BAD)]
        removed_by_keyword = before - len(df_sel)
    else:
        if require_error_keywords:
            m = df_sel["EventTemplate"].astype(str).str.contains(KEYWORD_BAD)
            kept_by_keyword = int(m.sum())
            df_sel = df_sel[m]
        if benign_patterns:
            regs = []
            with open(benign_patterns, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    regs.append(re.compile(line, re.I))

            def _is_benign(s: str) -> bool:
                t = str(s) or ""
                return any(rx.search(t) for rx in regs)

            if regs:
                mask_benign = df_sel["EventTemplate"].astype(str).map(_is_benign)
                removed_benign = int(mask_benign.sum())
                df_sel = df_sel[~mask_benign]

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
    ap.add_argument("--src_csv", required=True)
    ap.add_argument("--manifest", required=True, help="split_manifest.csv from preprocess")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--level_col", default="Level")
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--out_normal", required=True)
    ap.add_argument("--out_anomaly", required=True)
    ap.add_argument("--require_error_keywords", action="store_true")
    ap.add_argument("--benign_patterns", default=None)
    ap.add_argument("--out_train_csv", default=None)
    ap.add_argument(
        "--check_json",
        default=None,
        help="write vdb_source_check.json (default: next to out_normal)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    man = pd.read_csv(args.manifest)
    train_rows = man[man["split"] == "train"]
    other = man[man["split"].isin(["val", "test"])]
    train_orig = train_rows["orig_idx"].astype(int).tolist()
    other_orig = set(other["orig_idx"].astype(int).tolist())

    # leakage check: train orig must not overlap val/test
    overlap = sorted(set(train_orig) & other_orig)
    if overlap:
        raise SystemExit(f"[x] manifest leakage: train overlaps val/test orig_idx={overlap[:10]}")

    n_rows = len(df)
    mask = train_row_mask_from_manifest(n_rows, args.window_size, train_orig)
    df_train = df.loc[mask].copy()

    # also verify no val/test window line ranges appear if we can
    for _, row in other.iterrows():
        s, e = int(row["start_line"]), int(row["end_line"])
        if any(mask[i] for i in range(s, min(e + 1, n_rows))):
            raise SystemExit(
                f"[x] train mask covers non-train window orig={row['orig_idx']} lines={s}-{e}"
            )

    if args.out_train_csv:
        df_train.to_csv(args.out_train_csv, index=False, encoding="utf-8")

    normal_tpls, nstats = extract_templates(
        df_train, args.label_col, args.level_col, want_anomaly=False
    )
    anomaly_tpls, astats = extract_templates(
        df_train,
        args.label_col,
        args.level_col,
        want_anomaly=True,
        require_error_keywords=args.require_error_keywords,
        benign_patterns=args.benign_patterns,
    )

    os.makedirs(os.path.dirname(args.out_normal) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_anomaly) or ".", exist_ok=True)
    pd.DataFrame({"EventTemplate": normal_tpls}).to_csv(
        args.out_normal, index=False, encoding="utf-8"
    )
    pd.DataFrame({"EventTemplate": anomaly_tpls}).to_csv(
        args.out_anomaly, index=False, encoding="utf-8"
    )

    check = {
        "ok": True,
        "manifest": os.path.abspath(args.manifest),
        "train_windows": len(train_orig),
        "val_test_windows": len(other_orig),
        "train_rows": int(len(df_train)),
        "n_rows_total": int(n_rows),
        "overlap_orig_idx": overlap,
        "normal_templates": int(len(normal_tpls)),
        "anomaly_templates": int(len(anomaly_tpls)),
        "normal_stats": nstats,
        "anomaly_stats": astats,
    }
    check_path = args.check_json or os.path.join(
        os.path.dirname(args.out_normal) or ".", "vdb_source_check.json"
    )
    with open(check_path, "w", encoding="utf-8") as f:
        json.dump(check, f, indent=2, ensure_ascii=False)

    print(f"[✓] train rows: {len(df_train):,} / {n_rows:,}")
    print(f"[✓] normal templates: {len(normal_tpls):,} → {args.out_normal}")
    print(f"[✓] anomaly templates: {len(anomaly_tpls):,} → {args.out_anomaly}")
    print(f"[✓] source check: {check_path}")


if __name__ == "__main__":
    main()
