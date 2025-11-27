#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/clean_normal_templates.py
──────────────────────────────────────────────────────────────
Extract only “normal” log templates from a labeled CSV to build a clean normal corpus.

Rules:
1) If a label column exists, label==0 is normal (string values like "0","normal","false"... are handled; "1","anomaly","true"... → anomaly).
2) If there is no label but a Level column, treat E/F as anomaly and others (I/W/D etc.) as normal.
3) Drop templates containing error-like keywords (blacklist).
4) Normalize whitespace, deduplicate, and keep only EventTemplate.

Example:
  python tools/clean_normal_templates.py \
      --src_csv dataset/Android.csv \
      --label_col label \
      --out dataset/normal_templates_clean.csv
"""

import argparse, re, pandas as pd

TRUE_TOKENS  = {"1","true","y","yes","anomaly","abnormal","pos","positive","e","f"}
FALSE_TOKENS = {"0","false","n","no","normal","neg","negative","i","w","d"}

# Error-like keywords (case-insensitive)
KEYWORD_BAD = re.compile(
    r"(error|fail(?:ed|ure)?|panic|fatal|anr|oom|denied|"
    r"missed|not\s+found|could\s+not\s+find|service\s+died|binder\s+died|"
    r"timeout|unavailable|abort|crash|killed|permission|segfault)",
    re.I
)

def to_bool_anomaly(v):
    s = str(v).strip().lower()
    if s in TRUE_TOKENS:  return True
    if s in FALSE_TOKENS: return False
    # If ambiguous, fall back to keyword heuristic
    return bool(KEYWORD_BAD.search(s))

def normalize_tpl(s: str) -> str:
    t = str(s).strip()
    return " ".join(t.split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True, help="Source log CSV (requires EventTemplate)")
    ap.add_argument("--label_col", default="label", help="Label column (0/1); if absent, Level is used")
    ap.add_argument("--level_col", default="Level", help="Log level column (e.g., E/F/I/W/D)")
    ap.add_argument("--out", required=True, help="Output CSV path for normal templates")
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    if "EventTemplate" not in df.columns:
        raise ValueError("src_csv must contain EventTemplate column.")

    # Select normal rows
    if args.label_col in df.columns:
        is_anom = df[args.label_col].apply(to_bool_anomaly)
        is_norm = ~is_anom
    elif args.level_col in df.columns:
        lv = df[args.level_col].astype(str).str.strip().str.upper()
        is_anom = lv.isin(["E","F"])
        is_norm = ~is_anom
    else:
        raise ValueError("Either label_col or level_col is required.")

    df_norm = df.loc[is_norm, ["EventTemplate"]].copy()
    n0 = len(df_norm)

    # Remove blacklist keywords
    df_norm = df_norm[~df_norm["EventTemplate"].astype(str).str.contains(KEYWORD_BAD)]
    n1 = len(df_norm)

    # Normalize + deduplicate
    df_norm["EventTemplate"] = df_norm["EventTemplate"].apply(normalize_tpl)
    uniq = df_norm["EventTemplate"].dropna().drop_duplicates().sort_values()
    out = pd.DataFrame({"EventTemplate": uniq})

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[✓] saved: {args.out}  templates={len(out):,}")
    print(f"[i] normals(raw)={n0:,}  removed_by_keyword={n0-n1:,}  uniq_final={len(out):,}")

if __name__ == "__main__":
    main()
