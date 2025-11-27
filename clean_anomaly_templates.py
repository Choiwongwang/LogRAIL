#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/clean_anomaly_templates.py
──────────────────────────────────────────────────────────────
Extract anomaly log templates from a labeled CSV to build an anomaly corpus.

Rules (mirror of normal script):
1) If label column exists, label==1 is anomaly (strings like "1","anomaly","true","e","f"... handled; "0","normal","false","i","w","d"... → normal).
2) If no label, with a Level column, treat E/F as anomaly and others as normal.
3) Optional `--require_error_keywords`: keep only templates with error-like keywords.
4) Optional `--benign_patterns`: drop templates matching benign regex list.
5) Normalize whitespace, deduplicate, save EventTemplate only.

Example:
  python tools/clean_anomaly_templates.py \
      --src_csv dataset/Android.csv \
      --label_col label \
      --out dataset/anomaly_templates_clean.csv

Advanced example (require keywords + remove benign):
  python tools/clean_anomaly_templates.py \
      --src_csv dataset/Android.csv \
      --out dataset/anomaly_templates_strict.csv \
      --require_error_keywords \
      --benign_patterns config/benign_patterns.txt
"""

import argparse, re, pandas as pd

TRUE_TOKENS  = {"1","true","y","yes","anomaly","abnormal","pos","positive","e","f"}
FALSE_TOKENS = {"0","false","n","no","normal","neg","negative","i","w","d"}

# Error keywords (case-insensitive), same as normal corpus
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

def load_benign_regexes(path: str):
    regs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # skip comments/blank lines
                continue
            regs.append(re.compile(line, re.I))
    return regs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True, help="Source log CSV (EventTemplate required)")
    ap.add_argument("--label_col", default="label", help="Label column (0/1); if absent, Level is used")
    ap.add_argument("--level_col", default="Level", help="Log level column (e.g., E/F/I/W/D)")
    ap.add_argument("--out", required=True, help="Output CSV path for anomaly templates")
    ap.add_argument("--require_error_keywords", action="store_true",
                    help="Keep only templates containing error-like keywords (precision-focused)")
    ap.add_argument("--benign_patterns", default=None,
                    help="Benign regex list file; matched templates will be removed")
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    if "EventTemplate" not in df.columns:
        raise ValueError("src_csv must contain EventTemplate column.")

    # Select anomaly rows
    if args.label_col in df.columns:
        is_anom = df[args.label_col].apply(to_bool_anomaly)
    elif args.level_col in df.columns:
        lv = df[args.level_col].astype(str).str.strip().str.upper()
        is_anom = lv.isin(["E","F"])
    else:
        raise ValueError("Either label_col or level_col is required.")

    df_anom = df.loc[is_anom, ["EventTemplate"]].copy()
    n0 = len(df_anom)

    # (Optional) remove benign matches
    removed_benign = 0
    if args.benign_patterns:
        regs = load_benign_regexes(args.benign_patterns)
        if regs:
            def _is_benign(s: str) -> bool:
                t = str(s) or ""
                for rx in regs:
                    if rx.search(t):
                        return True
                return False
            mask_benign = df_anom["EventTemplate"].astype(str).map(_is_benign)
            removed_benign = int(mask_benign.sum())
            df_anom = df_anom[~mask_benign]

    # (Optional) require error keywords
    kept_by_keyword = None
    if args.require_error_keywords:
        m = df_anom["EventTemplate"].astype(str).str.contains(KEYWORD_BAD)
        kept_by_keyword = int(m.sum())
        df_anom = df_anom[m]

    # Normalize + deduplicate
    df_anom["EventTemplate"] = df_anom["EventTemplate"].apply(normalize_tpl)
    uniq = df_anom["EventTemplate"].dropna().drop_duplicates().sort_values()
    out = pd.DataFrame({"EventTemplate": uniq})

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[✓] saved: {args.out}  templates={len(out):,}")
    print(f"[i] anomalies(raw)={n0:,}  removed_benign={removed_benign:,}  "
          f"kept_by_keyword={kept_by_keyword if kept_by_keyword is not None else '-'}  "
          f"uniq_final={len(out):,}")

if __name__ == "__main__":
    main()
