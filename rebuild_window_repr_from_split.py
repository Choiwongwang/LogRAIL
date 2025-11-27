#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild window_id → representative EventTemplate mapping using the same
non-overlapping window split/shuffle rules as the NPZ preprocessing.

Why:
- The prediction CSV window_id is not the original line index; it is the
  concatenation of shuffled train.npz + test.npz order.
- Therefore `loc[window_id]` on the raw CSV is wrong; this script rebuilds
  the mapping to align window_id = [train_shuffle..., test_shuffle...].

Inputs
- --src_csv : Raw structured log CSV (must include EventTemplate and label/Level)
- --label_col : Label column name (default: label)
- --window_size : Window size used in preprocessing (e.g., 20)
- --train_ratio : Train split ratio used in preprocessing (e.g., 0.7)
- --seed : Seed used for shuffling (e.g., 42)
- --pred_csv : (optional) prediction CSV for sanity check
- --out : Output CSV path

Output
- CSV `[window_id, EventTemplate]` in the same order as the prediction CSV
"""

import argparse, os
import numpy as np
import pandas as pd
from collections import Counter

THIS_DIR = os.path.dirname(__file__)
DEFAULTS = {
    "src_csv": os.path.join(THIS_DIR, "dataset", "Android.csv"),
    "label_col": "label",
    "window_size": 20,
    "train_ratio": 0.7,
    "seed": 42,
    "pred_csv": os.path.join(THIS_DIR, "output", "anomaly_logs_detected_by_logformer.csv"),
    "out": os.path.join(THIS_DIR, "output", "window_repr_by_pred_info.csv"),
}


def _info_score(tpl: str) -> float:
    """Heuristic information score for a template.

    High when the template has many meaningful tokens and few wildcards.
    """
    if not tpl:
        return 0.0
    tokens = str(tpl).split()
    if not tokens:
        return 0.0
    wildcard_tokens = sum(1 for tok in tokens if '<*>' in tok or tok in {'*', '<num>'})
    alpha_tokens = sum(1 for tok in tokens if any(ch.isalpha() for ch in tok))
    num_tokens = len(tokens)
    # Encourage longer/alphabetic templates, penalise wildcard-heavy ones.
    return num_tokens + 0.5 * alpha_tokens - 2.0 * wildcard_tokens

TRUE_TOKENS = {"1","true","y","yes","anomaly","abnormal","pos","positive","e","f"}
FALSE_TOKENS = {"0","false","n","no","normal","neg","negative","i","w","d"}

def to_bool_anom(v):
    v = str(v).strip().lower()
    if v in TRUE_TOKENS: return True
    if v in FALSE_TOKENS: return False
    return any(k in v for k in ["anomaly","error","fail","panic","fatal"])  # heuristic


def build_repr_templates(df: pd.DataFrame, label_col: str, window_size: int):
    N = len(df)
    num_win = N // window_size
    reps = []  # index -> repr template
    et = df['EventTemplate'].astype(str).tolist()
    lab = (df[label_col] if label_col in df.columns else df.get('Level','normal')).tolist()

    for i in range(num_win):
        s, e = i * window_size, (i + 1) * window_size
        slice_templates = et[s:e]
        slice_labels = lab[s:e]
        slice_scores = [_info_score(tpl) for tpl in slice_templates]

        # If all scores are tiny, allow a loose threshold fallback
        score_threshold = max(0.5, max(slice_scores) * 0.3)

        # Candidates sorted by information score
        candidates = sorted(
            range(window_size),
            key=lambda idx: (slice_scores[idx], -idx),
            reverse=True,
        )

        rep = None

        # 1) Prefer an informative anomaly template
        for offset in candidates:
            if slice_scores[offset] >= score_threshold and to_bool_anom(slice_labels[offset]):
                rep = slice_templates[offset]
                break

        # 2) Otherwise, pick the most frequent informative template
        if rep is None:
            filtered = [slice_templates[idx] for idx in candidates if slice_scores[idx] >= score_threshold]
            if filtered:
                c = Counter(filtered)
                rep = c.most_common(1)[0][0]

        # 3) Fallback: first anomaly, else mode
        if rep is None:
            for j in range(s, e):
                if to_bool_anom(lab[j]):
                    rep = et[j]
                    break
        if rep is None:
            c = Counter(slice_templates)
            rep = c.most_common(1)[0][0]

        reps.append(rep)
    return reps  # len = num_win


def main():
    ap = argparse.ArgumentParser("Rebuild window_id→EventTemplate mapping matching shuffled train/test split")
    ap.add_argument('--src_csv', default=DEFAULTS["src_csv"])
    ap.add_argument('--label_col', default=DEFAULTS["label_col"])
    ap.add_argument('--window_size', type=int, default=DEFAULTS["window_size"])
    ap.add_argument('--train_ratio', type=float, default=DEFAULTS["train_ratio"])
    ap.add_argument('--seed', type=int, default=DEFAULTS["seed"])
    ap.add_argument('--pred_csv', default=DEFAULTS["pred_csv"])
    ap.add_argument('--out', default=DEFAULTS["out"])
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    if 'EventTemplate' not in df.columns:
        raise ValueError('src_csv must contain EventTemplate column')

    reps = build_repr_templates(df, args.label_col, args.window_size)

    # Reproduce train/test split (same as preprocessing)
    num_win = len(reps)
    idx = list(range(num_win))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(idx)
    n_train = int(len(idx) * args.train_ratio)
    idx_train, idx_test = idx[:n_train], idx[n_train:]

    # pred window_id order = shuffled train followed by shuffled test
    rows = []
    for wid, orig in enumerate(idx_train):
        rows.append((wid, reps[orig]))
    for j, orig in enumerate(idx_test):
        rows.append((n_train + j, reps[orig]))

    out = pd.DataFrame(rows, columns=['window_id','EventTemplate'])

    # Optional sanity check against pred_csv
    if args.pred_csv:
        pred = pd.read_csv(args.pred_csv)
        uniq = pred['window_id'].nunique()
        if uniq != len(out):
            print(f"[!] sanity: pred unique window_id={uniq} vs built={len(out)} (mismatch)")
        else:
            print(f"[i] sanity: pred unique window_id={uniq} (OK)")

    out.to_csv(args.out, index=False)
    print(f"[✓] wrote {args.out} rows={len(out):,}  (train={n_train:,} test={len(idx_test):,})")

if __name__ == '__main__':
    main()
