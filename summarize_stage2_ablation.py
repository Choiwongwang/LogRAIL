#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize Stage-2 ablation outputs into a single CSV.

It scans output/ablation/*/rag.csv, evaluates against gt_test.csv,
and reports metrics + flip counts vs Stage-1.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

import eval_final as evalmod


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default(path: str) -> str:
    return os.path.join(_repo_root(), path)


def _load_stage1(pred_csv: str, gt_csv: str) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    if "anomaly_pred" not in pred.columns and "is_anomaly_pred" in pred.columns:
        pred = pred.rename(columns={"is_anomaly_pred": "anomaly_pred"})
    pred = pred[["window_id", "anomaly_pred", "prob"]]
    gt = pd.read_csv(gt_csv)[["window_id", "label"]]
    base = pred.merge(gt, on="window_id", how="inner").dropna()
    base["anomaly_pred"] = base["anomaly_pred"].astype(int)
    base["label"] = base["label"].astype(int)
    return base


def _eval_one(rag_csv: str, base: pd.DataFrame) -> dict:
    rag = pd.read_csv(rag_csv)
    rag = rag[["window_id", "anomaly"]]
    rag["anomaly"] = rag["anomaly"].astype(int)

    merged = base.merge(rag, on="window_id", how="inner")
    m = evalmod._metrics(merged["label"], merged["anomaly"])

    flip = (merged["anomaly"] != merged["anomaly_pred"])
    benefit = (flip & (merged["anomaly"] == merged["label"])).sum()
    harm = (flip & (merged["anomaly"] != merged["label"])).sum()

    return {
        "rag_csv": rag_csv,
        "n_eval": int(len(merged)),
        "flip_count": int(flip.sum()),
        "flip_benefit": int(benefit),
        "flip_harm": int(harm),
        "net_gain": int(benefit - harm),
        **m,
    }


def main() -> None:
    ap = argparse.ArgumentParser("Summarize Stage-2 ablation runs")
    ap.add_argument("--ablation_dir", default=_default("output/ablation_val"))
    ap.add_argument(
        "--pred_csv",
        default=_default("output/logformer_preds_val.csv"),
    )
    ap.add_argument("--gt_csv", default=_default("output/gt_val.csv"))
    ap.add_argument("--out_csv", default=_default("output/ablation_summary_val.csv"))
    args = ap.parse_args()

    if not os.path.exists(args.ablation_dir):
        raise SystemExit(f"[x] ablation_dir not found: {args.ablation_dir}")
    if not os.path.exists(args.pred_csv):
        raise SystemExit(f"[x] pred_csv not found: {args.pred_csv}")
    if not os.path.exists(args.gt_csv):
        raise SystemExit(f"[x] gt_csv not found: {args.gt_csv}")

    base = _load_stage1(args.pred_csv, args.gt_csv)

    rag_paths = sorted(glob.glob(os.path.join(args.ablation_dir, "*", "rag.csv")))
    if not rag_paths:
        raise SystemExit(f"[x] no rag.csv found under: {args.ablation_dir}")

    rows = []
    for rag_csv in rag_paths:
        tag = os.path.basename(os.path.dirname(rag_csv))
        row = {"tag": tag}
        row.update(_eval_one(rag_csv, base))
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["tag"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[✓] wrote summary: {args.out_csv}  rows={len(df)}")


if __name__ == "__main__":
    main()
