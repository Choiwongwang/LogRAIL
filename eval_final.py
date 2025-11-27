#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate final metrics by comparing RAG_v2/LogRAG outputs with GT.
Defaults are embedded so it can run without arguments.

Defaults:
- pred_csv : output/rag_result.csv         (RAG_v2 result)
- gt_csv   : output/gt_all.csv             (window-level GT)
- gt_label_col : label
- final_only   : True   (print final metrics only)
- export_report: output/metrics_final_only.json

Example (default):
    python eval_final.py

Override example:
    python eval_final.py --final_only False --export_report output/metrics_all.json
"""

import argparse, json, os
import pandas as pd

DEFAULTS = {
    "pred_csv": "output/rag_result.csv",
    "gt_csv": "output/gt_all.csv",
    "gt_label_col": "label",
    "raw_json": None,  # e.g., "output/rag_raw.json"
    "export_report": "output/metrics_final_only.json",
    "final_only": True,
}

def _metrics(y_true, y_pred):
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)
    TP = int(((y_pred==1) & (y_true==1)).sum())
    TN = int(((y_pred==0) & (y_true==0)).sum())
    FP = int(((y_pred==1) & (y_true==0)).sum())
    FN = int(((y_pred==0) & (y_true==1)).sum())
    prec = TP / (TP + FP) if (TP+FP)>0 else 0.0
    rec  = TP / (TP + FN) if (TP+FN)>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN)>0 else 0.0
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                precision=prec, recall=rec, f1=f1, acc=acc)

def _fmt_block(title, m):
    print(f"\n── {title} ──")
    print(f"Precision = {m['precision']:.4f}   Recall = {m['recall']:.4f}   F1 = {m['f1']:.4f}   Acc = {m['acc']:.4f}")
    print("Confusion-matrix  [ [TN FP] [FN TP] ]")
    print(f"[[ {m['TN']:>3} {m['FP']:>3}]\n [ {m['FN']:>3} {m['TP']:>3}]]")

def build_parser_with_defaults():
    ap = argparse.ArgumentParser("Evaluate final metrics (defaults embedded for no-arg run)")
    ap.add_argument("--pred_csv", default=DEFAULTS["pred_csv"])
    ap.add_argument("--gt_csv", default=DEFAULTS["gt_csv"])
    ap.add_argument("--gt_label_col", default=DEFAULTS["gt_label_col"])
    ap.add_argument("--raw_json", default=DEFAULTS["raw_json"])
    ap.add_argument("--export_report", default=DEFAULTS["export_report"])
    # boolean toggle helper
    ap.add_argument("--final_only", type=lambda s: str(s).lower() in ["1","true","t","yes","y"], default=DEFAULTS["final_only"])
    return ap

def main():
    ap = build_parser_with_defaults()
    args = ap.parse_args()

    # input checks
    if not os.path.exists(args.pred_csv):
        raise SystemExit(f"[x] pred_csv not found: {args.pred_csv}")
    if not os.path.exists(args.gt_csv):
        raise SystemExit(f"[x] gt_csv not found: {args.gt_csv}")

    pred = pd.read_csv(args.pred_csv)
    gt   = pd.read_csv(args.gt_csv)

    if "window_id" not in pred.columns or "window_id" not in gt.columns:
        raise SystemExit("window_id column is required.")

    if args.gt_label_col not in gt.columns:
        raise SystemExit(f"GT must have column {args.gt_label_col}.")

    df = pred.merge(gt[["window_id", args.gt_label_col]], on="window_id", how="left")
    df.rename(columns={args.gt_label_col: "label"}, inplace=True)

    # drop rows without labels
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    have_stage1 = "anomaly_pred" in df.columns
    have_final  = "anomaly" in df.columns

    report = {}

    if (not args.final_only) and have_stage1:
        m1 = _metrics(df["label"], df["anomaly_pred"])
        _fmt_block("LogFormer stage1", m1)
        report["stage1"] = m1

    if have_final:
        mf = _metrics(df["label"], df["anomaly"])
        _fmt_block("RAG + LLM final", mf)
        report["final"] = mf
    else:
        raise SystemExit("pred_csv is missing column 'anomaly'.")

    if (not args.final_only) and have_stage1:
        dP = report["final"]["precision"] - report["stage1"]["precision"]
        dR = report["final"]["recall"]    - report["stage1"]["recall"]
        dF = report["final"]["f1"]        - report["stage1"]["f1"]
        dA = report["final"]["acc"]       - report["stage1"]["acc"]
        print("\n── Improvement (Final - Stage1) ──")
        print(f"Δ precision = {dP:+.4f}")
        print(f"Δ recall    = {dR:+.4f}")
        print(f"Δ f1        = {dF:+.4f}")
        print(f"Δ accuracy  = {dA:+.4f}")
        report["delta"] = dict(precision=dP, recall=dR, f1=dF, acc=dA)

    # raw_json summary (optional)
    if args.raw_json and os.path.exists(args.raw_json):
        try:
            js = json.load(open(args.raw_json, "r", encoding="utf-8"))
            n = len(js) if isinstance(js, list) else 0
            print(f"\n[i] raw_json items: {n}")
            report["raw_json_items"] = n
        except Exception:
            pass

    # save report (auto-create dir)
    if args.export_report:
        os.makedirs(os.path.dirname(args.export_report), exist_ok=True)
        with open(args.export_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[✓] Saved metrics: {args.export_report}")

    print(f"\n[✓] Evaluated samples: {len(df):,}")

if __name__ == "__main__":
    main()
