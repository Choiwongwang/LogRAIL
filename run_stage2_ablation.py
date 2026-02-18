#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Stage-2 ablations sequentially without modifying existing scripts.

This script runs sequential ablations using the original two paths:
  - precision path: postprocess/RAG_Normal.py
  - recall path:    postprocess/RAG_Abnormal.py

Ablations:
  1) Precision bands (boundary policy): 3 runs
  2) Recall bands (boundary policy): 3 runs
  3) Call-policy ablation (fixed bands): 3 runs
  4) Evidence-k ablation (fixed bands): 4 runs per mode
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class Paths:
    pred_csv: str
    window_csv: str
    normal_csv: str
    anomaly_csv: str
    gt_csv: str
    chroma_base: str
    out_base: str
    stage2_normal_script: str
    stage2_abnormal_script: str


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_paths() -> Paths:
    root = _repo_root()
    return Paths(
        pred_csv=os.path.join(root, "output", "logformer_preds_val.csv"),
        window_csv=os.path.join(root, "output", "window_repr_by_pred_info_val.csv"),
        normal_csv=os.path.join(root, "dataset", "normal_templates_clean.csv"),
        anomaly_csv=os.path.join(root, "dataset", "anomaly_templates_clean.csv"),
        gt_csv=os.path.join(root, "output", "gt_val.csv"),
        chroma_base=os.path.join(root, "rag_db", "ablation_val"),
        out_base=os.path.join(root, "output", "ablation_val"),
        stage2_normal_script=os.path.join(root, "postprocess", "RAG_Normal.py"),
        stage2_abnormal_script=os.path.join(root, "postprocess", "RAG_Abnormal.py"),
    )


def _ensure_exists(label: str, path: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"[x] Missing {label}: {path}")


def _run(cmd: list[str]) -> None:
    print("\n[i] Running:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _has_result(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "rag.csv"))


def main() -> None:
    d = _default_paths()

    ap = argparse.ArgumentParser("Run Stage-2 ablation experiments")
    ap.add_argument("--pred_csv", default=d.pred_csv)
    ap.add_argument("--window_csv", default=d.window_csv)
    ap.add_argument("--normal_csv", default=d.normal_csv)
    ap.add_argument("--anomaly_csv", default=d.anomaly_csv)
    ap.add_argument("--gt_csv", default=d.gt_csv)
    ap.add_argument("--chroma_base", default=d.chroma_base)
    ap.add_argument("--out_base", default=d.out_base)
    ap.add_argument("--stage2_normal_script", default=d.stage2_normal_script)
    ap.add_argument("--stage2_abnormal_script", default=d.stage2_abnormal_script)
    ap.add_argument("--sleep_sec", type=int, default=300,
                    help="Sleep seconds between ablation runs (default: 300)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip runs if out_dir already has rag.csv")
    ap.add_argument(
        "--only_recall",
        action="store_true",
    )
    ap.add_argument(
        "--only_precision",
        action="store_true",
    )
    args = ap.parse_args()

    # Input checks
    _ensure_exists("pred_csv", args.pred_csv)
    _ensure_exists("window_csv", args.window_csv)
    _ensure_exists("normal_csv", args.normal_csv)
    _ensure_exists("anomaly_csv", args.anomaly_csv)
    _ensure_exists("gt_csv", args.gt_csv)
    _ensure_exists("stage2_normal_script", args.stage2_normal_script)
    _ensure_exists("stage2_abnormal_script", args.stage2_abnormal_script)

    os.makedirs(args.chroma_base, exist_ok=True)
    os.makedirs(args.out_base, exist_ok=True)

    py = sys.executable

    if args.only_precision and args.only_recall:
        raise SystemExit("[x] Use only one of --only_precision or --only_recall.")

    # 1) Precision bands (boundary policy, pred==1 & lo<=p<hi)
    precision_bands = [
        (0.67, 0.75, "prec_b067_075"),
        (0.67, 0.80, "prec_b067_080"),
        (0.67, 0.85, "prec_b067_085"),
        (0.67, 0.90, "prec_b067_090"),
        (0.67, 0.95, "prec_b067_095"),
    ]
    if not args.only_recall:
        for lo, hi, tag in precision_bands:
            out_dir = os.path.join(args.out_base, tag)
            chroma_dir = os.path.join(args.chroma_base, f"{tag}_prec")
            os.makedirs(out_dir, exist_ok=True)
            if args.skip_existing and _has_result(out_dir):
                print(f"[i] skip (exists): {out_dir}")
                continue
            cmd = [
                py,
                args.stage2_normal_script,
                "--call_policy", "pred1_boundary",
                "--prob_lo", f"{lo:.2f}",
                "--prob_hi_b", f"{hi:.2f}",
                "--pred_csv", args.pred_csv,
                "--window_csv", args.window_csv,
                "--normal_csv", args.normal_csv,
                "--chroma_dir", chroma_dir,
                "--eval_gt_csv", args.gt_csv,
                "--out_csv", os.path.join(out_dir, "rag.csv"),
                "--raw_json", os.path.join(out_dir, "rag_raw.json"),
                "--flips_csv", os.path.join(out_dir, "rag_flips.csv"),
            ]
            _run(cmd)
            time.sleep(args.sleep_sec)

    if args.only_precision:
        print("\n[i] only_precision set: skipping recall/call-policy/evidence-k ablations.")
        return

    if args.only_recall:
        print("\n[i] only_recall set: skipping call-policy/evidence-k ablations.")
        # continue to recall bands only

    # 2) Recall bands (boundary policy, pred==0 & lo<=p<hi)
    recall_bands = [
        (0.10, 0.67, "rec_b010_067"),
        (0.20, 0.67, "rec_b020_067"),
        (0.30, 0.67, "rec_b030_067"),
        (0.40, 0.67, "rec_b040_067"),
        (0.50, 0.67, "rec_b050_067"),
        (0.60, 0.67, "rec_b060_067"),
    ]
    for lo, hi, tag in recall_bands:
        out_dir = os.path.join(args.out_base, tag)
        chroma_dir = os.path.join(args.chroma_base, f"{tag}_rec")
        os.makedirs(out_dir, exist_ok=True)
        if args.skip_existing and _has_result(out_dir):
            print(f"[i] skip (exists): {out_dir}")
            continue
        cmd = [
            py,
            args.stage2_abnormal_script,
            "--prob0_min", f"{lo:.2f}",
            "--prob0_max", f"{hi:.2f}",
            "--pred_csv", args.pred_csv,
            "--window_csv", args.window_csv,
            "--anomaly_csv", args.anomaly_csv,
            "--chroma_dir", chroma_dir,
            "--eval_gt_csv", args.gt_csv,
            "--out_csv", os.path.join(out_dir, "rag.csv"),
            "--raw_json", os.path.join(out_dir, "rag_raw.json"),
            "--flips_csv", os.path.join(out_dir, "rag_flips.csv"),
        ]
        _run(cmd)
        time.sleep(args.sleep_sec)

    # 3) Call-policy ablation (fixed bands)
    # Uses the best-performing bands: precision [0.67,0.95), recall [0.60,0.67)
    call_policies = [
        ("boundary", "pred1_boundary", "boundary"),
        ("pred_only", "pred1", "pred0"),
    ]
    for tag, prec_policy, rec_policy in call_policies:
        # precision side
        out_dir = os.path.join(args.out_base, f"call_{tag}_prec")
        chroma_dir = os.path.join(args.chroma_base, f"call_{tag}_prec")
        os.makedirs(out_dir, exist_ok=True)
        if args.skip_existing and _has_result(out_dir):
            print(f"[i] skip (exists): {out_dir}")
        else:
            cmd_prec = [
                py,
                args.stage2_normal_script,
                "--call_policy", prec_policy,
                "--prob_lo", "0.67",
                "--prob_hi_b", "0.95",
                "--pred_csv", args.pred_csv,
                "--window_csv", args.window_csv,
                "--normal_csv", args.normal_csv,
                "--chroma_dir", chroma_dir,
                "--eval_gt_csv", args.gt_csv,
                "--out_csv", os.path.join(out_dir, "rag.csv"),
                "--raw_json", os.path.join(out_dir, "rag_raw.json"),
                "--flips_csv", os.path.join(out_dir, "rag_flips.csv"),
            ]
            _run(cmd_prec)
            time.sleep(args.sleep_sec)

        # recall side
        out_dir = os.path.join(args.out_base, f"call_{tag}_rec")
        chroma_dir = os.path.join(args.chroma_base, f"call_{tag}_rec")
        os.makedirs(out_dir, exist_ok=True)
        if args.skip_existing and _has_result(out_dir):
            print(f"[i] skip (exists): {out_dir}")
        else:
            cmd_rec = [
                py,
                args.stage2_abnormal_script,
                "--call_policy", rec_policy,
                "--prob0_min", "0.60",
                "--prob0_max", "0.67",
                "--pred_csv", args.pred_csv,
                "--window_csv", args.window_csv,
                "--anomaly_csv", args.anomaly_csv,
                "--chroma_dir", chroma_dir,
                "--eval_gt_csv", args.gt_csv,
                "--out_csv", os.path.join(out_dir, "rag.csv"),
                "--raw_json", os.path.join(out_dir, "rag_raw.json"),
                "--flips_csv", os.path.join(out_dir, "rag_flips.csv"),
            ]
            _run(cmd_rec)
            time.sleep(args.sleep_sec)

    # 4) Evidence-k ablation (fixed bands, boundary policies)
    evidence_ks = [1, 3, 5, 7]
    for k in evidence_ks:
        # precision side
        tag = f"evidence_k{k}_prec"
        out_dir = os.path.join(args.out_base, tag)
        chroma_dir = os.path.join(args.chroma_base, f"{tag}_prec")
        os.makedirs(out_dir, exist_ok=True)
        if args.skip_existing and _has_result(out_dir):
            print(f"[i] skip (exists): {out_dir}")
        else:
            cmd_prec = [
                py,
                args.stage2_normal_script,
                "--call_policy", "pred1_boundary",
                "--prob_lo", "0.67",
                "--prob_hi_b", "0.95",
                "--evidence_k", str(k),
                "--pred_csv", args.pred_csv,
                "--window_csv", args.window_csv,
                "--normal_csv", args.normal_csv,
                "--chroma_dir", chroma_dir,
                "--eval_gt_csv", args.gt_csv,
                "--out_csv", os.path.join(out_dir, "rag.csv"),
                "--raw_json", os.path.join(out_dir, "rag_raw.json"),
                "--flips_csv", os.path.join(out_dir, "rag_flips.csv"),
            ]
            _run(cmd_prec)
            time.sleep(args.sleep_sec)

        # recall side
        tag = f"evidence_k{k}_rec"
        out_dir = os.path.join(args.out_base, tag)
        chroma_dir = os.path.join(args.chroma_base, f"{tag}_rec")
        os.makedirs(out_dir, exist_ok=True)
        if args.skip_existing and _has_result(out_dir):
            print(f"[i] skip (exists): {out_dir}")
        else:
            cmd_rec = [
                py,
                args.stage2_abnormal_script,
                "--prob0_min", "0.60",
                "--prob0_max", "0.67",
                "--evidence_k", str(k),
                "--pred_csv", args.pred_csv,
                "--window_csv", args.window_csv,
                "--anomaly_csv", args.anomaly_csv,
                "--chroma_dir", chroma_dir,
                "--eval_gt_csv", args.gt_csv,
                "--out_csv", os.path.join(out_dir, "rag.csv"),
                "--raw_json", os.path.join(out_dir, "rag_raw.json"),
                "--flips_csv", os.path.join(out_dir, "rag_flips.csv"),
            ]
            _run(cmd_rec)
            time.sleep(args.sleep_sec)

    print("\n[✓] All ablation runs completed.")
    print(f"[i] Outputs under: {args.out_base}")


if __name__ == "__main__":
    main()
