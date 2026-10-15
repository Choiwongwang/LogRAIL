#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Stage-2 config selection from δ=0.30 wide-run LLM labels.

Premise (experiment plan §1 item 9 + §7):
  - Greedy decoding => LLM label for a window is fixed given (mode, k, evidence).
  - Narrower δ or budget B only changes which windows *apply* that label;
    others fall back to Stage-1 (θ).
  - Different k need separate wide runs (evidence differs).

Example:
  python tools/offline_select_config.py \\
    --mode precision \\
    --pred_csv output/logformer_preds_val.csv \\
    --gt_csv output/gt_val.csv \\
    --theta 0.53 \\
    --wide_run 5:path/to/rag_normal.csv \\
    --wide_raw 5:path/to/rag_normal_raw.json \\
    --out_dir /tmp/offline_precision
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import eval_final as evalmod  # noqa: E402

DEFAULT_DELTAS = [0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_BUDGETS = [0.05, 0.10, 0.20, 1.0]
F1_TIE = 0.005  # 0.5 percentage points
CONSTRAINT = 0.90


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_wide_spec(items: Sequence[str]) -> Dict[int, Path]:
    """Parse repeated 'k:path' into {k: Path}."""
    out: Dict[int, Path] = {}
    for raw in items:
        if ":" not in raw:
            raise SystemExit(f"[x] expected k:path, got: {raw}")
        k_s, path_s = raw.split(":", 1)
        k = int(k_s)
        p = Path(os.path.expanduser(path_s))
        if not p.exists():
            raise SystemExit(f"[x] wide run file not found: {p}")
        out[k] = p
    return out


def _load_stage1(pred_csv: Path, theta: float) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    if "anomaly_pred" not in df.columns and "is_anomaly_pred" in df.columns:
        df = df.rename(columns={"is_anomaly_pred": "anomaly_pred"})
    need = {"window_id", "prob"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[x] pred_csv needs columns {need}, got {list(df.columns)}")
    df = df[["window_id", "prob"] + (["anomaly_pred"] if "anomaly_pred" in df.columns else [])].copy()
    df["window_id"] = df["window_id"].astype(int)
    df["prob"] = df["prob"].astype(float)
    df["stage1"] = (df["prob"] >= float(theta)).astype(int)
    return df


def _load_gt(gt_csv: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_csv)
    if "window_id" not in gt.columns or "label" not in gt.columns:
        raise SystemExit("[x] gt_csv needs window_id,label")
    gt = gt[["window_id", "label"]].copy()
    gt["window_id"] = gt["window_id"].astype(int)
    gt["label"] = gt["label"].astype(int)
    return gt


def _load_called_ids(
    k: int,
    rag_csv: Path,
    raw_map: Dict[int, Path],
    called_map: Dict[int, Path],
    flips_map: Dict[int, Path],
) -> Set[int]:
    if k in called_map:
        data = json.loads(called_map[k].read_text(encoding="utf-8"))
        if isinstance(data, dict) and "window_ids" in data:
            return {int(x) for x in data["window_ids"]}
        if isinstance(data, list):
            return {int(x) for x in data}
        raise SystemExit(f"[x] bad called_json for k={k}")

    if k in raw_map:
        raw = json.loads(raw_map[k].read_text(encoding="utf-8"))
        ids = set()
        for item in raw:
            if isinstance(item, dict) and "window_id" in item:
                # Abnormal raw may mark llm_called
                if item.get("llm_called") is False:
                    continue
                ids.add(int(item["window_id"]))
        if ids:
            return ids

    if k in flips_map:
        flips = pd.read_csv(flips_map[k])
        if "window_id" in flips.columns:
            # flips alone under-count (non-flip LLM calls missing); still better than nothing
            flip_ids = {int(x) for x in flips["window_id"].tolist()}
        else:
            flip_ids = set()
    else:
        flip_ids = set()

    # Default: all rows in rag out_csv are treated as having a reusable LLM label
    # (correct when wide run only wrote LLM-called / candidate rows, or full val
    #  where non-called rows equal stage1 — applying llm label == stage1 is fine).
    rag = pd.read_csv(rag_csv)
    all_ids = {int(x) for x in rag["window_id"].tolist()}
    if flip_ids and not raw_map and k not in called_map:
        # Prefer union: rag rows that appear in raw would be ideal; without raw,
        # use all rag rows (smoke cand20 case).
        return all_ids
    return all_ids


def _load_llm_labels(rag_csv: Path, called: Set[int]) -> Dict[int, int]:
    rag = pd.read_csv(rag_csv)
    if "anomaly" not in rag.columns:
        raise SystemExit(f"[x] rag csv missing anomaly column: {rag_csv}")
    out: Dict[int, int] = {}
    for _, row in rag.iterrows():
        wid = int(row["window_id"])
        if wid in called:
            out[wid] = int(row["anomaly"])
    return out


def _eligible_mask(
    mode: str,
    stage1: pd.Series,
    prob: pd.Series,
    theta: float,
    delta: float,
) -> pd.Series:
    lo_p = float(theta)
    hi_p = min(1.0, float(theta) + float(delta))
    lo_r = max(0.0, float(theta) - float(delta))
    hi_r = float(theta)
    if mode == "precision":
        return (stage1 == 1) & (prob >= lo_p) & (prob <= hi_p)
    if mode == "recall":
        return (stage1 == 0) & (prob >= lo_r) & (prob < hi_r)
    raise ValueError(mode)


def _apply_budget(
    eligible_idx: List[int],
    probs: pd.Series,
    theta: float,
    budget: float,
    m_total: int,
) -> List[int]:
    if budget >= 1.0 - 1e-12:
        return eligible_idx
    if budget <= 0:
        return []
    cap = int(math.floor(float(budget) * int(m_total)))
    if cap <= 0:
        return []
    ranked = sorted(eligible_idx, key=lambda i: abs(float(probs.iloc[i]) - float(theta)))
    return ranked[:cap]


def _simulate(
    base: pd.DataFrame,
    llm: Dict[int, int],
    mode: str,
    theta: float,
    delta: float,
    budget: float,
) -> Tuple[pd.Series, Dict[str, int]]:
    """Return final predictions aligned to base index + stats."""
    elig = _eligible_mask(mode, base["stage1"], base["prob"], theta, delta)
    elig_idx = [i for i, ok in enumerate(elig.tolist()) if ok]
    kept_idx = _apply_budget(elig_idx, base["prob"], theta, budget, len(base))
    kept_wids = {int(base.iloc[i]["window_id"]) for i in kept_idx}

    finals = []
    n_llm = 0
    n_missing_llm = 0
    for _, row in base.iterrows():
        wid = int(row["window_id"])
        s1 = int(row["stage1"])
        if wid in kept_wids:
            if wid in llm:
                finals.append(int(llm[wid]))
                n_llm += 1
            else:
                finals.append(s1)
                n_missing_llm += 1
        else:
            finals.append(s1)
    stats = {
        "n_cand": len(elig_idx),
        "n_after_budget": len(kept_idx),
        "n_llm_applied": n_llm,
        "n_missing_llm": n_missing_llm,
    }
    return pd.Series(finals, index=base.index, dtype=int), stats


def _flip_stats(y_true: pd.Series, stage1: pd.Series, final: pd.Series) -> Dict[str, int]:
    flip = final != stage1
    benefit = int((flip & (final == y_true)).sum())
    harm = int((flip & (final != y_true)).sum())
    return {
        "flip_count": int(flip.sum()),
        "flip_benefit": benefit,
        "flip_harm": harm,
        "net_gain": benefit - harm,
    }


def _band_endpoints(mode: str, theta: float, delta: float) -> Dict[str, float]:
    if mode == "precision":
        return {
            "prob_lo": float(theta),
            "prob_hi": float(min(1.0, theta + delta)),
        }
    return {
        "prob_lo": float(max(0.0, theta - delta)),
        "prob_hi": float(theta),
    }


def run_grid(
    mode: str,
    base: pd.DataFrame,
    gt: pd.DataFrame,
    llm_by_k: Dict[int, Dict[int, int]],
    theta: float,
    deltas: List[float],
    budgets: List[float],
) -> pd.DataFrame:
    merged = base.merge(gt, on="window_id", how="inner")
    if len(merged) == 0:
        raise SystemExit("[x] no overlap between pred_csv and gt_csv")
    coverage = {
        k: len(set(llm_by_k[k]) & set(merged["window_id"])) for k in llm_by_k
    }

    rows = []
    for k, llm in sorted(llm_by_k.items()):
        for delta in deltas:
            for budget in budgets:
                final, stats = _simulate(merged, llm, mode, theta, delta, budget)
                m = evalmod._metrics(merged["label"], final)
                flips = _flip_stats(merged["label"], merged["stage1"], final)
                band = _band_endpoints(mode, theta, delta)
                # stage1 baseline metrics once reusable
                rows.append(
                    {
                        "mode": mode,
                        "k": int(k),
                        "delta": float(delta),
                        "budget": float(budget),
                        "theta": float(theta),
                        "prob_lo": band["prob_lo"],
                        "prob_hi": band["prob_hi"],
                        "llm_label_coverage": int(coverage[k]),
                        "n_eval": int(len(merged)),
                        **stats,
                        **m,
                        **flips,
                        "constraint_precision_ok": int(m["precision"] >= CONSTRAINT),
                        "constraint_recall_ok": int(m["recall"] >= CONSTRAINT),
                    }
                )
    return pd.DataFrame(rows)


def _stage1_metrics(base: pd.DataFrame, gt: pd.DataFrame) -> Dict[str, float]:
    merged = base.merge(gt, on="window_id", how="inner")
    return evalmod._metrics(merged["label"], merged["stage1"])


def select_config(grid: pd.DataFrame, mode: str) -> Dict[str, Any]:
    """Apply experiment-plan §7 selection rules."""
    unlim = grid[grid["budget"] >= 1.0 - 1e-12].copy()
    if unlim.empty:
        raise SystemExit("[x] grid has no unlimited-budget (B=1.0) rows")

    if mode == "precision":
        constrained = unlim[unlim["precision"] >= CONSTRAINT]
        metric_name = "precision"
    else:
        constrained = unlim[unlim["recall"] >= CONSTRAINT]
        metric_name = "recall"

    constraint_met = len(constrained) > 0
    pool = constrained if constraint_met else unlim

    best_f1 = float(pool["f1"].max())
    tied = pool[pool["f1"] >= best_f1 - F1_TIE].copy()
    # prefer smaller k, then smaller delta
    tied = tied.sort_values(["k", "delta", "f1"], ascending=[True, True, False])
    pick_dk = tied.iloc[0]
    k_sel = int(pick_dk["k"])
    delta_sel = float(pick_dk["delta"])
    f1_unlim = float(pick_dk["f1"])

    sub = grid[(grid["k"] == k_sel) & (grid["delta"] == delta_sel)].copy()
    # budgets that keep F1 within 0.5pp of unlimited
    ok_b = sub[sub["f1"] >= f1_unlim - F1_TIE].copy()
    ok_b = ok_b.sort_values(["budget", "f1"], ascending=[True, False])
    pick_b = ok_b.iloc[0]
    budget_sel = float(pick_b["budget"])

    selected_row = pick_b.to_dict()
    net_gain = int(selected_row.get("net_gain", 0))
    notes = []
    if not constraint_met:
        notes.append(
            f"No (δ,k) met {metric_name}>={CONSTRAINT:.2f} at B=1; fell back to max F1."
        )
    notes.append(
        f"Among F1 within {F1_TIE} of best unlimited, chose smallest k then smallest δ."
    )
    notes.append(
        f"Among B with F1 drop <= {F1_TIE} vs B=1, chose smallest B={budget_sel}."
    )
    if net_gain <= 0:
        notes.append("net_gain_warning: net_gain <= 0 (plan fallback condition).")

    band = _band_endpoints(mode, float(selected_row["theta"]), delta_sel)
    return {
        "mode": mode,
        "theta": float(selected_row["theta"]),
        "delta": delta_sel,
        "k": k_sel,
        "budget": budget_sel,
        "budget_unlimited": budget_sel >= 1.0 - 1e-12,
        "prob_lo": band["prob_lo"],
        "prob_hi": band["prob_hi"],
        "constraint_metric": metric_name,
        "constraint_threshold": CONSTRAINT,
        "constraint_met": bool(constraint_met),
        "f1_tie_tol": F1_TIE,
        "metrics": {
            "precision": float(selected_row["precision"]),
            "recall": float(selected_row["recall"]),
            "f1": float(selected_row["f1"]),
            "accuracy": float(selected_row["accuracy"]),
            "n": int(selected_row["n"]),
            "tp": int(selected_row["tp"]),
            "fp": int(selected_row["fp"]),
            "fn": int(selected_row["fn"]),
            "tn": int(selected_row["tn"]),
            "n_cand": int(selected_row["n_cand"]),
            "n_after_budget": int(selected_row["n_after_budget"]),
            "n_llm_applied": int(selected_row["n_llm_applied"]),
            "n_missing_llm": int(selected_row["n_missing_llm"]),
            "flip_count": int(selected_row["flip_count"]),
            "flip_benefit": int(selected_row["flip_benefit"]),
            "flip_harm": int(selected_row["flip_harm"]),
            "net_gain": net_gain,
            "llm_label_coverage": int(selected_row["llm_label_coverage"]),
            "n_eval": int(selected_row["n_eval"]),
        },
        "net_gain_warning": bool(net_gain <= 0),
        "selection_notes": notes,
        "cli_hints": {
            "precision": {
                "call_policy": "pred1_boundary",
                "base_threshold": float(selected_row["theta"]),
                "prob_lo": band["prob_lo"],
                "prob_hi_b": band["prob_hi"],
                "evidence_k": k_sel,
                "budget": None if budget_sel >= 1.0 - 1e-12 else budget_sel,
            },
            "recall": {
                "call_policy": "boundary",
                "prob0_min": band["prob_lo"],
                "prob0_max": band["prob_hi"],
                "evidence_k": k_sel,
                "top_k": 1,
                "budget": None if budget_sel >= 1.0 - 1e-12 else budget_sel,
            },
        }[mode],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline select δ/k/B from δ=0.30 wide-run LLM labels"
    )
    ap.add_argument("--mode", choices=["precision", "recall"], required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument(
        "--wide_run",
        action="append",
        default=[],
        help="k:rag_out.csv (repeatable)",
    )
    ap.add_argument(
        "--wide_raw",
        action="append",
        default=[],
        help="k:raw.json listing called window_ids (preferred)",
    )
    ap.add_argument(
        "--wide_flips",
        action="append",
        default=[],
        help="k:flips.csv (optional hint)",
    )
    ap.add_argument(
        "--called_json",
        action="append",
        default=[],
        help="k:json with list or {window_ids:[...]}",
    )
    ap.add_argument(
        "--deltas",
        default=",".join(str(x) for x in DEFAULT_DELTAS),
    )
    ap.add_argument(
        "--budgets",
        default=",".join(str(x) for x in DEFAULT_BUDGETS),
        help="use 1.0 for unlimited",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if not args.wide_run:
        raise SystemExit("[x] provide at least one --wide_run k:path")
    wide = _parse_wide_spec(args.wide_run)
    raw_map = _parse_wide_spec(args.wide_raw) if args.wide_raw else {}
    flips_map = _parse_wide_spec(args.wide_flips) if args.wide_flips else {}
    called_map = _parse_wide_spec(args.called_json) if args.called_json else {}

    deltas = _parse_float_list(args.deltas)
    budgets = _parse_float_list(args.budgets)
    if 1.0 not in budgets and not any(b >= 1.0 - 1e-12 for b in budgets):
        budgets.append(1.0)

    base = _load_stage1(Path(args.pred_csv), args.theta)
    gt = _load_gt(Path(args.gt_csv))
    s1m = _stage1_metrics(base, gt)

    llm_by_k: Dict[int, Dict[int, int]] = {}
    called_stats = {}
    for k, rag_path in sorted(wide.items()):
        called = _load_called_ids(k, rag_path, raw_map, called_map, flips_map)
        llm_by_k[k] = _load_llm_labels(rag_path, called)
        called_stats[k] = {
            "n_called": len(called),
            "n_llm_labels": len(llm_by_k[k]),
            "rag_csv": str(rag_path),
        }

    grid = run_grid(args.mode, base, gt, llm_by_k, args.theta, deltas, budgets)
    selected = select_config(grid, args.mode)
    selected["stage1_metrics"] = s1m
    selected["wide_runs"] = called_stats
    # Coverage vs δ=0.30 band candidates (not vs all val rows).
    # Formal wide runs only label band windows; warn if labels < that band size.
    n_eval = int(grid["n_eval"].iloc[0])
    max_cov = int(grid["llm_label_coverage"].max())
    band30 = grid[grid["delta"] >= 0.30 - 1e-12]
    n_band30 = int(band30["n_cand"].max()) if len(band30) else 0
    selected["llm_label_coverage"] = max_cov
    selected["n_eval"] = n_eval
    selected["n_cand_delta030"] = n_band30
    if n_band30 > 0 and max_cov < n_band30:
        selected["coverage_warning"] = (
            f"LLM labels cover {max_cov}/{n_band30} δ=0.30 band candidates "
            f"(eval n={n_eval}); wide run incomplete — do not treat as formal freeze."
        )
        selected["selection_notes"].append(selected["coverage_warning"])

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "grid_metrics.csv"
    grid.to_csv(grid_path, index=False)
    cfg_path = out_dir / "selected_config.json"
    cfg_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = out_dir / "README_offline.txt"
    readme.write_text(
        "Offline selection from greedy δ=0.30 wide-run labels.\n"
        "Do not reuse labels across different evidence_k.\n"
        "Use same theta and split as the wide run. Formal freeze requires full-val coverage.\n",
        encoding="utf-8",
    )

    print(f"[✓] wrote {grid_path}  rows={len(grid)}")
    print(f"[✓] wrote {cfg_path}")
    print(
        f"[✓] selected mode={args.mode} δ={selected['delta']} k={selected['k']} "
        f"B={selected['budget']} F1={selected['metrics']['f1']:.4f} "
        f"constraint_met={selected['constraint_met']} net_gain={selected['metrics']['net_gain']}"
    )
    if selected.get("coverage_warning"):
        print(f"[w] {selected['coverage_warning']}")


if __name__ == "__main__":
    main()
