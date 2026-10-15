#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Stage-2 runtime helpers: config dump, budget, resume, LLM cost log."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd


def dump_effective_config(cfg: Dict[str, Any], out_csv: str) -> str:
    path = os.path.join(os.path.dirname(out_csv) or ".", "effective_config.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("[i] Effective config:")
    for k in sorted(cfg.keys()):
        print(f"    {k}: {cfg[k]}")
    print(f"[✓] saved {path}")
    return path


def cost_log_path(out_csv: str) -> str:
    return os.path.join(os.path.dirname(out_csv) or ".", "llm_cost.jsonl")


def append_cost(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_ids(out_csv: str) -> Set[int]:
    if not out_csv or not os.path.exists(out_csv):
        return set()
    try:
        df = pd.read_csv(out_csv)
        if "window_id" not in df.columns:
            return set()
        return set(int(x) for x in df["window_id"].tolist())
    except Exception:
        return set()


def apply_budget(
    pred_df: pd.DataFrame,
    eligible_mask: List[bool],
    *,
    budget: Optional[float],
    theta: float,
    prob_col: str = "prob",
) -> List[bool]:
    """Keep at most floor(B*M) eligible rows, nearest to theta first."""
    if budget is None:
        return eligible_mask
    b = float(budget)
    if b <= 0:
        return [False] * len(eligible_mask)
    if b >= 1.0:
        return eligible_mask

    m = len(pred_df)
    cap = int(m * b)
    idxs = [i for i, ok in enumerate(eligible_mask) if ok]
    if len(idxs) <= cap:
        return eligible_mask

    probs = pred_df[prob_col].astype(float).tolist()
    idxs.sort(key=lambda i: abs(float(probs[i]) - float(theta)))
    keep = set(idxs[:cap])
    return [bool(eligible_mask[i] and i in keep) for i in range(m)]


class TimedLLM:
    """Wrap a generate callable and append cost records."""

    def __init__(self, model_name: str, cost_path: str, call_kind: str = "decision"):
        self.model_name = model_name
        self.cost_path = cost_path
        self.call_kind = call_kind

    def run(self, gen_fn, *, window_id: Optional[int] = None) -> Any:
        t0 = time.time()
        err = None
        out = None
        parse_ok = None
        try:
            out = gen_fn()
            if isinstance(out, dict):
                parse_ok = True
            elif out is None:
                parse_ok = False
            else:
                parse_ok = True
        except Exception as e:
            err = str(e)
            parse_ok = False
            raise
        finally:
            append_cost(
                self.cost_path,
                {
                    "ts": time.time(),
                    "window_id": window_id,
                    "kind": self.call_kind,
                    "model": self.model_name,
                    "latency_sec": round(time.time() - t0, 4),
                    "parse_ok": parse_ok,
                    "error": err,
                },
            )
        return out
