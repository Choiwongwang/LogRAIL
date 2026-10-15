#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Stage-2 / final metrics helpers used by RAG_* and ablation summary."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ArrayLike = Union[Iterable[Any], np.ndarray]


def _as_int_array(y: ArrayLike) -> np.ndarray:
    arr = np.asarray(y).reshape(-1)
    # tolerate bool / float labels
    return arr.astype(int)


def _metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Return a flat metrics dict (keys are expanded into ablation CSVs)."""
    yt = _as_int_array(y_true)
    yp = _as_int_array(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"length mismatch: y_true={len(yt)} y_pred={len(yp)}")

    tn = fp = fn = tp = 0
    if len(yt) > 0:
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return {
        "n": int(len(yt)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float(accuracy_score(yt, yp)) if len(yt) else 0.0,
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
    }


def _fmt_block(title: str, m: Mapping[str, Any]) -> None:
    print(f"\n===== {title} =====")
    print(
        f"n={m.get('n', '?')}  "
        f"P={m.get('precision', 0):.4f}  "
        f"R={m.get('recall', 0):.4f}  "
        f"F1={m.get('f1', 0):.4f}  "
        f"Acc={m.get('accuracy', 0):.4f}"
    )
    print(
        f"TP={m.get('tp', '?')}  FP={m.get('fp', '?')}  "
        f"FN={m.get('fn', '?')}  TN={m.get('tn', '?')}"
    )


def main() -> None:
    """Tiny self-check: python eval_final.py"""
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    m = _metrics(y_true, y_pred)
    _fmt_block("self-check", m)


if __name__ == "__main__":
    main()
