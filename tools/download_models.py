#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download LogRAIL models into ~/lograil/models (or --out).

Default set (phase-0):
  - distilbert-base-nli-mean-tokens   (Stage-1 window embeddings)
  - BAAI/bge-small-en-v1.5            (Stage-2 VDB embeddings)
  - Qwen/Qwen2.5-7B-Instruct          (Stage-2 LLM; Llama can wait)

Mirrors (China-friendly):
  export HF_ENDPOINT=https://hf-mirror.com
  # or: --backend modelscope
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_OUT = Path(os.path.expanduser("~/lograil/models"))

# local_dirname -> (hf_id, modelscope_id_or_None)
MODEL_CATALOG: Dict[str, Tuple[str, str | None]] = {
    "distilbert-base-nli-mean-tokens": (
        "sentence-transformers/distilbert-base-nli-mean-tokens",
        "AI-ModelScope/distilbert-base-nli-mean-tokens",
    ),
    "bge-small-en-v1.5": (
        "BAAI/bge-small-en-v1.5",
        "Xorbits/bge-small-en-v1.5",
    ),
    "Qwen2.5-7B-Instruct": (
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ),
    "Meta-Llama-3-8B-Instruct": (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        None,  # usually gated; prefer HF with token if needed
    ),
}

PHASE0 = [
    "distilbert-base-nli-mean-tokens",
    "bge-small-en-v1.5",
    "Qwen2.5-7B-Instruct",
]


def _download_hf(repo_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    # download to a temp sibling then rename for atomicity
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(tmp),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def _download_modelscope(ms_id: str, dest: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    # modelscope writes into cache_dir/model_id; we force local_dir
    path = ms_snapshot_download(ms_id, local_dir=str(tmp))
    path = Path(path)
    if path.resolve() != tmp.resolve():
        # unexpected layout: copy contents
        tmp.mkdir(parents=True, exist_ok=True)
        for p in path.iterdir():
            target = tmp / p.name
            if p.is_dir():
                shutil.copytree(p, target, dirs_exist_ok=True)
            else:
                shutil.copy2(p, target)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def download_one(name: str, out_root: Path, backend: str) -> Path:
    if name not in MODEL_CATALOG:
        raise SystemExit(f"[x] unknown model key: {name}. known={list(MODEL_CATALOG)}")
    hf_id, ms_id = MODEL_CATALOG[name]
    dest = out_root / name
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] already exists: {dest}")
        return dest

    print(f"[↓] {name}  backend={backend}  -> {dest}")
    if backend == "modelscope":
        if not ms_id:
            raise SystemExit(f"[x] no modelscope id for {name}; use --backend hf")
        _download_modelscope(ms_id, dest)
    else:
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print(f"[i] HF_ENDPOINT -> {os.environ['HF_ENDPOINT']}")
        _download_hf(hf_id, dest)
    print(f"[✓] {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Download LogRAIL models")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--backend",
        choices=["hf", "modelscope"],
        default="hf",
        help="hf uses HF_ENDPOINT (default hf-mirror); modelscope uses ModelScope hub",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=PHASE0,
        help=f"keys from {list(MODEL_CATALOG)} (default: phase0 set)",
    )
    ap.add_argument("--list", action="store_true", help="list catalog and exit")
    args = ap.parse_args()

    if args.list:
        for k, (hf, ms) in MODEL_CATALOG.items():
            print(f"{k}\n  hf: {hf}\n  ms: {ms}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    ok: List[str] = []
    for name in args.models:
        try:
            download_one(name, args.out, args.backend)
            ok.append(name)
        except Exception as e:
            print(f"[!] failed {name}: {e}", file=sys.stderr)
            raise
    print(f"[done] {len(ok)} models under {args.out}")


if __name__ == "__main__":
    main()
