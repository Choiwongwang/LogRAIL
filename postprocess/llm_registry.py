#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve Stage-2 LLM specs to local paths or HF ids.

Default for campus offline runs: local Qwen2.5-7B-Instruct.
Llama is reserved: download into MODELS_ROOT then pass --llm_model llama.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

MODELS_ROOT = Path(
    os.environ.get("LOGRAIL_MODELS", os.path.expanduser("~/lograil/models"))
).expanduser()

# short alias -> (local dirname, hf_id)
LLM_CATALOG: Dict[str, tuple[str, str]] = {
    "qwen": ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    "Qwen2.5-7B-Instruct": ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    "llama": ("Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
    "Meta-Llama-3-8B-Instruct": (
        "Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ),
}

DEFAULT_LLM_KEY = "qwen"

# Embedding models used by Stage-1 / Stage-2
EMBED_CATALOG: Dict[str, tuple[str, str]] = {
    "distilbert": (
        "distilbert-base-nli-mean-tokens",
        "sentence-transformers/distilbert-base-nli-mean-tokens",
    ),
    "distilbert-base-nli-mean-tokens": (
        "distilbert-base-nli-mean-tokens",
        "sentence-transformers/distilbert-base-nli-mean-tokens",
    ),
    "bge": ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
    "bge-small-en-v1.5": ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
    "BAAI/bge-small-en-v1.5": ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
}


def models_root() -> Path:
    return Path(
        os.environ.get("LOGRAIL_MODELS", str(MODELS_ROOT))
    ).expanduser()


def _prefer_local(dirname: str, hf_id: str, *, warn_missing: bool = False) -> str:
    local = models_root() / dirname
    if local.is_dir() and any(local.iterdir()):
        return str(local.resolve())
    if warn_missing:
        print(
            f"[w] local model not found: {local}  "
            f"(falling back to HF id `{hf_id}`; download later to enable offline)"
        )
    return hf_id


def resolve_llm_model(spec: Optional[str] = None) -> str:
    """Resolve registry key / path / HF id to a loadable model location."""
    raw = (spec or DEFAULT_LLM_KEY).strip()
    if not raw:
        raw = DEFAULT_LLM_KEY

    expanded = os.path.expanduser(raw)
    if os.path.isdir(expanded) and any(Path(expanded).iterdir()):
        return str(Path(expanded).resolve())

    if raw in LLM_CATALOG:
        dirname, hf_id = LLM_CATALOG[raw]
        warn = raw in ("llama", "Meta-Llama-3-8B-Instruct")
        return _prefer_local(dirname, hf_id, warn_missing=warn)

    # bare dirname under models root
    candidate = models_root() / raw
    if candidate.is_dir() and any(candidate.iterdir()):
        return str(candidate.resolve())

    return raw


def resolve_embed_model(spec: Optional[str] = None) -> str:
    raw = (spec or "bge-small-en-v1.5").strip()
    expanded = os.path.expanduser(raw)
    if os.path.isdir(expanded) and any(Path(expanded).iterdir()):
        return str(Path(expanded).resolve())
    if raw in EMBED_CATALOG:
        dirname, hf_id = EMBED_CATALOG[raw]
        return _prefer_local(dirname, hf_id, warn_missing=False)
    candidate = models_root() / raw
    if candidate.is_dir() and any(candidate.iterdir()):
        return str(candidate.resolve())
    return raw


def default_llm_model() -> str:
    return resolve_llm_model(DEFAULT_LLM_KEY)


def default_embed_model() -> str:
    return resolve_embed_model("bge-small-en-v1.5")


def default_distilbert() -> str:
    return resolve_embed_model("distilbert")
