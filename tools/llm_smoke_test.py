#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal LLM load + generate smoke test (Qwen / Llama Instruct).

Example:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python tools/llm_smoke_test.py \
    --model ~/lograil/models/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from postprocess.llm_registry import DEFAULT_LLM_KEY, resolve_llm_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM smoke test")
    ap.add_argument(
        "--model",
        default=DEFAULT_LLM_KEY,
        help="registry key (qwen|llama), local dir, or HF id",
    )
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Reply with exactly: OK")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model_path = resolve_llm_model(args.model)
    print(f"[i] model={model_path}")
    print(f"[i] HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    print(f"[i] TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[✓] loaded in {time.time() - t0:.1f}s")

    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": args.prompt},
        ]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = args.prompt

    inputs = tok(text, return_tensors="pt")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t1 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    reply = tok.decode(gen, skip_special_tokens=True).strip()
    dt = time.time() - t1
    print(f"[✓] generate in {dt:.2f}s")
    print("--- reply ---")
    print(reply)
    print("-------------")
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[i] peak CUDA alloc: {mem:.2f} GiB")


if __name__ == "__main__":
    main()
