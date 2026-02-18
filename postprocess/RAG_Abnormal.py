#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG_v3_recall_anomonly_recallboost.py — recall-focused post-process (anomaly VDB only)
- Keep pred==1 (anomaly) as-is (no FP correction)
- For pred==0 (normal) only: query anomaly VDB + LLM to promote 0→1
- Enhancements:
  * relaxed gates: looser prob/sim thresholds
  * top-k=10
- Outputs: final CSV (4 columns), raw JSON, flips CSV
- Evaluation (optional): slide-style block with confusion matrix, no JSON report
"""

from __future__ import annotations
import os, sys, re, gc, json, argparse, types
from typing import List, Tuple, Optional

import pandas as pd
from tqdm import tqdm
from packaging.version import parse as vparse

# ── runtime patches (posthog, sqlite, chroma telemetry) ──
posthog_stub = types.ModuleType("posthog")
posthog_stub.Client = lambda *a, **k: None
posthog_stub.capture = lambda *a, **k: None
sys.modules["posthog"] = posthog_stub
try:
    import pysqlite3.dbapi2 as sqlite3  # noqa
    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass
os.environ["CHROMA_TELEMETRY"] = "FALSE"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.embeddings   import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
IS_NEW_CHROMA = vparse(chromadb.__version__) >= vparse("0.5.0")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import importlib
_pf_mod = None
for name in ("postprocess.prompts", "postprocess.prompts_Normal", "prompts_Normal"):
    try:
        _pf_mod = importlib.import_module(name)
        break
    except Exception:
        continue
if _pf_mod is None:
    raise ImportError("Failed to import PROMPT_FINAL_REASON from prompts")
PROMPT_FINAL_REASON = getattr(_pf_mod, "PROMPT_FINAL_REASON")

# eval/metrics helper (keep consistent with RAG_Normal)
import eval_final as evalmod

# ───────── Defaults (recall boost) ─────────
DEFAULTS = {
    # Inputs/DB
    "pred_csv":   "output/logformer_preds_test.csv",
    "window_csv": "output/window_repr_by_pred_info_anom.csv",
    "anomaly_csv":"dataset/anomaly_templates_clean.csv",
    "chroma_dir": "rag_db/aosp_bge_v3_recall_anomonly_recallboost",

    # Models/embeddings
    "llm_model":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "embed_model": "BAAI/bge-small-en-v1.5",
    "normalize_embeddings": True,
    "top_k": 5,
    "evidence_k": 5,
    "pad_token": "<PAD>",
    "device": "cuda",

    # pred==0 recovery policy/thresholds (relaxed)
    "prob0_min": 0.40,
    "prob0_max": 0.67,
    # call policy for recall path
    # boundary: pred==0 & prob within [prob0_min, prob0_max)
    # pred0:    pred==0 (ignore prob)
    # all:      all windows (ignore pred/prob)
    "call_policy": "boundary",
    "anom_sim_th": 0.0,      # disable min similarity guard
    "anom_high_th": 0.0,     # disable high-confidence guard

    # Prompt mode (default strict)
    "prompt_mode": "strict",  # strict | moderate

    # Outputs
    "out_csv":   "output/rag_result_v3_recall_anomonly_recallboost.csv",
    "raw_json":  "output/rag_raw_v3_recall_anomonly_recallboost.json",
    "flips_csv": "output/rag_flips_v3_recall_anomonly_recallboost.csv",

    # ===== Evaluation =====
    # Default to test-only GT evaluation (avoid train+test mixing)
    "eval_gt_csv": "output/gt_test.csv",
    "eval_label_col": "label",
    "export_report": None,
    "skip_eval": False,
}

# ============ Utilities ============

RE_CODEJSON = re.compile(r"```json\s*({[\s\S]*?})\s*```", re.I)
RE_ANYJSON  = re.compile(r"({[\s\S]*})")

def _extract_json(txt: str):
    if not isinstance(txt, str):
        return None
    m = RE_CODEJSON.search(txt) or RE_ANYJSON.search(txt)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except Exception:
        return None

def _read_anomaly(js: dict) -> int:
    """Read anomaly from LLM JSON. Prefer anomaly, fallback to is_anomaly."""
    if not isinstance(js, dict):
        return 0
    if "anomaly" in js:
        try: return int(js.get("anomaly", 0))
        except Exception: return 0
    if "is_anomaly" in js:
        try: return int(js.get("is_anomaly", 0))
        except Exception: return 0
    return 0

def _load_window_templates_map(window_csv: Optional[str]) -> dict[int, List[str]]:
    """
    Load window templates mapping from CSV.
    Supports:
      - json: window_id, templates_json
      - wide: window_id, tpl_0..tpl_n
      - long: window_id, pos, EventTemplate
      - single: window_id, EventTemplate
    """
    if not window_csv or not os.path.exists(window_csv):
        return {}
    df = pd.read_csv(window_csv)
    out: dict[int, List[str]] = {}

    if "templates_json" in df.columns:
        for _, r in df.iterrows():
            wid = int(r["window_id"])
            raw = r["templates_json"]
            if pd.isna(raw):
                out[wid] = []
                continue
            try:
                lst = json.loads(raw)
                out[wid] = [str(x) for x in lst] if isinstance(lst, list) else []
            except Exception:
                out[wid] = []
        return out

    tpl_cols = [c for c in df.columns if c.startswith("tpl_")]
    if tpl_cols:
        tpl_cols = ["window_id"] + tpl_cols
        for _, r in df[tpl_cols].iterrows():
            wid = int(r["window_id"])
            out[wid] = [str(r[c]) for c in tpl_cols[1:]]
        return out

    if "pos" in df.columns and "EventTemplate" in df.columns:
        df2 = df.sort_values(["window_id", "pos"])
        for wid, g in df2.groupby("window_id"):
            out[int(wid)] = [str(x) for x in g["EventTemplate"].tolist()]
        return out

    return {}

def _minify_reason(s: str, n_words: int = 12) -> str:
    if not s: return "unknown"
    w = str(s).strip().split()
    return " ".join(w[:n_words]) if w else "unknown"

def _dist_to_sim(dist: float) -> float:
    try:
        d = float(dist)
    except Exception:
        return 0.0
    if d < 0: d = -d
    return 1.0/(1.0+d)

# ============ prompts_FN only (no fallback, constants only) ============

def _resolve_prompt_builder(prompt_mode: str):
    """
    Use **string constants** from prompts_Abnormal (preferred) or prompts_FN (fallback).
    Required constants:
      - PROMPT_RECALL_STRICT
      - PROMPT_RECALL_MODERATE
    """
    import importlib

    last_err = None
    mod = None
    tried = []
    for name in ("postprocess.prompts_Abnormal", "prompts_Abnormal", "postprocess.prompts_FN", "prompts_FN"):
        try:
            m = importlib.import_module(name)
            mod = m
            print(f"[prompt] loaded module: {name} (file={getattr(m, '__file__', 'NA')})")
            break
        except Exception as e:
            tried.append(f"{name}: {e.__class__.__name__}: {e}")
            last_err = e

    if mod is None:
        detail = "\n".join(tried)
        raise ImportError(
            "[prompts] recall prompt module not found.\n"
            f"tried:\n{detail}"
        ) from last_err

    mode = str(prompt_mode).strip().lower()
    if mode not in {"strict","moderate"}:
        mode = "strict"

    const_name = "PROMPT_RECALL_STRICT" if mode == "strict" else "PROMPT_RECALL_MODERATE"
    if not hasattr(mod, const_name):
        keys = dir(mod)
        sample = ", ".join(k for k in keys if k.isupper())[:300]
        raise AttributeError(
            "[prompts_FN] required prompt constant not found.\n"
            f"  - required: {const_name}\n"
            f"  - module: {getattr(mod, '__file__', 'NA')}\n"
            f"  - uppercase symbols (sample): {sample}"
        )

    tmpl: str = getattr(mod, const_name)
    if not isinstance(tmpl, str):
        raise TypeError(f"[prompts_FN] {const_name} must be a string (got {type(tmpl)})")

    def _build(tok, top_k: int, pairs_block: str) -> str:
        filled = tmpl.format(top_k=top_k, pairs_block=pairs_block)
        messages = [{"role": "user", "content": filled}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"[prompt] mode={mode} -> using {const_name} from {getattr(mod, '__file__', 'NA')}")
    return _build

# ============ RAG core (anomaly VDB only) ============

class RAGPostProcessorV3RecallBoost:
    def __init__(
        self,
        llm_model: str,
        embed_model: str,
        chroma_dir: str,
        top_k: int = 5,
        evidence_k: int = 5,
        pad_token: str = "<PAD>",
        device: str = "cuda",
        normalize_embeddings: bool = True,
        # pred0 policy/thresholds (relaxed)
        prob0_min: float = 0.30,
        prob0_max: float = 0.80,
        anom_sim_th: float = 0.93,
        anom_high_th: float = 0.980,
        # prompts
        prompt_mode: str = "strict",
        # call policy
        call_policy: str = "boundary",
    ):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.chroma_dir = chroma_dir
        self.top_k = int(top_k)
        self.evidence_k = int(evidence_k)
        self.pad_token = str(pad_token)
        self.device = device

        self.prob0_min    = float(prob0_min)
        self.prob0_max    = float(prob0_max)
        self.call_policy  = str(call_policy)
        self.anom_sim_th  = float(anom_sim_th)
        self.anom_high_th = float(anom_high_th)

        # Embeddings
        self.embed = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )

        # LLM
        self.tok = AutoTokenizer.from_pretrained(self.llm_model, trust_remote_code=True)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
        eos_ids = [self.tok.eos_token_id]
        try:
            eot_id = self.tok.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id >= 0: eos_ids.append(eot_id)
        except Exception:
            pass
        self.gen = pipeline(
            "text-generation",
            model=self.model, tokenizer=self.tok,
            do_sample=False, temperature=0.0, top_p=1.0,
            max_new_tokens=96, return_full_text=False,
            eos_token_id=eos_ids, pad_token_id=self.tok.eos_token_id
        )

        # prompts_FN hook (single mode)
        self.prompt_mode = str(prompt_mode).strip().lower()
        if self.prompt_mode not in {"strict","moderate"}:
            self.prompt_mode = "strict"
        self._build_prompt_pairs = _resolve_prompt_builder(self.prompt_mode)

        self.vdb_anom: Optional[Chroma] = None
        self.gate_stats = {"eligible":0, "blocked_out_of_boundary":0, "blocked_weak":0}

    # VDB
    def _build_vdb_anom(self, texts: List[str]) -> Chroma:
        os.makedirs(self.chroma_dir, exist_ok=True)
        if IS_NEW_CHROMA:
            client = chromadb.PersistentClient(path=self.chroma_dir)
            return Chroma.from_texts(texts, self.embed, client=client, collection_name="log_templates_anomaly")
        client = chromadb.PersistentClient(path=self.chroma_dir, settings=Settings(chroma_db_impl="duckdb+parquet"))
        return Chroma.from_texts(texts, self.embed, client=client, persist_directory=self.chroma_dir, collection_name="log_templates_anomaly")

    def _retrieve_anom(self, query: str, k: int) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        try:
            pairs = self.vdb_anom.similarity_search_with_relevance_scores(query, k=k)
            for d, s in pairs:
                try: sim = float(s)
                except Exception: sim = 0.0
                out.append((d.page_content, sim))
            if out: return out
        except Exception:
            pass
        try:
            pairs = self.vdb_anom.similarity_search_with_score(query, k=k)
            for d, dist in pairs:
                out.append((d.page_content, _dist_to_sim(dist)))
            return out
        except Exception:
            return out

    def _select_evidence_pairs_anom(self, templates: List[str]) -> List[dict]:
        """Top-k by similarity against Anomaly VDB (high sim = strong evidence)."""
        pairs: List[dict] = []
        for tpl in templates:
            t = str(tpl).strip()
            if not t or t == self.pad_token:
                continue
            hits = self._retrieve_anom(t, k=self.top_k)
            if hits:
                ctx, sim = hits[0]
            else:
                ctx, sim = "", 0.0
            pairs.append({"tpl": t, "ctx": str(ctx), "sim": float(sim)})
        if not pairs:
            return []
        pairs.sort(key=lambda d: d["sim"], reverse=True)  # top-k
        return pairs[: min(self.evidence_k, len(pairs))]

    def _build_pairs_block(self, pairs: List[dict]) -> str:
        lines = []
        for i, p in enumerate(pairs, 1):
            lines.append(
                f"Pair #{i}:\n"
                f"- Evidence template: {p.get('tpl','')}\n"
                f"- Top-1 anomaly template: {p.get('ctx','')}\n"
                f"- Similarity: {p.get('sim',0.0):.3f}"
            )
        return "\n\n".join(lines) if lines else "(no evidence pairs)"

    def _run_llm_pairs(self, pairs: List[dict]) -> Tuple[int, str, List[dict]]:
        """LLM decision using evidence pairs."""
        pairs_block = self._build_pairs_block(pairs)
        prompt_text = self._build_prompt_pairs(self.tok, len(pairs), pairs_block)
        out = self.gen(prompt_text)[0]["generated_text"]
        js  = _extract_json(out)
        vote = _read_anomaly(js)
        reason = _minify_reason(js.get("reason", "") if isinstance(js, dict) else "", 12)
        llm_logs = [{"prompt_id": 0, "json": js if isinstance(js, dict) else {"raw": out}}]
        return vote, reason, llm_logs

    # pred0 call/promotion gate (relaxed)
    def _eligible(
        self,
        prob: float,
        tpl: str,
        sims: List[float],
        texts: List[str],
        use_boundary: bool = True,
    ) -> Tuple[bool, dict]:
        """
        Returns: (eligible, flags)
          - boundary: candidate if within prob0_min~prob0_max
          - fast-path: promote immediately if identical or high sim (skip LLM)
          - otherwise: LLM vote if anom_high_th or anom_sim_th satisfied
        """
        if use_boundary and not (self.prob0_min <= prob < self.prob0_max):
            self.gate_stats["blocked_out_of_boundary"] += 1
            return False, {}

        top1 = sims[0] if sims else 0.0
        top2 = sims[1] if len(sims) > 1 else 0.0

        # similarity gate (optional)
        if self.anom_sim_th <= 0 and self.anom_high_th <= 0:
            sim_ok = True
        else:
            sim_ok = False
            if self.anom_sim_th > 0 and top1 >= self.anom_sim_th:
                sim_ok = True
            if self.anom_high_th > 0 and top1 >= self.anom_high_th:
                sim_ok = True
        if not sim_ok:
            self.gate_stats["blocked_weak"] += 1
            return False, {"top1": top1, "top2": top2, "reason": "sim_low"}

        self.gate_stats["eligible"] += 1
        return True, {"top1": top1, "top2": top2}

    def _build_final_reason_prompt(
        self,
        log_tpl: str,
        top_tpl: str,
        final_pred: int,
        llm_reason: str,
        guard_note: str,
        prob: float,
    ) -> str:
        tpl_snip = (log_tpl or "").strip()
        if len(tpl_snip) > 320:
            tpl_snip = tpl_snip[:317].rstrip() + "..."

        top_snip = (top_tpl or "").strip()
        if len(top_snip) > 320:
            top_snip = top_snip[:317].rstrip() + "..."

        user = (
            f"Log Template:\n{tpl_snip}\n\n"
            f"Top-1 anomaly template:\n{top_snip if top_snip else '(none)'}\n\n"
            f"Final decision (1=anomaly,0=normal): {final_pred}\n"
            f"Initial LLM note: {llm_reason or '(none)'}\n"
            f"Safety note: {guard_note or '(none)'}\n"
            f"Baseline probability: {prob:.6f}\n"
            "Return JSON with `explanation` (≤ 40 English words) describing the semantics that justify the decision."
        )

        messages = [
            {"role": "system", "content": PROMPT_FINAL_REASON},
            {"role": "user", "content": user},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _run_final_reason(self, prompt_text: str) -> Optional[str]:
        try:
            js = self.gen(prompt_text)[0]["generated_text"]
        except Exception:
            return None
        js_obj = _extract_json(js)
        if isinstance(js_obj, dict):
            expl = js_obj.get("explanation")
            if isinstance(expl, str):
                expl = expl.strip()
                if expl:
                    return expl
        return None

    def _compose_final_reason(
        self,
        tpl: str,
        top_tpl: str,
        final_pred: int,
        llm_reason: str,
        guard_note: str,
        prob: float,
        llm_called: bool,
    ) -> str:
        try:
            prompt = self._build_final_reason_prompt(
                tpl,
                top_tpl,
                final_pred,
                llm_reason,
                guard_note,
                prob,
            )
            explanation = self._run_final_reason(prompt)
            if explanation:
                return explanation
        except Exception:
            pass

        tpl_snip = tpl.strip() if tpl else "(empty template)"
        if len(tpl_snip) > 160:
            tpl_snip = tpl_snip[:157].rstrip() + "..."

        if final_pred == 1:
            if llm_reason:
                return f"Log '{tpl_snip}' escalated as anomaly: {llm_reason}"
            return f"Log '{tpl_snip}' escalated as anomaly ({guard_note or 'votes insufficient detail'})"

        # final_pred == 0
        if llm_reason:
            return f"Log '{tpl_snip}' treated as normal: {llm_reason}"
        if not llm_called:
            return f"Log '{tpl_snip}' treated as normal (LLM not called)"
        if guard_note:
            return f"Log '{tpl_snip}' treated as normal ({guard_note})"
        return f"Log '{tpl_snip}' treated as normal"

    # Run
    def run(
        self,
        pred_csv: str,
        window_csv: Optional[str],
        anomaly_csv: str,
        out_csv: str,
        raw_json: Optional[str],
        flips_csv: Optional[str],
        max_rows: Optional[int] = None,
    ):
        # Index anomaly templates
        if not anomaly_csv or not os.path.exists(anomaly_csv):
            raise SystemExit(f"anomaly_csv not found: {anomaly_csv}")
        anomalies = (
            pd.read_csv(anomaly_csv)["EventTemplate"]
            .dropna().astype(str).map(lambda s: " ".join(s.split()))
            .unique().tolist()
        )
        self.vdb_anom = self._build_vdb_anom(anomalies)

        # Load inputs
        pred_df = pd.read_csv(pred_csv)
        if "anomaly_pred" not in pred_df.columns and "is_anomaly_pred" in pred_df.columns:
            pred_df = pred_df.rename(columns={"is_anomaly_pred": "anomaly_pred"})
        window_map = _load_window_templates_map(window_csv)

        if max_rows is not None:
            pred_df = pred_df.head(int(max_rows))

        rows, raw, flip_rows = [], [], []
        called = 0
        flips = 0

        it = tqdm(pred_df.itertuples(index=False), total=len(pred_df), desc="RAG_v3_recallboost")
        for r in it:
            wid = int(r.window_id)
            prob = float(r.prob)
            base_pred_orig = int(getattr(r, "anomaly_pred", 0))
            tpls = window_map.get(wid, [])
            tpl = ""
            for t in tpls:
                tt = str(t).strip()
                if tt and tt != self.pad_token:
                    tpl = tt
                    break

            base_pred = base_pred_orig

            final_pred = base_pred
            llm_reason_raw = ""
            guard_note = "baseline anomaly" if base_pred == 1 else "baseline normal"
            last_js = None
            llm_logs: List[dict] = []
            llm_called = False
            top1_tpl_text = ""

            # Recall-first: decide whether to call based on call_policy
            call_policy = getattr(self, "call_policy", "boundary")
            if call_policy == "boundary":
                should_call = (base_pred == 0) and (self.prob0_min <= prob < self.prob0_max)
            elif call_policy == "pred0":
                should_call = (base_pred == 0)
            elif call_policy == "all":
                should_call = True
            else:
                should_call = (base_pred == 0) and (self.prob0_min <= prob < self.prob0_max)

            if should_call:
                sim_top1 = 0.0
                if tpls:
                    pairs_anom = self._select_evidence_pairs_anom(tpls)
                    if pairs_anom:
                        tpl = pairs_anom[0]["tpl"]
                        top1_tpl_text = pairs_anom[0]["ctx"]
                        sim_top1 = float(pairs_anom[0]["sim"])

                        sims = [float(p["sim"]) for p in pairs_anom]
                        texts = [str(p["ctx"]) for p in pairs_anom]
                        ok, flags = self._eligible(
                            prob, tpl, sims, texts,
                            use_boundary=(call_policy == "boundary"),
                        )
                        if ok:
                            vote, last_reason, llm_logs = self._run_llm_pairs(pairs_anom)
                            called += 1
                            llm_called = True
                            llm_reason_raw = last_reason
                            if llm_logs:
                                last_entry = llm_logs[-1]
                                last_js = last_entry.get("json", last_entry.get("raw"))

                            if vote == 1:
                                final_pred = 1
                                guard_note = "llm_pairs"
                                flips += 1
                                flip_rows.append({
                                    "window_id": wid, "before": base_pred, "after": final_pred,
                                    "prob": prob, "sim_anom_top1": sim_top1, "tpl": tpl,
                                    "ctx_hit_anom": top1_tpl_text,
                                    "llm_reason": last_reason,
                                    "guard_note": guard_note,
                                })
                            else:
                                guard_note = "llm_pairs_keep0"

                            raw.append({
                                "window_id": wid,
                                "llm_called": llm_called,
                                "result": (
                                    json.dumps(last_js) if isinstance(last_js, dict)
                                    else (str(last_js) if last_js is not None else "")
                                ),
                                "llm_outputs": llm_logs,
                                "sim_anom": sim_top1,
                                "ctx_anom": top1_tpl_text,
                                "pairs": pairs_anom,
                                "flags": flags,
                            })
                        else:
                            guard_note = "not_eligible"
                            llm_reason_raw = ""
                            raw.append({
                                "window_id": wid,
                                "llm_called": False,
                                "result": "",
                                "sim_anom": sim_top1,
                                "ctx_anom": top1_tpl_text,
                                "pairs": pairs_anom,
                                "flags": flags,
                            })
                    else:
                        guard_note = "no_evidence_pairs"
                        llm_reason_raw = ""
                else:
                    guard_note = "no_templates"
                    llm_reason_raw = ""

            # ── Final output: window_id, EventTemplate, llm_reason, anomaly ──
            final_reason = self._compose_final_reason(
                tpl,
                top1_tpl_text,
                final_pred,
                llm_reason_raw,
                guard_note,
                prob,
                llm_called,
            )

            final_reason = _minify_reason(final_reason, 40)

            if final_pred != base_pred and flip_rows:
                flip_rows[-1]["llm_reason"] = final_reason

            rows.append({
                "window_id": wid,
                "EventTemplate": tpl,
                "llm_reason": final_reason,
                "anomaly": int(final_pred),
            })

        # Save (4 columns only)
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        pd.DataFrame(rows, columns=["window_id","EventTemplate","llm_reason","anomaly"]).to_csv(out_csv, index=False)

        if raw_json is None:
            raw_json = "output/rag_raw_v3_recall_anomonly_recallboost.json"
        os.makedirs(os.path.dirname(raw_json) or ".", exist_ok=True)
        with open(raw_json, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

        if flips_csv is None:
            flips_csv = "output/rag_flips_v3_recall_anomonly_recallboost.csv"
        os.makedirs(os.path.dirname(flips_csv) or ".", exist_ok=True)
        pd.DataFrame(flip_rows, columns=[
            "window_id","before","after","prob","sim_anom_top1","tpl","ctx_hit_anom","llm_reason","guard_note"
        ]).to_csv(flips_csv, index=False)

        print(f"[✓] saved: {out_csv}")
        print(f"[i] raw: {raw_json}")
        print(f"[i] flips: {flips} saved -> {flips_csv}")
        print(f"[i] LLM-called: {called}  "
              f"boundary=[{self.prob0_min},{self.prob0_max})  "
              f"anom_th={self.anom_sim_th}  anom_high={self.anom_high_th}")

        print("\n── Gate breakdown ──")
        for k in ["eligible","blocked_out_of_boundary","blocked_weak"]:
            print(f"{k}: {self.gate_stats.get(k,0)}")

        return {"out_csv": out_csv, "flips_csv": flips_csv}

# ============ CLI ============

def build_parser():
    p = argparse.ArgumentParser("RAG v3 (recall-only, anomaly-VDB only, recall-boost) + slide-style eval (no JSON report)")
    # Inputs/DB
    p.add_argument("--pred_csv",   default=DEFAULTS["pred_csv"])
    p.add_argument("--window_csv", default=DEFAULTS["window_csv"])
    p.add_argument("--anomaly_csv",default=DEFAULTS["anomaly_csv"])
    p.add_argument("--chroma_dir", default=DEFAULTS["chroma_dir"])
    # Models/embeddings
    p.add_argument("--llm_model",  default=DEFAULTS["llm_model"])
    p.add_argument("--embed_model",default=DEFAULTS["embed_model"])
    p.add_argument("--normalize_embeddings", dest="normalize_embeddings", action="store_true")
    p.add_argument("--no_normalize_embeddings", dest="normalize_embeddings", action="store_false")
    p.set_defaults(normalize_embeddings=DEFAULTS["normalize_embeddings"])
    p.add_argument("--top_k",      type=int, default=DEFAULTS["top_k"])
    p.add_argument("--evidence_k", type=int, default=DEFAULTS["evidence_k"])
    p.add_argument("--pad_token",  default=DEFAULTS["pad_token"])
    p.add_argument("--device",     default=DEFAULTS["device"])
    # pred0 gate
    p.add_argument("--prob0_min",  type=float, default=DEFAULTS["prob0_min"])
    p.add_argument("--prob0_max",  type=float, default=DEFAULTS["prob0_max"])
    p.add_argument(
        "--call_policy",
        choices=["boundary", "pred0", "all"],
        default=DEFAULTS["call_policy"],
    )
    p.add_argument("--anom_sim_th", type=float, default=DEFAULTS["anom_sim_th"])
    p.add_argument("--anom_high_th", type=float, default=DEFAULTS["anom_high_th"])
    # Prompt mode (single mode)
    p.add_argument("--prompt_mode", choices=["strict","moderate"], default=DEFAULTS["prompt_mode"])
    # Outputs
    p.add_argument("--out_csv",    default=DEFAULTS["out_csv"])
    p.add_argument("--raw_json",   default=DEFAULTS["raw_json"])
    p.add_argument("--flips_csv",  default=DEFAULTS["flips_csv"])
    p.add_argument("--max_rows", type=int, default=None, help="Optional limit on number of windows to process")
    # Evaluation
    p.add_argument("--eval_gt_csv",     default=DEFAULTS["eval_gt_csv"],
                   help="GT CSV for evaluation (default: output/gt_test.csv)")
    p.add_argument("--eval_label_col",  default=DEFAULTS["eval_label_col"])
    p.add_argument("--export_report",   default=DEFAULTS["export_report"])
    p.add_argument("--skip_eval",       action="store_true", default=DEFAULTS["skip_eval"])
    return p

def main():
    p = build_parser()
    args = p.parse_args()

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    rag = RAGPostProcessorV3RecallBoost(
        llm_model = args.llm_model,
        embed_model = args.embed_model,
        chroma_dir = args.chroma_dir,
        top_k = args.top_k,
        evidence_k = args.evidence_k,
        pad_token = args.pad_token,
        device = args.device,
        normalize_embeddings = args.normalize_embeddings,
        prob0_min = args.prob0_min,
        prob0_max = args.prob0_max,
        call_policy = args.call_policy,
        anom_sim_th = args.anom_sim_th,
        anom_high_th = args.anom_high_th,
        prompt_mode = args.prompt_mode,
    )

    ret = rag.run(
        pred_csv   = args.pred_csv,
        window_csv = args.window_csv,
        anomaly_csv= args.anomaly_csv,
        out_csv    = args.out_csv,
        raw_json   = args.raw_json,
        flips_csv  = args.flips_csv,
        max_rows   = args.max_rows,
    )

    if not args.skip_eval:
        if not args.eval_gt_csv or not os.path.exists(args.eval_gt_csv):
            print(f"[w] GT file not found, skipping evaluation: {args.eval_gt_csv}")
        else:
            try:
                pred_df = pd.read_csv(ret["out_csv"])[["window_id", "anomaly"]]
                gt_df = pd.read_csv(args.eval_gt_csv)[["window_id", args.eval_label_col]]
                merged = pred_df.merge(gt_df, on="window_id", how="inner").dropna()
                merged.rename(columns={args.eval_label_col: "label"}, inplace=True)
                m = evalmod._metrics(merged["label"], merged["anomaly"])
                evalmod._fmt_block("RAG + LLM Final", m)
                if args.export_report:
                    os.makedirs(os.path.dirname(args.export_report), exist_ok=True)
                    with open(args.export_report, "w", encoding="utf-8") as f:
                        json.dump({"final": m}, f, ensure_ascii=False, indent=2)
                    print(f"[✓] Metrics saved: {args.export_report}")
                print(f"[✓] Evaluation samples: {len(merged):,}")
            except Exception as e:
                print(f"[w] Evaluation error: {e}")

if __name__ == "__main__":
    main()
