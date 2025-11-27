#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG_v3_recall_anomonly_recallboost.py — recall-boost post-processing (anomaly VDB only)
- pred==1 (anomaly) is kept (no FP logic)
- pred==0 (normal): retrieve anomaly VDB + (fast path OR LLM voting) to flip 0→1
- Relaxations:
  * Looser prob/sim thresholds, no top2 requirement, no keyword requirement
  * Fast path: identical/very high sim → escalate without LLM
  * Dual prompt voting (optional) + evidence-based bonus vote (prior)
  * top-k=10
- Outputs: final CSV (4 cols), raw JSON, flips CSV
- Eval (optional): slide-style block with confusion matrix; no JSON report
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
try:
    prompts_mod_general = importlib.import_module("postprocess.prompts")
    PROMPT_FINAL_REASON = getattr(prompts_mod_general, "PROMPT_FINAL_REASON")
except Exception as e:
    raise ImportError(f"Failed to import PROMPT_FINAL_REASON: {e}")

# ───────── Defaults (recall-boost) ─────────
DEFAULTS = {
    # inputs/DB
    "pred_csv":   "output/anomaly_logs_detected_by_logformer.csv",
    "window_csv": "output/window_repr_by_pred_info.csv",
    "anomaly_csv":"dataset/anomaly_templates_clean.csv",
    "chroma_dir": "rag_db/aosp_bge_v3_recall_anomonly_recallboost",

    # model/embedding
    "llm_model":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "embed_model": "BAAI/bge-small-en-v1.5",
    "normalize_embeddings": True,
    "top_k": 10,
    "device": "cuda",

    # pred==0 (missed anomaly) gates/thresholds (relaxed)
    "prob0_min": 0.30,
    "prob0_max": 0.80,       # wider than base_threshold=0.70
    "anom_sim_th": 0.93,     # default sim threshold, no keywords needed
    "anom_high_th": 0.980,   # high-confidence sim
    "top2_th": 0.88,
    "require_top2": False,   # no top-2 requirement
    "use_keywords": False,   # no keyword requirement (increase coverage)

    # voting (relaxed)
    "vote_prompts": 5,
    "min_votes": 3,          # 3/5 agreement flips
    "dual_prompt_voting": True,       # mix strict + moderate
    "strict_ratio": 0.6,              # 60% strict, 40% moderate

    # evidence-based bonus vote
    "prior_bonus_sim": 0.990, # add +1 vote if top-1 sim >= 0.990

    # fast-path: very high sim → escalate without LLM
    "enable_fast_path": True,
    "fast_path_sim": 0.992,   # escalate immediately if above
    "identical_sim": 0.995,   # treat as identical

    # Base threshold override
    "base_threshold": 0.70,

    # Prompt mode (strict/moderate, mixed when dual_prompt_voting=True)
    "prompt_mode": "strict",  # strict | moderate

    # output
    "out_csv":   "output/rag_result_v3_recall_anomonly_recallboost.csv",
    "raw_json":  "output/rag_raw_v3_recall_anomonly_recallboost.json",
    "flips_csv": "output/rag_flips_v3_recall_anomonly_recallboost.csv",

    # ===== Evaluation =====
    "eval_gt_csv": "output/gt_all.csv",
    "eval_label_col": "label",
    "skip_eval": False,
}

# ============ Utils ============

RE_CODEJSON = re.compile(r"```json\s*({[\s\S]*?})\s*```", re.I)
RE_ANYJSON  = re.compile(r"({[\s\S]*})")
RE_ANOMKW   = re.compile(r"\b(error|fail(?:ed|ure)?|panic|anr|crash|died|timeout|exception|segfault)\b", re.I)

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
    """Read anomaly value from LLM JSON; prefers anomaly, supports is_anomaly."""
    if not isinstance(js, dict):
        return 0
    if "anomaly" in js:
        try: return int(js.get("anomaly", 0))
        except Exception: return 0
    if "is_anomaly" in js:
        try: return int(js.get("is_anomaly", 0))
        except Exception: return 0
    return 0

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

def _norm_tpl(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _same_template(a: str, b: str, sim: float, sim_th: float) -> bool:
    """Treat as identical if case/space-insensitive match or very high sim."""
    if _norm_tpl(a) == _norm_tpl(b):
        return True
    return sim >= sim_th

def _load_benign_patterns(path: Optional[str]) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    if not path: return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith("#"): continue
                out.append(re.compile(line, re.I))
        print(f"[i] loaded benign patterns: {len(out)}")
    except Exception as e:
        print(f"[w] failed to load benign patterns: {e}")
    return out

def _match_benign(regexes: List[re.Pattern], text: str) -> bool:
    if not regexes: return False
    t = text or ""
    for rx in regexes:
        if rx.search(t): return True
    return False

# ============ Force prompts_FN (constants only, no fallback) ============

def _resolve_prompt_builder(prompt_mode: str):
    """
    Use only string constants from external prompts_FN module (no fallback).
    required constants:
      - PROMPT_RECALL_STRICT
      - PROMPT_RECALL_MODERATE
    """
    import importlib

    last_err = None
    mod = None
    tried = []
    for name in ("postprocess.prompts_FN", "prompts_FN"):
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
            "[prompts_FN] Cannot find prompts_FN module for missed anomaly.\n"
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
            "[prompts_FN] required prompt constants not found.\n"
            f"  - Required const: {const_name}\n"
            f"  - Loaded module: {getattr(mod, '__file__', 'NA')}\n"
            f"  - example uppercase symbols: {sample}"
        )

    tmpl: str = getattr(mod, const_name)
    if not isinstance(tmpl, str):
        raise TypeError(f"[prompts_FN] {const_name} must be a string. (got {type(tmpl)})")

    def _build(tok, top_k: int, tpl: str, ctx_lines: List[str]) -> str:
        context = "\n".join(ctx_lines) if ctx_lines else "(no similar anomalies)"
        filled = tmpl.format(top_k=top_k, question=tpl, context=context)
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
        top_k: int = 10,
        device: str = "cuda",
        normalize_embeddings: bool = True,
        # pred0 policy/thresholds (relaxed)
        prob0_min: float = 0.30,
        prob0_max: float = 0.80,
        anom_sim_th: float = 0.93,
        anom_high_th: float = 0.980,
        top2_th: float = 0.88,
        require_top2: bool = False,
        use_keywords: bool = False,
        # multi-vote
        vote_prompts: int = 5,
        min_votes: int = 3,
        dual_prompt_voting: bool = True,
        strict_ratio: float = 0.6,
        # bonus vote(prior)
        prior_bonus_sim: float = 0.990,
        # fast-path escalation
        enable_fast_path: bool = True,
        fast_path_sim: float = 0.992,
        identical_sim: float = 0.995,
        # baseline recomputation
        base_threshold: Optional[float] = 0.70,
        # prompts
        prompt_mode: str = "strict",
    ):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.chroma_dir = chroma_dir
        self.top_k = int(top_k)
        self.device = device
        self.base_threshold = base_threshold

        self.prob0_min    = float(prob0_min)
        self.prob0_max    = float(prob0_max)
        self.anom_sim_th  = float(anom_sim_th)
        self.anom_high_th = float(anom_high_th)
        self.top2_th      = float(top2_th)
        self.require_top2 = bool(require_top2)
        self.use_keywords = bool(use_keywords)

        self.vote_prompts = int(vote_prompts)
        self.min_votes    = int(min_votes)
        self.dual_prompt_voting = bool(dual_prompt_voting)
        self.strict_ratio = float(strict_ratio)

        self.prior_bonus_sim = float(prior_bonus_sim)

        self.enable_fast_path = bool(enable_fast_path)
        self.fast_path_sim    = float(fast_path_sim)
        self.identical_sim    = float(identical_sim)

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

        # connect prompts_FN
        self.prompt_mode = str(prompt_mode).strip().lower()
        if self.prompt_mode not in {"strict","moderate"}:
            self.prompt_mode = "strict"
        self._build_prompt_single = _resolve_prompt_builder(self.prompt_mode)
        # helper for mixed voting
        self._build_prompt_strict  = _resolve_prompt_builder("strict")
        self._build_prompt_mod     = _resolve_prompt_builder("moderate")

        self.vdb_anom: Optional[Chroma] = None
        self.gate_stats = {"eligible":0, "vote_fail":0, "blocked_out_of_boundary":0, "blocked_weak":0, "fast_path":0}

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

    # pred0 call/escalation gates (relaxed)
    def _eligible(self, prob: float, tpl: str, sims: List[float], texts: List[str]) -> Tuple[bool, dict]:
        """
        Return: (eligible, flags)
          - call candidate if within boundary (prob0_min~prob0_max)
          - fast-path: identical or very high sim → escalate immediately (skip LLM)
          - general candidate: if anom_high_th or anom_sim_th met, try voting to escalate
        """
        if not (self.prob0_min <= prob <= self.prob0_max):
            self.gate_stats["blocked_out_of_boundary"] += 1
            return False, {}

        top1 = sims[0] if sims else 0.0
        top2 = sims[1] if len(sims) > 1 else 0.0

        # identical / near-identical check
        identical = False
        if texts:
            if _same_template(tpl, texts[0], top1, self.identical_sim):
                identical = True

        # fast-path escalation
        if self.enable_fast_path and (identical or top1 >= self.fast_path_sim):
            self.gate_stats["fast_path"] += 1
            return True, {"fast_path": True, "identical": identical, "top1": top1, "top2": top2}

        # general candidate (vote to escalate)
        strong = (top1 >= self.anom_high_th) or (top1 >= self.anom_sim_th)
        if strong:
            self.gate_stats["eligible"] += 1
            return True, {"fast_path": False, "identical": identical, "top1": top1, "top2": top2}

        self.gate_stats["blocked_weak"] += 1
        return False, {"top1": top1, "top2": top2}

    def _vote_llm(
        self, tpl: str, pairs_anom: List[Tuple[str, float]]
    ) -> Tuple[int, str, int, int, List[dict]]:
        """
        LLM multi-vote (dual_prompt_voting applicable)
        Return: (final_vote_1, last_reason, raw_vote_1, bonus, llm_logs)
        """
        ctx_a = [f"{s:.3f} :: {t}" for (t, s) in pairs_anom]
        sims  = [float(s) for (_, s) in pairs_anom]
        top1  = sims[0] if sims else 0.0

        votes = []
        llm_logs: List[dict] = []
        # build dual prompts
        if self.dual_prompt_voting:
            n_strict = int(round(self.vote_prompts * self.strict_ratio))
            n_mod    = max(0, self.vote_prompts - n_strict)
            builders = ([self._build_prompt_strict]*n_strict) + ([self._build_prompt_mod]*n_mod)
        else:
            builders = [self._build_prompt_single]*self.vote_prompts

        last_js, last_reason = None, ""
        for idx, build in enumerate(builders):
            prompt_text = build(self.tok, self.top_k, tpl, ctx_a)
            out = self.gen(prompt_text)[0]["generated_text"]
            js  = _extract_json(out)
            last_js = js
            ia = _read_anomaly(js)
            try: votes.append(int(ia))
            except Exception: votes.append(0)
            if isinstance(js, dict):
                last_reason = _minify_reason(js.get("reason", ""), 12)
                llm_logs.append({"prompt_id": idx, "json": js})
            else:
                llm_logs.append({"prompt_id": idx, "raw": out})

        raw_ones = sum(votes)

        # evidence bonus vote: +1 if top1 similarity is very high
        bonus = 1 if top1 >= self.prior_bonus_sim else 0
        final_ones = raw_ones + bonus

        return final_ones, last_reason, raw_ones, bonus, llm_logs

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

    # run
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
        # index anomaly templates
        if not anomaly_csv or not os.path.exists(anomaly_csv):
            raise SystemExit(f"anomaly_csv not found: {anomaly_csv}")
        anomalies = (
            pd.read_csv(anomaly_csv)["EventTemplate"]
            .dropna().astype(str).map(lambda s: " ".join(s.split()))
            .unique().tolist()
        )
        self.vdb_anom = self._build_vdb_anom(anomalies)

        # load input
        pred_df = pd.read_csv(pred_csv)
        if "anomaly_pred" not in pred_df.columns and "is_anomaly_pred" in pred_df.columns:
            pred_df = pred_df.rename(columns={"is_anomaly_pred": "anomaly_pred"})
        if window_csv and os.path.exists(window_csv):
            w = pd.read_csv(window_csv)[["window_id","EventTemplate"]]
            pred_df = pred_df.merge(w, on="window_id", how="left")

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
            tpl = str(getattr(r, "EventTemplate", ""))

            # recompute baseline (prob threshold)
            base_pred = base_pred_orig
            if self.base_threshold is not None:
                base_pred = 1 if prob >= float(self.base_threshold) else 0

            final_pred = base_pred
            llm_reason_raw = ""
            guard_note = "baseline anomaly" if base_pred == 1 else "baseline normal"
            last_js = None
            llm_logs: List[dict] = []
            llm_called = False
            top1_tpl_text = ""

            # recover missed anomaly only when pred==0
            if tpl and base_pred == 0:
                pairs_anom = self._retrieve_anom(tpl, self.top_k)
                sims  = [float(s) for (_, s) in pairs_anom]
                texts = [str(t)   for (t, _) in pairs_anom]
                sim_top1 = sims[0] if sims else 0.0
                top1_tpl_text = texts[0] if texts else ""

                ok, flags = self._eligible(prob, tpl, sims, texts)
                if ok:
                    if flags.get("fast_path", False):
                        # escalate immediately without LLM
                        fpred = 1
                        flips += 1
                        flip_rows.append({
                            "window_id": wid, "before": base_pred, "after": fpred,
                            "prob": prob, "sim_anom_top1": sim_top1, "tpl": tpl,
                            "ctx_hit_anom": (pairs_anom[0][0] if pairs_anom else ""),
                            "llm_reason": "",
                            "guard_note": "fast_path" + (" (identical)" if flags.get("identical") else ""),
                        })
                        final_pred = fpred
                        guard_note = "fast_path"
                        llm_reason_raw = "fast-path promote"
                    else:
                        # LLM voting
                        vote_ones, last_reason, raw_ones, bonus, llm_logs = self._vote_llm(tpl, pairs_anom)
                        called += 1
                        llm_called = True
                        if llm_logs:
                            last_entry = llm_logs[-1]
                            last_js = last_entry.get("json", last_entry.get("raw"))
                        if vote_ones >= self.min_votes:
                            fpred = 1
                            flips += 1
                            flip_rows.append({
                                "window_id": wid, "before": base_pred, "after": fpred,
                                "prob": prob, "sim_anom_top1": sim_top1, "tpl": tpl,
                                "ctx_hit_anom": (pairs_anom[0][0] if pairs_anom else ""),
                                "llm_reason": "",
                                "guard_note": f"escalated (votes={raw_ones}+{bonus}prior >= {self.min_votes})",
                            })
                            final_pred = fpred
                            guard_note = f"escalated (votes={raw_ones}+{bonus}prior >= {self.min_votes})"
                            llm_reason_raw = last_reason
                        else:
                            self.gate_stats["vote_fail"] += 1
                            guard_note = "vote_fail"
                            llm_reason_raw = last_reason
                else:
                    guard_note = "not_eligible"
                    llm_reason_raw = ""

                raw.append({
                    "window_id": wid,
                    "llm_called": llm_called,
                    "result": (
                        json.dumps(last_js) if isinstance(last_js, dict)
                        else (str(last_js) if last_js is not None else "")
                    ),
                    "llm_outputs": llm_logs,
                    "sim_anom": sim_top1 if pairs_anom else 0.0,
                    "ctx_anom": (pairs_anom[0][0] if pairs_anom else ""),
                })

            # ── final output: window_id, EventTemplate, llm_reason, anomaly ──
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

        # save (only requested 4 columns)
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        pd.DataFrame(rows, columns=["window_id","EventTemplate","llm_reason","anomaly"]).to_csv(out_csv, index=False)

        if raw_json is None:
            os.makedirs("output", exist_ok=True)
            raw_json = "output/rag_raw_v3_recall_anomonly_recallboost.json"
        with open(raw_json, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

        if flips_csv is None:
            flips_csv = "output/rag_flips_v3_recall_anomonly_recallboost.csv"
        pd.DataFrame(flip_rows, columns=[
            "window_id","before","after","prob","sim_anom_top1","tpl","ctx_hit_anom","llm_reason","guard_note"
        ]).to_csv(flips_csv, index=False)

        print(f"[✓] saved: {out_csv}")
        print(f"[i] raw: {raw_json}")
        print(f"[i] flips: {flips} saved -> {flips_csv}")
        print(f"[i] LLM-called: {called}  "
              f"boundary=[{self.prob0_min},{self.prob0_max}]  "
              f"anom_th={self.anom_sim_th}  anom_high={self.anom_high_th}  "
              f"top2={'on' if self.require_top2 else 'off'} (th={self.top2_th})  "
              f"vote={self.vote_prompts}/{self.min_votes}  "
              f"dual_prompt={self.dual_prompt_voting} (strict_ratio={self.strict_ratio})  "
              f"prior@{self.prior_bonus_sim}  fast_path={'on' if self.enable_fast_path else 'off'}@{self.fast_path_sim}")

        print("\n── Gate breakdown ──")
        for k in ["fast_path","eligible","vote_fail","blocked_out_of_boundary","blocked_weak"]:
            print(f"{k}: {self.gate_stats.get(k,0)}")

        return {"out_csv": out_csv, "flips_csv": flips_csv}

# ============ Evaluation (slide-style final block only) ============

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
    return dict(TP=TP, TN=TN, FP=FP, FN=FN, precision=prec, recall=rec, f1=f1)

def _fmt_final_slide_style(m: dict):
    print("\n— RAG + LLM Final —")
    print(f"Precision = {m['precision']:.4f}   Recall = {m['recall']:.4f}   F1 = {m['f1']:.4f}")
    print("Confusion-matrix  [ [TN FP] [FN TP] ]")
    wl = max(len(str(m["TN"])), len(str(m["FN"])))
    wr = max(len(str(m["FP"])), len(str(m["TP"])))
    print(f"[[ {m['TN']:>{wl}} {m['FP']:>{wr}}]\n [ {m['FN']:>{wl}} {m['TP']:>{wr}}]]")

def evaluate_simple_only_final(pred_csv: str, out_csv: str, gt_csv: str, label_col: str,
                               base_threshold: Optional[float] = 0.70):
    out  = pd.read_csv(out_csv)      # window_id, EventTemplate, llm_reason, anomaly
    gt   = pd.read_csv(gt_csv)[["window_id", label_col]].rename(columns={label_col:"label"})
    pred = pd.read_csv(pred_csv)[["window_id","prob"]]

    df = out.merge(pred, on="window_id", how="left").merge(gt, on="window_id", how="left")
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    # Compute Stage-1 (prob threshold) and Final, but display Final only
    stage1 = (df["prob"] >= float(base_threshold)).astype(int)
    final  = df["anomaly"].astype(int)

    mf = _metrics(df["label"], final)
    _fmt_final_slide_style(mf)

# ============ CLI ============

def build_parser():
    p = argparse.ArgumentParser("RAG v3 (recall-only, anomaly-VDB only, recall-boost) + slide-style eval (no JSON report)")
    # inputs/DB
    p.add_argument("--pred_csv",   default=DEFAULTS["pred_csv"])
    p.add_argument("--window_csv", default=DEFAULTS["window_csv"])
    p.add_argument("--anomaly_csv",default=DEFAULTS["anomaly_csv"])
    p.add_argument("--chroma_dir", default=DEFAULTS["chroma_dir"])
    # model/embedding
    p.add_argument("--llm_model",  default=DEFAULTS["llm_model"])
    p.add_argument("--embed_model",default=DEFAULTS["embed_model"])
    p.add_argument("--normalize_embeddings", action="store_true", default=DEFAULTS["normalize_embeddings"])
    p.add_argument("--top_k",      type=int, default=DEFAULTS["top_k"])
    p.add_argument("--device",     default=DEFAULTS["device"])
    # pred0 gate (relaxed)
    p.add_argument("--prob0_min",  type=float, default=DEFAULTS["prob0_min"])
    p.add_argument("--prob0_max",  type=float, default=DEFAULTS["prob0_max"])
    p.add_argument("--anom_sim_th", type=float, default=DEFAULTS["anom_sim_th"])
    p.add_argument("--anom_high_th", type=float, default=DEFAULTS["anom_high_th"])
    p.add_argument("--top2_th",    type=float, default=DEFAULTS["top2_th"])
    p.add_argument("--require_top2", action="store_true", default=DEFAULTS["require_top2"])
    p.add_argument("--use_keywords", action="store_true", default=DEFAULTS["use_keywords"])
    # voting
    p.add_argument("--vote_prompts", type=int, default=DEFAULTS["vote_prompts"])
    p.add_argument("--min_votes",    type=int, default=DEFAULTS["min_votes"])
    p.add_argument("--dual_prompt_voting", action="store_true", default=DEFAULTS["dual_prompt_voting"])
    p.add_argument("--strict_ratio", type=float, default=DEFAULTS["strict_ratio"])
    # bonus vote / fast-path
    p.add_argument("--prior_bonus_sim", type=float, default=DEFAULTS["prior_bonus_sim"])
    p.add_argument("--enable_fast_path", action="store_true", default=DEFAULTS["enable_fast_path"])
    p.add_argument("--fast_path_sim", type=float, default=DEFAULTS["fast_path_sim"])
    p.add_argument("--identical_sim", type=float, default=DEFAULTS["identical_sim"])
    # first-stage threshold
    p.add_argument("--base_threshold", type=float, default=DEFAULTS["base_threshold"])
    # prompt mode (mix when dual_prompt_voting)
    p.add_argument("--prompt_mode", choices=["strict","moderate"], default=DEFAULTS["prompt_mode"])
    # output
    p.add_argument("--out_csv",    default=DEFAULTS["out_csv"])
    p.add_argument("--raw_json",   default=DEFAULTS["raw_json"])
    p.add_argument("--flips_csv",  default=DEFAULTS["flips_csv"])
    p.add_argument("--max_rows", type=int, default=None, help="Optional limit on number of windows to process")
    # evaluation
    p.add_argument("--eval_gt_csv",     default=DEFAULTS["eval_gt_csv"])
    p.add_argument("--eval_label_col",  default=DEFAULTS["eval_label_col"])
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
        device = args.device,
        normalize_embeddings = args.normalize_embeddings,
        prob0_min = args.prob0_min,
        prob0_max = args.prob0_max,
        anom_sim_th = args.anom_sim_th,
        anom_high_th = args.anom_high_th,
        top2_th = args.top2_th,
        require_top2 = args.require_top2,
        use_keywords = args.use_keywords,
        vote_prompts = args.vote_prompts,
        min_votes = args.min_votes,
        dual_prompt_voting = args.dual_prompt_voting,
        strict_ratio = args.strict_ratio,
        prior_bonus_sim = args.prior_bonus_sim,
        enable_fast_path = args.enable_fast_path,
        fast_path_sim = args.fast_path_sim,
        identical_sim = args.identical_sim,
        base_threshold = args.base_threshold,
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
            print(f"[w] GT file not found; skipping evaluation: {args.eval_gt_csv}")
        else:
            evaluate_simple_only_final(
                pred_csv = args.pred_csv,
                out_csv  = ret["out_csv"],
                gt_csv   = args.eval_gt_csv,
                label_col= args.eval_label_col,
                base_threshold = args.base_threshold,
            )

if __name__ == "__main__":
    main()
