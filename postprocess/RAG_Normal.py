from __future__ import annotations
import os, sys, re, gc, json, argparse, types
from typing import List, Tuple, Optional

import pandas as pd
from tqdm import tqdm
from packaging.version import parse as vparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# robust import of prompts with fallback
import importlib
try:
    prompts_mod = importlib.import_module("postprocess.prompts")
except Exception as e:
    raise ImportError(f"Failed to import postprocess.prompts: {e}")

try:
    PROMPT_SYSTEM_MODERATE = getattr(prompts_mod, "PROMPT_SYSTEM_MODERATE")
except AttributeError:
    raise ImportError("PROMPT_SYSTEM_MODERATE is missing in postprocess/prompts.py")

# Use MODERATE if STRICT is missing
PROMPT_SYSTEM_STRICT = getattr(prompts_mod, "PROMPT_SYSTEM_STRICT", PROMPT_SYSTEM_MODERATE)
PROMPT_FINAL_REASON = getattr(prompts_mod, "PROMPT_FINAL_REASON")
PROMPT_FINAL_REASON = getattr(prompts_mod, "PROMPT_FINAL_REASON")

# ── ===== Defaults (based on last run) ===== ──
DEFAULTS = {
    "pred_csv":   "output/anomaly_logs_detected_by_logformer.csv",
    "window_csv": "output/window_repr_by_pred_info.csv",
    "normal_csv": "dataset/normal_templates_clean_.csv",
    "chroma_dir": "rag_db/aosp_bge_run1",

    "llm_model":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "embed_model": "BAAI/bge-small-en-v1.5",
    "normalize_embeddings": True,

    "top_k": 3,
    "device": "cuda",

    "call_policy": "hybrid",
    "prob_lo": 0.35,
    "prob_hi_b": 0.85,
    "base_threshold": 0.70,

    "flip_sim_th": 0.98,
    "prob_hi": 0.98,
    "enable_identical_autoflip": True,
    "identical_sim": 0.998,
    "prob_hi_identical": 0.98,

    "enable_zero_to_one": False,   # default OFF (precision-first)
    "low_sim_th": 0.70,
    "prob0_min": 0.45,

    "benign_patterns": None,  # e.g., "config/benign_patterns.txt"

    "out_csv":   "output/rag_result.csv",
    "raw_json":  "output/rag_raw.json",
    "flips_csv": "output/rag_flips.csv",
}

# ── runtime patches (posthog, sqlite, chroma telemetry) ──
posthog_stub = types.ModuleType("posthog")
posthog_stub.Client = lambda *a, **k: None
sys.modules["posthog"] = posthog_stub
try:
    import pysqlite3.dbapi2 as sqlite3  # noqa
    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass
os.environ["CHROMA_TELEMETRY"] = "FALSE"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# langchain 0.2.2 warnings can be ignored; upgrade if needed.
from langchain_community.embeddings   import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings

IS_NEW_CHROMA = vparse(chromadb.__version__) >= vparse("0.5.0")

# ============ Utils ============
RE_ANCHOR_JSON = re.compile(r"```json\s*({[\s\S]*?})\s*```", re.I)
RE_ANY_JSON    = re.compile(r"({[\s\S]*})")
RE_ANOMALY_KEYWORDS = re.compile(
    r"\b(error|fail(?:ed|ure)?|panic|anr|crash|died|not\s+found|unavailable|permission\s+denied|timeout|segfault|exception)\b",
    re.I
)

# Additional regex to clean reasons (remove meta words/quotes)
RE_META_WORDS = re.compile(r"\b(similarity|normal|anomaly|confidence|score|probability|cosine)\b", re.I)
RE_QUOTES     = re.compile(r"[`“”\"']+")

def _minify_reason(s: str, n_words: int = 12) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    words = s.split()
    return " ".join(words[:n_words])

def _clean_reason(s: str) -> str:
    """Remove meta words/quotes to keep CSV reason tidy."""
    s = (s or "").strip()
    s = RE_META_WORDS.sub("", s)
    s = RE_QUOTES.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "unknown"

def _extract_json(txt: str):
    """Parse the first JSON object from model output."""
    if not isinstance(txt, str):
        return None
    m = RE_ANCHOR_JSON.search(txt)
    if not m:
        m = RE_ANY_JSON.search(txt)
    if not m:
        return None
    blob = m.group(1).strip()
    try:
        return json.loads(blob)
    except Exception:
        blob = re.sub(r"\s+", " ", blob)
        try:
            return json.loads(blob)
        except Exception:
            return None


def _safe_int(v, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return default


def _is_strict_zero(v) -> bool:
    try:
        return int(v) == 0
    except Exception:
        return False

def _is_strict_one(v) -> bool:
    try:
        return int(v) == 1
    except Exception:
        return False

def _has_anomaly_keywords(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(RE_ANOMALY_KEYWORDS.search(text))

def _norm_tpl(s: str) -> str:
    """Normalize template for comparison: lowercase and collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _same_template(a: str, b: str) -> bool:
    """Check exact/effectively identical templates (case/space insensitive)."""
    return _norm_tpl(a) == _norm_tpl(b)

def _dist_to_sim(dist: float) -> float:
    """Convert distance (lower=closer) to a safe similarity in [0,1]."""
    try:
        d = float(dist)
    except Exception:
        return 0.0
    if d < 0:
        d = abs(d)
    return 1.0 / (1.0 + d)  # d=0→1.0, d=1→0.5, d→∞→0

# ============ RAG core ============
class RAGPostProcessorV2:
    def __init__(
        self,
        llm_name: str,
        top_k: int,
        chroma_dir: str,
        device: str = "cuda",
        flip_sim_th: float = 0.92,
        prob_hi: float = 0.98,
        # identical template auto 1→0 guard
        enable_identical_autoflip: bool = False,
        identical_sim: float = 0.995,
        prob_hi_identical: float = 0.999,
        # 0→1 escalation (default OFF, very conservative)
        enable_zero_to_one: bool = False,
        low_sim_th: float = 0.70,
        prob0_min: float = 0.45,
        # override first-stage threshold
        base_threshold: Optional[float] = None,
        # embeddings replace/normalize options
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = False,
        # normal pattern (regex) whitelist file
        benign_patterns_path: Optional[str] = None,):
        self.llm_name   = llm_name
        self.top_k      = top_k
        self.chroma_dir = chroma_dir
        self.device     = device

    # 1→0 general guard
        self.flip_sim_th = float(flip_sim_th)
        self.prob_hi     = float(prob_hi)

        # 1→0 identical guard
        self.enable_identical_autoflip = bool(enable_identical_autoflip)
        self.identical_sim = float(identical_sim)
        self.prob_hi_identical = float(prob_hi_identical)

        # 0→1 correction
        self.enable_zero_to_one = bool(enable_zero_to_one)
        self.low_sim_th  = float(low_sim_th)
        self.prob0_min   = float(prob0_min)

        # override first-stage threshold
        self.base_threshold = base_threshold

        # embeddings (replaceable)
        self.embed = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )

        # LLM setup (chat format + EOS)
        self.tok = AutoTokenizer.from_pretrained(self.llm_name, trust_remote_code=True)
        self.tok.pad_token    = self.tok.eos_token
        self.tok.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_name, device_map="auto", torch_dtype="auto",
            trust_remote_code=True
        )

        # set eos (<|eot_id|> if available)
        eot_id = None
        try:
            eot_id = self.tok.convert_tokens_to_ids("<|eot_id|>")
        except Exception:
            pass
        self.eos_ids = [i for i in [self.tok.eos_token_id, eot_id] if isinstance(i, int) and i >= 0]

        self.gen = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tok,
            do_sample=False,            # greedy
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=96,
            repetition_penalty=1.05,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.eos_ids if self.eos_ids else None,
            return_full_text=False,
        )

        # benign patterns
        self.benign_regexes: List[re.Pattern] = []
        if benign_patterns_path:
            try:
                with open(benign_patterns_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        self.benign_regexes.append(re.compile(line, re.I))
                print(f"[i] loaded benign patterns: {len(self.benign_regexes)}")
            except Exception as e:
                print(f"[w] failed to load benign patterns: {e}")

    # ---- Vector DB ----
    def _build_vdb(self, normals: List[str]) -> Chroma:
        os.makedirs(self.chroma_dir, exist_ok=True)

        if IS_NEW_CHROMA:
            client = chromadb.PersistentClient(path=self.chroma_dir)
            return Chroma.from_texts(
                normals, self.embed, client=client, collection_name="log_templates_clean"
            )

        client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(chroma_db_impl="duckdb+parquet")
        )
        return Chroma.from_texts(
            normals, self.embed, client=client,
            persist_directory=self.chroma_dir,
            collection_name="log_templates_clean"
        )

    def _retrieve(self, vdb: Chroma, query: str, k: int) -> List[Tuple[str, float]]:
        """
        Return top-k templates and similarity for a query.
        List of (text, similarity[0..1]).

        Prefer relevance_scores; fallback to distance → 1/(1+dist).
        """
        out: List[Tuple[str, float]] = []

        # 1) prefer relevance scores
        try:
            pairs = vdb.similarity_search_with_relevance_scores(query, k=k)
            for d, s in pairs:
                try:
                    sim = float(s)
                except Exception:
                    sim = 0.0
                out.append((d.page_content, sim))
            if out:
                return out
        except Exception:
            pass

        # 2) distance score fallback
        try:
            pairs = vdb.similarity_search_with_score(query, k=k)
            for d, dist in pairs:
                sim = _dist_to_sim(dist)
                out.append((d.page_content, sim))
            return out
        except Exception:
            return out

    # ---- build chat prompt ----
    def _build_chat(self, log_tpl: str, ctx_lines: List[str], strict: bool = False) -> str:
        """
        Build Meta-Llama-3 chat prompt:
          - System prompt: MODERATE / STRICT
          - Context: strip similarity numbers, keep text only
        """
        # strip similarity numbers ("0.996 :: text" → "• text")
        if ctx_lines:
            cleaned_ctx = []
            for line in ctx_lines[: self.top_k]:
                parts = line.split("::", 1)
                txt = parts[-1].strip() if len(parts) == 2 else line.strip()
                if txt:
                    cleaned_ctx.append(f"• {txt}")
            top_ctx = "\n".join(cleaned_ctx) if cleaned_ctx else "(no similar normals)"
        else:
            top_ctx = "(no similar normals)"

        system = PROMPT_SYSTEM_STRICT if strict else PROMPT_SYSTEM_MODERATE

        user = (
            f"Log Template:\n{log_tpl}\n\n"
            f"Top-{self.top_k} candidate normal templates:\n{top_ctx}\n\n"
            "Return exactly one fenced `json` block, nothing else."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        rendered = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return rendered

    def _build_final_reason_prompt(
        self,
        log_tpl: str,
        top1_tpl: str,
        final_pred: int,
        guard_note: str,
        llm_reason: str,
        llm_vote: Optional[int],
        prob: float,
    ) -> str:
        tpl_snip = (log_tpl or "").strip()
        if len(tpl_snip) > 320:
            tpl_snip = tpl_snip[:317].rstrip() + "..."

        ctx_snip = (top1_tpl or "").strip()
        if len(ctx_snip) > 320:
            ctx_snip = ctx_snip[:317].rstrip() + "..."

        vote_txt = "unknown" if llm_vote is None else str(llm_vote)
        init_reason = llm_reason or "(none)"
        safety_note = guard_note or "(none)"

        user_content = (
            f"Log Template:\n{tpl_snip}\n\n"
            f"Top-1 normal template:\n{ctx_snip if ctx_snip else '(none)'}\n\n"
            f"Final decision (1=anomaly,0=normal): {final_pred}\n"
            f"Initial LLM vote (if any): {vote_txt}\n"
            f"Initial LLM note: {init_reason}\n"
            f"Safety note (for context only, avoid policy words): {safety_note}\n"
            f"Baseline probability: {prob:.6f}\n"
            "Return a JSON object with key `explanation` (≤ 40 English words) describing why the final decision matches the log semantics."
        )

        messages = [
            {"role": "system", "content": PROMPT_FINAL_REASON},
            {"role": "user", "content": user_content},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _run_final_reason(self, prompt_text: str) -> Optional[str]:
        try:
            js = self._run_llm(prompt_text)
        except Exception:
            return None
        if isinstance(js, dict):
            expl = js.get("explanation")
            if isinstance(expl, str):
                expl = expl.strip()
                if expl:
                    return expl
        return None

    def _run_llm(self, prompt_text: str) -> dict | None:
        out = self.gen(prompt_text)[0]["generated_text"]
        js = _extract_json(out)
        return js

    def _match_benign(self, text: str) -> bool:
        if not self.benign_regexes:
            return False
        t = text or ""
        for rx in self.benign_regexes:
            if rx.search(t):
                return True
        return False

    # ---- flip guard ----
    def _flip_guard(
        self,
        pred: int,
        prob: float,
        sim_top1: float,
        js: dict,
        log_tpl: str,
        top1_tpl: str
    ) -> Tuple[int, str, str, Optional[int]]:
        """
        Return: (final_label, guard_note, llm_reason)

        - 1→0 (false-positive correction)
          A) identical-autoflip (strict): top1 effectively identical & sim>=identical_sim & prob<prob_hi_identical
          B) (ignore reason) if LLM anomaly==0 AND sim_top1>=flip_sim_th AND prob<prob_hi
          C) optional relaxation if benign pattern matches

        - 0→1 (recover missed anomaly): default OFF, very conservative
          Conditions (all must hold):
            * enable_zero_to_one = True
            * js["anomaly"] == 1
            * prob >= prob0_min
            * sim_top1 <= low_sim_th
            * strong anomaly keyword exists in template
        """
        llm_reason = ""
        llm_vote: Optional[int] = None
        if isinstance(js, dict):
            llm_reason = _minify_reason(js.get("reason", ""), 12)
            llm_vote = _safe_int(js.get("anomaly"))

        # pred==1 : 1→0
        if pred == 1:
            # A) identical-autoflip
            if (
                self.enable_identical_autoflip
                and _same_template(log_tpl, top1_tpl)
                and sim_top1 >= self.identical_sim
                and float(prob) < self.prob_hi_identical
            ):
                return 0, "identical normal", llm_reason, llm_vote

            # If LLM says normal: relax only when high sim + low prob (reason string unused)
            if not isinstance(js, dict):
                # If LLM fails to parse → no flip
                return 1, "ParseError (keep 1)", llm_reason, llm_vote

            ia = js.get("anomaly", 1)
            benign_hit = self._match_benign(log_tpl) or self._match_benign(top1_tpl)

            if _is_strict_zero(ia) and float(prob) < self.prob_hi:
                if sim_top1 >= self.flip_sim_th:
                    return 0, "normal by LLM", llm_reason, llm_vote
                if benign_hit:
                    return 0, "normal (benign pattern)", llm_reason, llm_vote
                return 1, "guarded (keep 1)", llm_reason, llm_vote
            else:
                return 1, "guarded (keep 1)", llm_reason, llm_vote

        # pred==0 : 0→1 (optional)
        if pred == 0:
            if not self.enable_zero_to_one:
                return 0, "kept baseline (pred0)", llm_reason, llm_vote
            if not isinstance(js, dict):
                return 0, "ParseError (keep 0)", llm_reason, llm_vote

            ia = js.get("anomaly", 1)
            strong_kw = _has_anomaly_keywords(log_tpl)

            if (
                _is_strict_one(ia)
                and float(prob) >= self.prob0_min
                and sim_top1 <= self.low_sim_th
                and strong_kw
            ):
                return 1, "escalated by LLM", llm_reason, llm_vote
            else:
                return 0, "guarded (keep 0)", llm_reason, llm_vote

        # otherwise
        return pred, "unknown", llm_reason, llm_vote

    def _compose_final_reason(
        self,
        final_pred: int,
        guard_note: str,
        llm_reason: str,
        llm_vote: Optional[int],
        prob: float,
        llm_called: bool,
        log_tpl: str,
        top1_tpl: str,
    ) -> str:
        guard_note = (guard_note or "").strip()
        llm_reason = (llm_reason or "").strip()
        vote = _safe_int(llm_vote)

        template_text = (log_tpl or "").strip()
        top1_text = (top1_tpl or "").strip()

        final_explanation = None
        try:
            prompt = self._build_final_reason_prompt(
                template_text,
                top1_text,
                final_pred,
                guard_note,
                llm_reason,
                vote,
                prob,
            )
            final_explanation = self._run_final_reason(prompt)
        except Exception:
            final_explanation = None

        if final_explanation:
            return final_explanation

        tpl_snip = template_text
        if len(tpl_snip) > 160:
            tpl_snip = tpl_snip[:157].rstrip() + "..."

        if final_pred == 1:
            if llm_reason:
                return f"Log '{tpl_snip}' flagged anomalous: {llm_reason}"
            detail = guard_note or "anomaly retained"
            return f"Log '{tpl_snip}' kept anomalous ({detail})"

        # final_pred == 0 fallback
        if llm_reason:
            return f"Log '{tpl_snip}' judged normal: {llm_reason}"
        if guard_note == "identical normal":
            return f"Log '{tpl_snip}' identical to known normal template"
        if guard_note and "benign" in guard_note.lower():
            return f"Log '{tpl_snip}' matches benign pattern"
        if guard_note == "kept baseline (pred0)":
            return f"Log '{tpl_snip}' below threshold; treated as normal"
        if not llm_called:
            return f"Log '{tpl_snip}' treated as normal (LLM not called)"
        if guard_note.startswith("ParseError"):
            return f"Log '{tpl_snip}' treated as normal (LLM parse issue)"
        if guard_note:
            return f"Log '{tpl_snip}' treated as normal ({guard_note})"
        return f"Log '{tpl_snip}' treated as normal"

    # ---- run ----
    def run(
        self,
        pred_csv: str,
        window_csv: str | None,
        normal_csv: str,
        out_csv: str,
        raw_json: str | None,
        call_policy: str = "pred1",
        prob_lo: float = 0.35,
        prob_hi: float = 0.65,
        flips_csv: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        # 1) index normal templates
        normals = (
            pd.read_csv(normal_csv)["EventTemplate"]
            .dropna().astype(str).map(lambda s: " ".join(s.split()))
            .unique().tolist()
        )
        vdb = self._build_vdb(normals)

        # 2) load inputs
        pred_df = pd.read_csv(pred_csv)
        if "anomaly_pred" not in pred_df.columns and "is_anomaly_pred" in pred_df.columns:
            pred_df = pred_df.rename(columns={"is_anomaly_pred": "anomaly_pred"})
        if window_csv:
            w = pd.read_csv(window_csv)[["window_id", "EventTemplate"]]
            pred_df = pred_df.merge(w, on="window_id", how="left")

        if max_rows is not None:
            pred_df = pred_df.head(int(max_rows))

        # 3) call policy
        call_policy = call_policy.lower().strip()
        lo, hi = float(prob_lo), float(prob_hi)

        # 4) main loop
        rows = []
        raw = []
        flip_rows = []
        called = 0
        flips = 0
        it = tqdm(pred_df.itertuples(index=False), total=len(pred_df), desc="RAG_v3")

        for r in it:
            wid = int(r.window_id)
            base_pred = int(r.anomaly_pred)
            prob      = float(r.prob)
            tpl       = str(getattr(r, "EventTemplate", ""))

            # optional base_threshold override
            if self.base_threshold is not None:
                base_pred = 1 if prob >= float(self.base_threshold) else 0

            # decide whether to call LLM
            do_call = (
                (call_policy == "all") or
                (call_policy == "pred1"    and base_pred == 1) or
                (call_policy == "boundary" and (lo <= prob <= hi)) or
                (call_policy == "hybrid"   and (base_pred == 1 or (lo <= prob <= hi)))
            )

            sim_top1, ctx_top1 = 0.0, ""
            final_pred = base_pred
            guard_note = "kept baseline (pred0)" if base_pred == 0 else "guarded (keep 1)"
            llm_reason_raw = ""
            llm_vote = None
            llm_called = False

            js = None

            if do_call and tpl:
                # ① retrieve
                pairs = self._retrieve(vdb, tpl, self.top_k)
                ctx_lines = []
                if pairs:
                    for (t, s) in pairs:
                        # Hide numbers in prompt but keep originals for logging/guard
                        ctx_lines.append(f"{s:.3f} :: {t}")
                    sim_top1, ctx_top1 = float(pairs[0][1]), str(pairs[0][0])

                # ② chat prompt → LLM
                # Recommended: use strict when base_pred==1 (precision-first)
                strict_mode = (base_pred == 1)
                prompt_text = self._build_chat(tpl, ctx_lines, strict=strict_mode)
                js = self._run_llm(prompt_text)
                called += 1
                llm_called = True

                # ③ flip-guard
                fpred, fnote, lrsn, lvote = self._flip_guard(base_pred, prob, sim_top1, js, tpl, ctx_top1)
                final_pred, guard_note, llm_reason_raw, llm_vote = fpred, fnote, lrsn, lvote
                if final_pred != base_pred:
                    flips += 1
                    flip_rows.append({
                        "window_id": wid,
                        "before": base_pred,
                        "after": final_pred,
                        "prob": prob,
                        "sim_max": sim_top1,
                        "tpl": tpl,
                        "ctx_hit": ctx_top1,
                        "llm_reason": "",
                        "guard_note": guard_note,
                    })

                # raw logging
                raw.append({
                    "window_id": wid,
                    "result": json.dumps(js) if isinstance(js, dict) else str(js),
                    "sim": sim_top1,
                    "ctx": ctx_top1
                })

            else:
                # No-call: record reason cleanly
                guard_note = "kept baseline (pred0)" if base_pred == 0 else "guarded (keep 1)"

            final_reason = self._compose_final_reason(
                final_pred,
                guard_note,
                llm_reason_raw,
                llm_vote,
                prob,
                llm_called,
                tpl,
                ctx_top1,
            )
            final_reason = _minify_reason(final_reason, 40)

            if final_pred != base_pred and flip_rows:
                flip_rows[-1]["llm_reason"] = final_reason

            # ── save minimal columns ─────────────────────────────────────────
            rows.append({
                "window_id": wid,
                "EventTemplate": tpl,
                "llm_reason": final_reason,
                "anomaly": final_pred,
            })
            # ───────────────────────────────────────────────────────────

        # out.csv — save minimal columns in fixed order
        out_df = pd.DataFrame(rows, columns=["window_id", "EventTemplate", "llm_reason", "anomaly"])
        out_df.to_csv(out_csv, index=False)

        if raw_json is None:
            os.makedirs("output", exist_ok=True)
            raw_json = "output/rag_raw.json"
        with open(raw_json, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

        # flips csv (with header, even if flips=0)
        if flips_csv is None:
            flips_csv = "output/rag_flips.csv"
        pd.DataFrame(flip_rows, columns=[
            "window_id","before","after","prob","sim_max","tpl","ctx_hit","llm_reason","guard_note"
        ]).to_csv(flips_csv, index=False)

        print(f"[✓] saved: {out_csv}")
        print(f"[i] raw: {raw_json}")
        print(f"[i] flips: {flips} saved -> {flips_csv}")
        print(f"[i] call_policy={call_policy}  LLM-called: {called}  flip_sim_th={self.flip_sim_th}  prob_hi={self.prob_hi}  identical_auto={self.enable_identical_autoflip}")

# ============ CLI ============
def build_parser_with_defaults() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "RAG post-process with chat-formatted Llama-3 and strong flip-guard "
        "(defaults are embedded so you can run without args)"
    )

    # All required args now have defaults (optional).
    p.add_argument("--pred_csv",   default=DEFAULTS["pred_csv"])
    p.add_argument("--window_csv", default=DEFAULTS["window_csv"])
    p.add_argument("--normal_csv", default=DEFAULTS["normal_csv"])
    p.add_argument("--chroma_dir", default=DEFAULTS["chroma_dir"])
    p.add_argument("--llm_model",  default=DEFAULTS["llm_model"])
    p.add_argument("--top_k",      type=int, default=DEFAULTS["top_k"])
    p.add_argument("--device",     default=DEFAULTS["device"])

    # flip guard params (1→0 general)
    p.add_argument("--flip_sim_th",type=float, default=DEFAULTS["flip_sim_th"])
    p.add_argument("--prob_hi",    type=float, default=DEFAULTS["prob_hi"])

    # identical auto 1→0
    p.add_argument("--enable_identical_autoflip", dest="enable_identical_autoflip", action="store_true")
    p.add_argument("--no_identical_autoflip", dest="enable_identical_autoflip", action="store_false")
    p.set_defaults(enable_identical_autoflip=DEFAULTS["enable_identical_autoflip"])
    p.add_argument("--identical_sim", type=float, default=DEFAULTS["identical_sim"])
    p.add_argument("--prob_hi_identical", type=float, default=DEFAULTS["prob_hi_identical"])

    # call policy
    p.add_argument("--call_policy",choices=["all","pred1","boundary","hybrid"], default=DEFAULTS["call_policy"])
    p.add_argument("--prob_lo",    type=float, default=DEFAULTS["prob_lo"])
    p.add_argument("--prob_hi_b",  type=float, default=DEFAULTS["prob_hi_b"])

    # 0→1 correction option (default OFF, very conservative)
    p.add_argument("--enable_zero_to_one", dest="enable_zero_to_one", action="store_true")
    p.add_argument("--disable_zero_to_one", dest="enable_zero_to_one", action="store_false")
    p.set_defaults(enable_zero_to_one=DEFAULTS["enable_zero_to_one"])
    p.add_argument("--low_sim_th", type=float, default=DEFAULTS["low_sim_th"])
    p.add_argument("--prob0_min",  type=float, default=DEFAULTS["prob0_min"])

    # first-stage threshold override
    p.add_argument("--base_threshold", type=float, default=DEFAULTS["base_threshold"])

    # embedding replace/normalize
    p.add_argument("--embed_model", default=DEFAULTS["embed_model"])
    # normalize_embeddings default True → disable with --no_normalize_embeddings
    p.add_argument("--normalize_embeddings", dest="normalize_embeddings", action="store_true")
    p.add_argument("--no_normalize_embeddings", dest="normalize_embeddings", action="store_false")
    p.set_defaults(normalize_embeddings=DEFAULTS["normalize_embeddings"])

    # normal pattern whitelist
    p.add_argument("--benign_patterns", default=DEFAULTS["benign_patterns"])

    # compatibility (unused)
    p.add_argument("--prompt", default=None, help="(compat) unused")

    # outputs
    p.add_argument("--out_csv",   default=DEFAULTS["out_csv"])
    p.add_argument("--raw_json",  default=DEFAULTS["raw_json"])
    p.add_argument("--flips_csv", default=DEFAULTS["flips_csv"])
    p.add_argument("--max_rows", type=int, default=None,
                  help="Optional limit on number of windows to process (for testing)")
    return p

def sanity_check_files(args):
    missing = []
    for k in ["pred_csv","normal_csv"]:
        if not os.path.exists(getattr(args, k)):
            missing.append((k, getattr(args, k)))
    # window_csv is for merge; works without it but will warn if missing.
    if not os.path.exists(args.window_csv):
        print(f"[w] window_csv not found: {args.window_csv} (LLM EventTemplate may be missing in prompt)")
    if missing:
        print("[x] Required input file(s) not found:")
        for k, v in missing:
            print(f"    - {k}: {v}")
        sys.exit(1)

def main():
    p = build_parser_with_defaults()
    args = p.parse_args()

    print("[i] Effective config:")
    for k in sorted(DEFAULTS.keys()):
        v = getattr(args, k, None)
        print(f"    {k}: {v}")
    print()

    sanity_check_files(args)

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    rag = RAGPostProcessorV2(
        llm_name   = args.llm_model,
        top_k      = args.top_k,
        chroma_dir = args.chroma_dir,
        device     = args.device,
        flip_sim_th= args.flip_sim_th,
        prob_hi    = args.prob_hi,
        enable_identical_autoflip = args.enable_identical_autoflip,
        identical_sim = args.identical_sim,
        prob_hi_identical = args.prob_hi_identical,
        enable_zero_to_one = args.enable_zero_to_one,
        low_sim_th = args.low_sim_th,
        prob0_min  = args.prob0_min,
        base_threshold = args.base_threshold,
        embed_model = args.embed_model,
        normalize_embeddings = args.normalize_embeddings,
        benign_patterns_path = args.benign_patterns,
    )

    rag.run(
        pred_csv   = args.pred_csv,
        window_csv = args.window_csv,
        normal_csv = args.normal_csv,
        out_csv    = args.out_csv,
        raw_json   = args.raw_json,
        call_policy = args.call_policy,
        prob_lo    = args.prob_lo,
        prob_hi    = args.prob_hi_b,
        flips_csv  = args.flips_csv,
        max_rows   = args.max_rows,
    )

if __name__ == "__main__":
    main()
