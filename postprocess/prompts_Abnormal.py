# ────────── Recall-first common rules (ANOMALY VDB) ──────────
_COMMON_RULES_RECALL = r"""
OUTPUT CONTRACT
• Return exactly one fenced `json` code-block — no extra text before/after.
• The block must contain one minified JSON object with exactly two keys:
  - "anomaly": integer 0 or 1 only
  - "reason" : ≤ 12 ASCII words, inferred from the INPUT template semantics
• All keys/strings must be double-quoted (RFC 8259). No trailing commas. One line preferred.
• Treat every <*> placeholder as a wildcard; ignore it during reasoning and in "reason".
• Do not quote/copy any context line. Do not mention similarity/normal/anomaly/confidence in "reason".
"""

# ────────── Decision hints: recall-first (moderate) ──────────
_DECISION_HINTS_RECALL_MODERATE = r"""
DECISION RULES (recall-first, anomaly-only context)
• Use Top-k ANOMALY candidates only as evidence FOR anomaly; do not quote/copy them.
• If the input is a near-duplicate of any anomaly template (same structure/meaning; ignoring <*>), predict anomaly (1).
• Otherwise judge by template-level failure semantics (not single keywords).
• When semantics are ambiguous but still suggest failure states, prefer anomaly (1).
"""

# ────────── Decision hints: recall-first (strict) ──────────
_DECISION_HINTS_RECALL_STRICT = r"""
DECISION RULES (strict recall-first)
• Predict 1 only with strong failure semantics OR near-identity to a known anomaly template.
• Generic error words alone are insufficient without semantic alignment.
• In ties or doubt, still lean to anomaly (1) given the recall-first goal.
"""

# ────────── Final prompt: recall-first (moderate) ──────────
PROMPT_RECALL_MODERATE = (
    "You are an Android system-log expert.\n"
    "Goal: recall-first (minimize false negatives).\n"
    "Decide normal (0) vs anomaly (1) for the given log TEMPLATE using Top-{top_k} ANOMALY templates as context.\n"
    "\n"
    "Log Template:\n"
    "{question}\n"
    "\n"
    "Top-{top_k} ANOMALY candidates (cosine):\n"
    "{context}\n"
    "\n"
    f"{_DECISION_HINTS_RECALL_MODERATE}\n"
    f"{_COMMON_RULES_RECALL}"
)

# ────────── Final prompt: recall-first (strict) ──────────
PROMPT_RECALL_STRICT = (
    "You are an Android system-log expert.\n"
    "Goal: recall-first (minimize false negatives).\n"
    "Decide normal (0) vs anomaly (1) for the given log TEMPLATE using Top-{top_k} ANOMALY templates as context.\n"
    "\n"
    "Log Template:\n"
    "{question}\n"
    "\n"
    "Top-{top_k} ANOMALY candidates (cosine):\n"
    "{context}\n"
    "\n"
    f"{_DECISION_HINTS_RECALL_STRICT}\n"
    f"{_COMMON_RULES_RECALL}"
)

__all__ = [
    "PROMPT_RECALL_MODERATE",
    "PROMPT_RECALL_STRICT",
]
