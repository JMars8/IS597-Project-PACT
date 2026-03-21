"""
Shared prompt building + parsing for local Llama privacy synthesis.
Used by backend/server.py and scripts/test_local_llama.py (no GPT_API_KEY required).
"""

from __future__ import annotations

import json
import re


def build_privacy_synthesis_prompt(
    original_query: str,
    candidates: list[str],
    privacy_preferences: dict,
) -> str:
    """
    Candidates-first ordering: models latch onto the first block; treat that list as the
    authority for what wording (including placeholders) is allowed in the final line.
    """
    return (
        "You produce one line that will be sent to a cloud assistant.\n\n"
        "Authority order (highest first):\n"
        "1) The `candidates` strings — they already apply privacy modules. Anything they "
        "generalize, redact, or replace (e.g. placeholders) is how the final line must read.\n"
        "2) `original_query` — use only for overall intent and tone, without contradicting "
        "the candidates. If anything in the original disagrees with a candidate string on a "
        "fact, name, number, or placeholder, follow the candidates.\n"
        "3) `enabled module flags` — hint which categories were on; stay consistent with (1).\n\n"
        "Output: exactly one line, same language as the user when natural. No labels, markdown, "
        "code fences, or extra sentences.\n\n"
        "candidates:\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "original_query (intent only; do not resurrect details missing from candidates):\n"
        f"{original_query}\n\n"
        "enabled module flags:\n"
        f"{json.dumps(privacy_preferences, ensure_ascii=False)}"
    )


def extract_final_prompt(hf_text: str) -> str:
    """
    Parse model output after privacy synthesis.

    The server prompt ends with ``final_prompt:`` (no trailing space), so the model
    often continues with the value only — no ``final_prompt:`` in the generated
    string. Models also ramble into ``` fences; we keep one line per spec.
    """
    txt = (hf_text or "").strip()
    if not txt:
        return ""
    # Never keep markdown / code the model was told not to emit.
    txt = txt.split("```")[0].strip()

    match = re.search(
        r"final_prompt\s*:\s*(.+?)(?:\r?\n|$)",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        inner = match.group(1).strip()
        if inner:
            return inner.splitlines()[0].strip()

    # Continuation-only: first non-empty line is the rewritten prompt.
    for line in txt.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


_SYNTHESIS_UNUSABLE_RE = re.compile(
    r"(incomplete|cut off|truncat|could you (please )?clarif|provide (more )?context|"
    r"not sure what you|unclear|didn'?t understand|as an ai|i (can|cannot) '?t help)",
    re.IGNORECASE,
)


def is_synthesis_unusable(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    return bool(_SYNTHESIS_UNUSABLE_RE.search(s))
