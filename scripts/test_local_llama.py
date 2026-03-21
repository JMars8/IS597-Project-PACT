#!/usr/bin/env python3
"""
Exercise Local Llama without the FastAPI server or GPT_API_KEY.

Run from repo root (recommended, same cwd as backend):
  python scripts/test_local_llama.py
  python scripts/test_local_llama.py --case synthesis
  python scripts/test_local_llama.py --case all --max-new-tokens 256

Uses modules.synthesis_prompt and modules.pipeline_collect (same candidate path as the API).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Repo root (parent of scripts/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import local_llama  # noqa: E402
from modules.pipeline_collect import collect_pipeline_inputs  # noqa: E402
from modules.synthesis_prompt import (  # noqa: E402
    build_privacy_synthesis_prompt,
    extract_final_prompt,
    is_synthesis_unusable,
)


def _divider(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def case_simple(model_name: str, max_new_tokens: int, use_chat_template: bool) -> None:
    prompt = "Answer in one short sentence. What is 17 + 25?"
    _divider("CASE: simple completion (no PACT template)")
    print("PROMPT:\n", prompt[:500], "\n" if len(prompt) > 500 else "")
    out = local_llama.generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        model_name=model_name,
        use_chat_template=use_chat_template,
    )
    print("RAW OUTPUT:\n", repr(out))


def case_synthesis(
    model_name: str,
    max_new_tokens: int,
    use_chat_template: bool,
    query: str,
    prefs: dict,
) -> None:
    original = query.strip()
    candidates, module_masks, financial_candidate = collect_pipeline_inputs(
        original, prefs
    )
    _divider("CASE: PACT synthesis (real modules → candidates → Llama)")
    print("privacy toggles:", prefs)
    print("\nmodule_masks (per detector):")
    for name, items in module_masks.items():
        print(f"  [{name}] ({len(items)} strings)")
        for i, s in enumerate(items):
            preview = (s[:200] + "…") if len(s) > 200 else s
            print(f"    ({i + 1}) {preview!r}")
    print("\nfinancial_candidate (if financial on):", repr(financial_candidate))
    print("\ncandidates_for_llama (merged list, len=%d):" % len(candidates))
    for i, s in enumerate(candidates):
        preview = (s[:220] + "…") if len(s) > 220 else s
        print(f"  ({i + 1}) {preview!r}")

    prompt = build_privacy_synthesis_prompt(original, candidates, prefs)
    print("\nSYNTHESIS PROMPT (tail 600 chars):\n", prompt[-600:])
    out = local_llama.generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        model_name=model_name,
        use_chat_template=use_chat_template,
    )
    extracted = extract_final_prompt(out)
    print("\nRAW MODEL OUTPUT:\n", repr(out))
    print("\nextract_final_prompt() ->\n", repr(extracted))
    print("is_synthesis_unusable ->", is_synthesis_unusable(extracted))




def case_modules_only(query: str, prefs: dict) -> None:
    """Run spaCy/detector pipeline only; no GPU Llama (fast sanity check)."""
    original = query.strip()
    candidates, module_masks, financial_candidate = collect_pipeline_inputs(
        original, prefs
    )
    _divider("CASE: modules only (candidates, no Llama)")
    print("privacy toggles:", prefs)
    for name, items in module_masks.items():
        print(f"\n[{name}]")
        for i, s in enumerate(items):
            print(f"  ({i + 1}) {s!r}")
    print("\nfinancial_candidate:", repr(financial_candidate))
    print("merged candidates count:", len(candidates))


CASE_HANDLERS = {
    "simple": case_simple,
    "synthesis": case_synthesis,
    "modules": case_modules_only,
}

ALL_LLAMA_CASES = ["simple", "synthesis"]

DEFAULT_SYNTHESIS_QUERY = (
    "I am a Korean student in Illinois. I want to transfer $500 from my Korean "
    "bank account to my US credit card ending in 4233. How do I do this?"
)

DEFAULT_PRIVACY_PREFS: dict = {
    "identity": True,
    "location": True,
    "demographic": True,
    "health": False,
    "financial": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Local Llama in isolation.")
    parser.add_argument(
        "--case",
        choices=["all", "modules", *sorted(k for k in CASE_HANDLERS if k != "modules")],
        default="all",
        help="Use 'modules' for detector output only (no GPU Llama load).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation cap (default 128; server synthesis uses 128).",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Raw completion instead of Instruct chat template.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "LOCAL_LLM_MODEL_NAME", local_llama.DEFAULT_MODEL_NAME
        ),
        help="HF model id (default: env LOCAL_LLM_MODEL_NAME or Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_SYNTHESIS_QUERY,
        help="User text for synthesis / modules cases.",
    )
    parser.add_argument(
        "--settings-json",
        default="",
        help='Override privacy toggles, e.g. {"identity":true,"location":true,"demographic":true,"health":false,"financial":true}',
    )
    args = parser.parse_args()

    if args.settings_json.strip():
        prefs = json.loads(args.settings_json)
    else:
        prefs = dict(DEFAULT_PRIVACY_PREFS)
    for key in ("identity", "location", "demographic", "health", "financial"):
        prefs.setdefault(key, False)

    if args.case == "modules":
        case_modules_only(args.query, prefs)
        _divider("Done")
        return

    print("Model:", args.model)
    print("Loading (first run may download weights; CUDA required)...")
    local_llama.load_model(model_name=args.model)
    print("Loaded. CUDA / device info:", local_llama.get_status())

    use_ct = not args.no_chat_template
    to_run = list(ALL_LLAMA_CASES) if args.case == "all" else [args.case]
    for name in to_run:
        if name == "synthesis":
            CASE_HANDLERS[name](
                args.model, args.max_new_tokens, use_ct, args.query, prefs
            )
        else:
            CASE_HANDLERS[name](args.model, args.max_new_tokens, use_ct)

    _divider("Done")


if __name__ == "__main__":
    main()
