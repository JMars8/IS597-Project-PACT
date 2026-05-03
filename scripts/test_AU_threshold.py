#!/usr/bin/env python3
"""
Evaluate AU uncertainty across prompts with different ambiguity levels.

Run from repo root:
  python scripts/test_AU_threshold.py
  python scripts/test_AU_threshold.py --model llama3.1:8b --thresholds 0.2,0.4,0.6,0.8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import local_llama  # noqa: E402


PROMPTS: list[tuple[str, str]] = [
    (
        "Very clear",
        "I am a Korean student living in Illinois, USA. I have a card number 3132 1234 4545 2134 and I want to know how to deactivate this card safely.",
    ),
    (
        "Clear",
        "I am from South Korea and currently live in Illinois. My bank card ending in 2134 was lost, and I need the exact steps to block and replace it.",
    ),
    (
        "Mostly clear",
        "I am an international student in the U.S. with a debit card. I want to cancel this card and request a new one without affecting my account.",
    ),
    (
        "Moderate",
        "I am from an Asian country and living in one of the states in America. I currently have [CARD_NUMBER] and want to know how to cancel this card.",
    ),
    (
        "Moderate",
        "I live in the U.S. now and have a card [REDACTED_FINANCIAL]. Please tell me how to deactivate it and what to do next.",
    ),
    (
        "Somewhat ambiguous",
        "I have [REDACTED_IDENTITY] in [REDACTED_LOCATION] with card [REDACTED_FINANCIAL], and I need help stopping this card.",
    ),
    (
        "Ambiguous",
        "I am [REDACTED_DEMOGRAPHIC] living in [REDACTED_LOCATION], and I need to handle my card issue [REDACTED_FINANCIAL].",
    ),
    (
        "More ambiguous",
        "I am [REDACTED_IDENTITY] from [REDACTED_LOCATION], and I need help with my payment card.",
    ),
    (
        "Highly ambiguous",
        "I have [REDACTED_FINANCIAL]. What should I do?",
    ),
    (
        "Extremely ambiguous",
        "How do I handle [REDACTED]?",
    ),
]


def parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"Threshold must be in [0,1], got {val}")
        values.append(val)
    if not values:
        raise ValueError("No thresholds provided.")
    return values


def yn(flag: bool) -> str:
    return "UNCERTAIN" if flag else "certain"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AU threshold checks on prompts with varying ambiguity."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LOCAL_LLM_MODEL_NAME", local_llama.DEFAULT_MODEL_NAME),
        help="Ollama model name (default: env LOCAL_LLM_MODEL_NAME or local_llama default).",
    )
    parser.add_argument(
        "--thresholds",
        default="0.2,0.4,0.6,0.8",
        help="Comma-separated AU thresholds in [0,1], e.g. 0.2,0.4,0.6,0.8",
    )
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    model_name = args.model

    probe_path = Path(ROOT) / "data" / "au_probe" / "linearprobe_layer_32.pt"
    if not probe_path.exists():
        raise FileNotFoundError(f"AU probe file not found: {probe_path}")

    print("=" * 96)
    print(f"AU Threshold Evaluation | model={model_name} | thresholds={thresholds}")
    print("=" * 96)

    local_llama.load_model(model_name=model_name)
    local_llama.load_au_probe(str(probe_path.resolve()), layer=32)

    header_cols = ["#", "Ambiguity", "Prompt", "AU Score"] + [f"th={t:.2f}" for t in thresholds]
    print(" | ".join(header_cols))
    print("-" * 96)

    for idx, (label, prompt) in enumerate(PROMPTS, start=1):
        score = local_llama.get_au_uncertainty(prompt, model_name=model_name)
        decisions = [yn(score >= t) for t in thresholds]
        prompt_preview = prompt if len(prompt) <= 56 else prompt[:53] + "..."
        row = [
            str(idx),
            label,
            prompt_preview,
            f"{score:.4f}",
            *decisions,
        ]
        print(" | ".join(row))

    print("-" * 96)
    print("Interpretation: score >= threshold => UNCERTAIN, else certain.")


if __name__ == "__main__":
    main()

