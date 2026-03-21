from __future__ import annotations

import re
from typing import Any

_DETECTOR = None

# Lightweight keyword-based PHI detection.
# (We keep it intentionally simple so it runs locally without scispaCy.)
_HEALTH_KEYWORDS = {
    # conditions
    "diabetes",
    "depression",
    "anxiety",
    "asthma",
    "cancer",
    "hypertension",
    "migraine",
    "insomnia",
    "epilepsy",
    # symptoms
    "pain",
    "fever",
    "cough",
    "shortness of breath",
    "nausea",
    # medications/treatments
    "insulin",
    "chemotherapy",
    "radiation",
    "surgery",
    "medication",
    "antibiotics",
}

_KEYWORD_PATTERNS: list[tuple[re.Pattern[str], str]] = []


def _build_keyword_patterns() -> None:
    global _KEYWORD_PATTERNS
    if _KEYWORD_PATTERNS:
        return

    patterns: list[tuple[re.Pattern[str], str]] = []
    # Prefer longer phrases first to reduce partial matches.
    keywords_sorted = sorted(_HEALTH_KEYWORDS, key=lambda k: (-len(k), k))
    for kw in keywords_sorted:
        # Escape keyword and allow simple word boundaries.
        escaped = re.escape(kw)
        pat = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
        # Replace everything with a coarse placeholder.
        patterns.append((pat, "[REDACTED HEALTH_INFO]"))
    _KEYWORD_PATTERNS = patterns


class HealthDetector:
    def detect_and_redact(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        _build_keyword_patterns()

        spans: list[tuple[int, int, str, str, str]] = []
        for pat, replacement in _KEYWORD_PATTERNS:
            for m in pat.finditer(text):
                spans.append((m.start(), m.end(), replacement, m.group(), "HEALTH"))

        spans.sort(key=lambda x: (x[0], x[1] - x[0]))

        redacted_parts: list[str] = []
        sanitized: list[dict[str, Any]] = []
        last_idx = 0

        for start, end, replacement, original, typ in spans:
            if start < last_idx:
                continue
            redacted_parts.append(text[last_idx:start])
            redacted_parts.append(replacement)
            last_idx = end
            sanitized.append({"original": original, "type": typ})

        redacted_parts.append(text[last_idx:])
        return "".join(redacted_parts), sanitized


def _get_detector() -> HealthDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = HealthDetector()
    return _DETECTOR


def make_candidates_health(text: str) -> list[str]:
    detector = _get_detector()
    redacted_text, _ = detector.detect_and_redact(text)
    return [redacted_text]

