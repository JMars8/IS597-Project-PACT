"""
Build privacy-module candidates for local Llama synthesis (same logic as the API).

Kept separate from backend/server.py so scripts can run without GPT_API_KEY.
"""

from __future__ import annotations

from typing import Any, Mapping

from modules.financial_detector import FinancialDetector
from modules import identity_module
from modules import demographic_module
from modules import health_module
from modules import modules_geo

_financial_detector = FinancialDetector()


def collect_pipeline_inputs(
    original_query: str,
    settings: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]], str | None]:
    """
    Run enabled privacy modules; return merged candidates, per-module outputs, financial text.

    `settings` must include booleans: identity, location, demographic, health, financial.
    """
    module_masks: dict[str, list[str]] = {
        "identity": [],
        "location": [],
        "demographic": [],
        "health": [],
        "financial": [],
    }
    candidates: list[str] = []
    financial_candidate: str | None = None

    if settings.get("identity"):
        c = identity_module.make_candidates_identity(original_query)
        module_masks["identity"] = list(c)
        candidates.extend(c)

    if settings.get("location"):
        c = modules_geo.make_candidates_location(original_query)
        module_masks["location"] = list(c)
        candidates.extend(c)

    if settings.get("demographic"):
        c = demographic_module.make_candidates_demographic(original_query)
        module_masks["demographic"] = list(c)
        candidates.extend(c)

    if settings.get("health"):
        c = health_module.make_candidates_health(original_query)
        module_masks["health"] = list(c)
        candidates.extend(c)

    if settings.get("financial"):
        processed_query, _ = _financial_detector.detect_and_redact(original_query)
        financial_candidate = processed_query
        module_masks["financial"] = [processed_query]
        candidates.append(processed_query)

    candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
    if not candidates:
        candidates = [original_query]

    return candidates, module_masks, financial_candidate
