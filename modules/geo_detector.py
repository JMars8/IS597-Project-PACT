"""
Geo + US street-address detection using spaCy.

- General geography: ``en_core_web_sm`` labels GPE, LOC, FAC.
- US addresses: ``en-us-address-ner-sm`` from NickCrews/spacy-address
  (https://github.com/NickCrews/spacy-address). Install the release wheel; then
  ``spacy.load("en-us-address-ner-sm")`` works. There is no PyPI package named
  ``spacy-address``—the model wheel is the supported install path.
"""

from __future__ import annotations

import warnings

import spacy


class GeoDetector:
    """
    Detect and redact geographic entities and US postal-style addresses.

    Uses:
    - ``en_core_web_sm`` for GPE, LOC, FAC.
    - ``en-us-address-ner-sm`` (spacy-address project) for fine-grained US
      address NER; contiguous spans are merged and redacted as one block.
    """

    def __init__(self):
        """Load geo NER and optionally the US address model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import os

            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self._geo_labels = {"GPE", "LOC", "FAC"}

        self.nlp_address: spacy.language.Language | None = None
        try:
            self.nlp_address = spacy.load("en-us-address-ner-sm")
        except OSError:
            warnings.warn(
                "US address redaction is disabled: could not load "
                "'en-us-address-ner-sm'. Install the wheel from "
                "NickCrews/spacy-address (see requirements.txt). "
                "Geo labels (GPE/LOC/FAC) still work.",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _merge_address_spans(doc) -> list[tuple[int, int]]:
        """Merge adjacent/overlapping address entities into character spans."""
        ents = [e for e in doc.ents if e.label_ != "NotAddress"]
        if not ents:
            return []

        ents = sorted(ents, key=lambda e: e.start_char)
        merged: list[tuple[int, int]] = []
        cur_start, cur_end = ents[0].start_char, ents[0].end_char

        for ent in ents[1:]:
            # Allow small gaps (e.g. ", ") between tagged components
            if ent.start_char <= cur_end + 2:
                cur_end = max(cur_end, ent.end_char)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = ent.start_char, ent.end_char

        merged.append((cur_start, cur_end))
        return merged

    def detect_and_redact(self, text: str):
        """
        Redact US address blocks first (on original text), then GPE/LOC/FAC.

        Args:
            text: Raw input text.

        Returns:
            (redacted_text, sanitizations) where each log has
            ``{"original": str, "type": str}`` (type is ADDRESS or a spaCy geo label).
        """
        redacted_text = text
        sanitizations: list[dict] = []
        seen: set[tuple[str, str]] = set()

        # 1) US addresses (spacy-address model) — right-to-left so indices stay valid
        if self.nlp_address is not None:
            addr_doc = self.nlp_address(text)
            for start, end in sorted(
                self._merge_address_spans(addr_doc),
                key=lambda span: span[0],
                reverse=True,
            ):
                original = redacted_text[start:end].strip()
                if not original:
                    continue
                key = (original, "ADDRESS")
                if key in seen:
                    continue
                seen.add(key)
                sanitizations.append({"original": original, "type": "ADDRESS"})
                redacted_text = (
                    redacted_text[:start] + "[REDACTED ADDRESS]" + redacted_text[end:]
                )

        # 2) GPE / LOC / FAC on text after address redaction
        doc = self.nlp(redacted_text)
        for ent in doc.ents:
            if ent.label_ not in self._geo_labels:
                continue
            original = ent.text
            key = (original, ent.label_)
            if key in seen:
                continue
            seen.add(key)
            sanitizations.append({"original": original, "type": ent.label_})
            redacted_text = redacted_text.replace(original, "[REDACTED GEO]")

        return redacted_text, sanitizations


if __name__ == "__main__":
    detector = GeoDetector()
    sample = (
        "Ship to 123 E Main St, Oklahoma City, OK 73102 — "
        "I also like Paris and Central Park."
    )
    redacted, logs = detector.detect_and_redact(sample)
    print(f"Original: {sample}")
    print(f"Redacted: {redacted}")
    print(f"Logs: {logs}")
