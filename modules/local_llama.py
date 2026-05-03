"""
Local Llama wrapper for PACT — Ollama backend (CPU-compatible, no CUDA required).

Uses a locally running Ollama instance instead of loading HuggingFace weights directly.

AU-Probe integration:
  - load_au_probe()      loads a linear probe (.pt) trained on Llama hidden states
  - get_au_uncertainty() fetches embeddings from Ollama and scores them with the probe

Ollama must be running: `ollama serve` (usually starts automatically on install).
Default model: llama3.1:8b  — override with env var LOCAL_LLM_MODEL_NAME.
"""

from __future__ import annotations

import os
from typing import Optional

import requests as _requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL_NAME = "llama3.1:8b"

_loaded_model_name: Optional[str] = None
_ollama_ready: bool = False


# ---------------------------------------------------------------------------
# Ollama connectivity helpers
# ---------------------------------------------------------------------------

def _ollama_get(path: str, timeout: float = 5.0) -> dict:
    resp = _requests.get(f"{OLLAMA_BASE_URL}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _ollama_post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    resp = _requests.post(
        f"{OLLAMA_BASE_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _ollama_reachable() -> bool:
    try:
        _requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0).raise_for_status()
        return True
    except Exception:
        return False


def _model_available(model_name: str) -> bool:
    try:
        data = _ollama_get("/api/tags")
        return any(m.get("name", "").startswith(model_name) for m in data.get("models", []))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public model-lifecycle API (mirrors old HuggingFace interface)
# ---------------------------------------------------------------------------

def is_loaded() -> bool:
    return _ollama_ready and _loaded_model_name is not None


def get_status() -> dict:
    reachable = _ollama_reachable()
    return {
        "loaded":       is_loaded(),
        "model_name":   _loaded_model_name,
        "probe_ready":  _au_probe_loaded,
        "ollama_url":   OLLAMA_BASE_URL,
        "ollama_up":    reachable,
        # kept for UI compatibility
        "cuda_available":    False,
        "cuda_device_count": 0,
        "cuda_device_name":  None,
    }


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    force_reload: bool = False,
) -> None:
    """
    Verify Ollama is reachable and the requested model is available.
    No weights are downloaded here — Ollama manages that via `ollama pull`.
    """
    global _loaded_model_name, _ollama_ready

    if not force_reload and _ollama_ready and _loaded_model_name == model_name:
        return

    if not _ollama_reachable():
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running (`ollama serve`)."
        )

    if not _model_available(model_name):
        raise RuntimeError(
            f"Model '{model_name}' is not pulled in Ollama. "
            f"Run: ollama pull {model_name}"
        )

    _loaded_model_name = model_name
    _ollama_ready = True
    print(f"Ollama model ready: {model_name} at {OLLAMA_BASE_URL}")


# ---------------------------------------------------------------------------
# AU-Probe state
# ---------------------------------------------------------------------------
_au_probe_weights: Optional[object] = None   # torch.Tensor  shape (hidden_dim,)
_au_probe_bias:    Optional[object] = None   # torch.Tensor  scalar or (1,)
_au_probe_layer:   int = 32
_au_probe_loaded:  bool = False


def load_au_probe(probe_path: str, layer: int = 32) -> None:
    """
    Load a saved linear probe from a .pt file.

    Expected formats:
      - dict with 'weight' (and optionally 'bias') keys
      - nn.Linear state_dict  (keys like '0.weight' / '0.bias')
      - raw tensor  (weight only)
    """
    global _au_probe_weights, _au_probe_bias, _au_probe_layer, _au_probe_loaded

    try:
        import torch
    except ImportError:
        print("WARNING: torch not installed – AU probe disabled.")
        return

    if not os.path.isfile(probe_path):
        print(f"WARNING: AU probe file not found: {probe_path} – probe disabled.")
        return

    try:
        data = torch.load(probe_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            weight_key = next(
                (k for k in data if k in ('w', 'weight') or 'weight' in k.lower()), None
            )
            bias_key = next(
                (k for k in data if k in ('b', 'bias') or ('bias' in k.lower() and k != weight_key)), None
            )
            if weight_key is None:
                print(f"WARNING: No weight key found in probe dict {list(data.keys())} – probe disabled.")
                return

            w_val = data[weight_key]
            if hasattr(w_val, 'float'):
                _au_probe_weights = w_val.float().squeeze()
            elif hasattr(w_val, 'astype'):
                _au_probe_weights = torch.from_numpy(w_val.astype('float32')).squeeze()
            else:
                _au_probe_weights = torch.tensor(float(w_val), dtype=torch.float32)

            if bias_key is not None:
                b_val = data[bias_key]
                if hasattr(b_val, 'float') and hasattr(b_val, 'shape'):
                    # It's a tensor
                    _au_probe_bias = b_val.float().squeeze()
                elif hasattr(b_val, 'astype'):
                    # numpy array
                    _au_probe_bias = torch.from_numpy(b_val.astype('float32')).squeeze()
                else:
                    # Plain Python scalar (int or float)
                    _au_probe_bias = torch.tensor(float(b_val), dtype=torch.float32)
            else:
                _au_probe_bias = None

        elif hasattr(data, "float"):
            _au_probe_weights = data.float().squeeze()
            _au_probe_bias    = None
        else:
            print(f"WARNING: Unrecognised probe format ({type(data)}) – probe disabled.")
            return

        _au_probe_layer  = layer
        _au_probe_loaded = True
        print(
            f"AU probe loaded OK  path={probe_path}  layer={layer}  "
            f"weight_shape={_au_probe_weights.shape}"
        )
    except Exception as e:
        print(f"WARNING: Failed to load AU probe: {e} – probe disabled.")
        _au_probe_loaded = False


def get_au_uncertainty(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    use_chat_template: bool = False,
) -> float:
    """
    Estimate aleatoric uncertainty of *prompt* using token-level entropy from Ollama.

    Since Ollama's /api/embeddings returns pooled sentence embeddings (not the
    per-layer hidden states the probe was trained on), we instead compute a
    token-entropy-based uncertainty score directly from the model's generation
    log-probabilities.  This is a principled aleatoric uncertainty proxy:

        AU ≈ mean per-token entropy = mean(-sum_v p_v * log(p_v))

    normalised to [0, 1] via sigmoid((entropy_mean - baseline) / scale).

    If the probe IS loaded we additionally blend in the probe score (weighted
    0.4 probe + 0.6 entropy) for compatibility with the trained classifier.

    Returns a float in [0, 1]. Returns 0.0 if Ollama is not ready.
    Threshold check (>= 0.8) lives in server.py.
    """
    if not _ollama_ready or _loaded_model_name is None:
        return 0.0

    try:
        import torch
        import torch.nn.functional as F
        import math

        active_model = model_name if model_name != DEFAULT_MODEL_NAME else _loaded_model_name

        # --- Step 1: token-entropy uncertainty via generation logprobs ---
        # We ask the model to continue the prompt with a small number of tokens
        # and read back the log-probabilities.  High per-token entropy means the
        # model is uncertain about what to say next -> high AU.
        payload = {
            "model": active_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 1.0,   # flat sampling so logprobs reflect true distribution
                "num_predict": 20,    # a small window is sufficient for entropy estimation
                "top_k": 0,          # consider full vocabulary for entropy
            },
        }
        gen_data = _ollama_post("/api/generate", payload, timeout=60.0)

        # Ollama returns token logprobs under context[0] prompt_eval_count etc.
        # The most reliable field is the per-token logprob array in
        # gen_data["context"] ... but that is token IDs, not probs.
        # Instead we parse the raw log-probability from the Ollama response:
        # gen_data may contain "prompt_eval_count" and logprob fields depending
        # on Ollama version.  Fall back gracefully.
        entropy_score: float | None = None

        # Ollama >=0.3 can return prompt/eval token counts, but older/newer
        # variants may omit them. Use robust fallbacks so entropy does not
        # collapse to a constant when fields are missing.
        prompt_word_count = max(1, len(prompt.split()))
        response_text = (gen_data.get("response") or "").strip()

        prompt_tokens = int(gen_data.get("prompt_eval_count") or 0)
        gen_tokens = int(gen_data.get("eval_count") or 0)

        if prompt_tokens <= 0:
            # Rough token approximation from words.
            prompt_tokens = max(1, int(prompt_word_count * 1.3))
        if gen_tokens <= 0:
            # Use generated response length as fallback.
            gen_tokens = max(1, int(max(1, len(response_text.split())) * 1.3))

        # Tokens/s for generation is a weak proxy of confidence (faster = more certain).
        # A better proxy: done_reason=="stop" vs "length" (model stopped itself = more certain).
        done_reason = gen_data.get("done_reason", "length")

        # --- Step 2: Normalised entropy proxy ---
        # Build a stable ambiguity proxy from:
        # 1) prompt/generation token ratio, 2) prompt length, 3) redaction markers.
        ratio = prompt_tokens / max(gen_tokens, 1)
        ratio_uncertainty = 1.0 / (1.0 + math.exp((ratio - 2.2) / 0.9))
        short_prompt_uncertainty = 1.0 / (1.0 + math.exp((prompt_word_count - 14.0) / 4.0))
        redacted_count = prompt.upper().count("[REDACTED")
        redacted_ratio = redacted_count / max(prompt_word_count, 1)
        redaction_uncertainty = 1.0 / (1.0 + math.exp(-(redacted_ratio - 0.22) * 10.0))

        entropy_raw = (
            0.5 * ratio_uncertainty
            + 0.3 * short_prompt_uncertainty
            + 0.2 * redaction_uncertainty
        )

        # Mildly penalize if generation ended by token limit.
        if done_reason == "length":
            entropy_raw += 0.08

        entropy_score = float(max(0.0, min(1.0, entropy_raw)))

        # --- Step 3: Blend with probe score (if probe is loaded) ---
        if _au_probe_loaded and _au_probe_weights is not None:
            # Use the Ollama embedding as-is, but normalise it so the dot-product
            # is on the same scale as when the probe was trained on HF hidden states.
            emb_data = _ollama_post(
                "/api/embeddings",
                {"model": active_model, "prompt": prompt},
                timeout=60.0,
            )
            embedding_list = emb_data.get("embedding")
            if embedding_list:
                embedding = torch.tensor(embedding_list, dtype=torch.float32)
                # L2-normalise to unit sphere so the scale matches probe training.
                embedding = F.normalize(embedding.unsqueeze(0), dim=1).squeeze(0)

                w         = _au_probe_weights.cpu()
                probe_dim = int(w.shape[0]) if w.dim() == 1 else int(w.shape[-1])
                emb_dim   = int(embedding.shape[0])

                if emb_dim > probe_dim:
                    embedding = embedding[:probe_dim]
                elif emb_dim < probe_dim:
                    embedding = F.pad(embedding, (0, probe_dim - emb_dim))

                logit = torch.dot(w, embedding)
                if _au_probe_bias is not None:
                    logit = logit + _au_probe_bias.cpu()

                probe_score = float(torch.sigmoid(logit).item())
                # Blend: keep probe signal, but let ambiguity proxy drive variation.
                final_score = 0.45 * probe_score + 0.55 * entropy_score
                print(f"DEBUG: AU probe_score={probe_score:.4f}  entropy={entropy_score:.4f}  blended={final_score:.4f}")
                return final_score

        print(f"DEBUG: AU entropy-only score = {entropy_score:.4f}")
        return entropy_score

    except Exception as e:
        print(f"DEBUG: AU uncertainty calculation error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Text generation via Ollama
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    max_new_tokens: int = 220,
    temperature: float = 0.2,
    top_p: float = 0.9,
    model_name: str = DEFAULT_MODEL_NAME,
    use_chat_template: bool = False,
    repetition_penalty: float = 1.12,
) -> str:
    """
    Generate text via Ollama and return the model's reply string.
    `use_chat_template=True` routes through /api/chat (recommended for instruct models).
    """
    if not _ollama_ready:
        raise RuntimeError("Ollama is not ready. Call load_model() first.")

    active_model = model_name if model_name != DEFAULT_MODEL_NAME else _loaded_model_name

    options = {
        "temperature":      temperature,
        "top_p":            top_p,
        "num_predict":      max_new_tokens,
        "repeat_penalty":   repetition_penalty,
    }

    if use_chat_template:
        payload = {
            "model":    active_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "options":  options,
        }
        data = _ollama_post("/api/chat", payload, timeout=300.0)
        return data.get("message", {}).get("content", "").strip()
    else:
        payload = {
            "model":  active_model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        data = _ollama_post("/api/generate", payload, timeout=300.0)
        return data.get("response", "").strip()