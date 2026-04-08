"""
Local Llama wrapper for PACT.

This module is intentionally self-contained inside the PACT repo.
It loads a Hugging Face causal LM locally (GPU if available) and generates
text from a prompt.

Authentication for gated models:
- Put a Hugging Face token in `hg.txt` (or `hg`) either in:
  - the repo root (where you run `backend/server.py`), or
  - this module's directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_tokenizer = None
_model = None
_loaded_model_name: Optional[str] = None

# AU Probe globals
_au_probe_w = None
_au_probe_b = None
_au_probe_layer = None

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def is_loaded() -> bool:
    return _model is not None and _loaded_model_name is not None


def get_status() -> dict:
    """
    Return current local llama status plus basic CUDA info.
    Note: this does not trigger loading.
    """
    status = {
        "loaded": is_loaded(),
        "model_name": _loaded_model_name,
    }
    try:
        import torch  # type: ignore

        status["cuda_available"] = bool(torch.cuda.is_available())
        status["cuda_device_count"] = int(torch.cuda.device_count())
        status["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        status["cuda_available"] = None
        status["cuda_device_count"] = None
        status["cuda_device_name"] = None
    return status


def _read_token_from_files() -> Optional[str]:
    # Look for `hg.txt` / `hg` in likely locations.
    candidates = [
        Path(__file__).resolve().parent / "hg.txt",
        Path(__file__).resolve().parent / "hg",
        Path.cwd() / "hg.txt",
        Path.cwd() / "hg",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            token = p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if token and token.startswith("hf_"):
            return token
    return None


def _ensure_auth() -> None:
    # transformers/Hub libraries look for these env vars.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return
    token = _read_token_from_files()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _lazy_import_torch_transformers():
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    return torch, AutoTokenizer, AutoModelForCausalLM


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    force_reload: bool = False,
) -> None:
    global _tokenizer, _model, _loaded_model_name

    if not force_reload and _model is not None and _loaded_model_name == model_name:
        return

    _ensure_auth()

    torch, AutoTokenizer, AutoModelForCausalLM = _lazy_import_torch_transformers()

    # "무조건 GPU" 요구사항: CUDA가 없으면 CPU로 내려가지 않고 실패 처리합니다.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current environment. "
            "Install a CUDA-enabled PyTorch build (torch+cuXX) or fix your CUDA setup."
        )

    device_map = "auto"
    torch_dtype = torch.bfloat16

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tokenizer_kwargs = {"trust_remote_code": True}
    if token:
        tokenizer_kwargs["token"] = token

    _tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if getattr(_tokenizer, "pad_token", None) is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    model_kwargs = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if token:
        model_kwargs["token"] = token

    _model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    _model.eval()
    _loaded_model_name = model_name


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
    Generate text locally and return only the newly generated part.

    For Instruct models (e.g. Llama-3.1-8B-Instruct), set use_chat_template=True
    so the prompt is wrapped as a user turn; raw continuation otherwise tends to
    repeat or dump code.
    """
    torch, _, _ = _lazy_import_torch_transformers()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run on CPU.")

    if _model is None or _loaded_model_name != model_name:
        load_model(model_name=model_name)

    if _tokenizer is None or _model is None:
        raise RuntimeError("Local Llama model not loaded.")

    if (
        use_chat_template
        and getattr(_tokenizer, "chat_template", None) is not None
    ):
        try:
            enc = _tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}
        except Exception:
            inputs = _tokenizer(prompt, return_tensors="pt")
    else:
        inputs = _tokenizer(prompt, return_tensors="pt")

    if "attention_mask" not in inputs and "input_ids" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # With device_map="auto", keeping inputs on CPU often works (accelerate dispatch).
    # We'll only move if it is compatible.
    try:
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    except Exception:
        # If moving fails due to sharded weights, continue with CPU inputs.
        pass

    is_phi3 = model_name.lower().find("phi") != -1

    with torch.no_grad():
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": _tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if is_phi3:
            generate_kwargs["use_cache"] = False

        outputs = _model.generate(**generate_kwargs)

    # Decode only the new tokens (remove the prompt from the output).
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    return _tokenizer.decode(generated_ids, skip_special_tokens=True)


def load_au_probe(probe_path: str, layer: int = 32) -> None:
    global _au_probe_w, _au_probe_b, _au_probe_layer
    import torch
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"AU probe file not found: {probe_path}")
    state = torch.load(probe_path, map_location="cpu", weights_only=True)
    _au_probe_w = state["w"].float()
    _au_probe_b = torch.tensor(state["b"], dtype=torch.float32)
    _au_probe_layer = layer
    print(f"[local_llama] Loaded AU Probe layer {layer}")

def get_au_uncertainty(prompt: str, model_name: str = DEFAULT_MODEL_NAME, use_chat_template: bool = True) -> float:
    """
    Run a forward pass to extract hidden states and compute AU uncertainty.
    Ensure load_au_probe is called previously.
    """
    torch, _, _ = _lazy_import_torch_transformers()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run on CPU.")

    if _model is None or _loaded_model_name != model_name:
        load_model(model_name=model_name)

    if _tokenizer is None or _model is None:
        raise RuntimeError("Local Llama model not loaded.")

    if _au_probe_w is None or _au_probe_b is None or _au_probe_layer is None:
        raise RuntimeError("AU Probe is not loaded. Call load_au_probe() first.")

    if (
        use_chat_template
        and getattr(_tokenizer, "chat_template", None) is not None
    ):
        try:
            enc = _tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}
        except Exception:
            inputs = _tokenizer(prompt, return_tensors="pt")
    else:
        inputs = _tokenizer(prompt, return_tensors="pt")

    if "attention_mask" not in inputs and "input_ids" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    try:
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    except Exception:
        pass

    with torch.no_grad():
        outputs = _model(**inputs, output_hidden_states=True)
        # outputs.hidden_states is a tuple of (L+1) tensors length (batch, seq_len, hidden_dim)
        layer_hidden = outputs.hidden_states[_au_probe_layer]
        # We need the last token's hidden state. layer_hidden[:, -1, :] -> [batch, hidden_dim]
        last_token_hidden = layer_hidden[:, -1, :].cpu().float()

    logits = torch.matmul(last_token_hidden, _au_probe_w) + _au_probe_b
    uncertainty = torch.sigmoid(logits).item()
    return float(uncertainty)


