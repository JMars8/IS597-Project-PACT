"""
Local Llama wrapper for PACT.

This module is intentionally self-contained inside the PACT repo.
It loads a Hugging Face causal LM locally (GPU if available) and generates
text from a prompt.

AU-Probe integration:
  - load_au_probe()      loads a linear probe (.pt) trained on Llama hidden states
  - get_au_uncertainty() extracts hidden states from the loaded model (layer 32)
                         and scores them with the probe

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

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Model load helpers
# ---------------------------------------------------------------------------

def is_loaded() -> bool:
    return _model is not None and _loaded_model_name is not None


def get_status() -> dict:
    """
    Return current local llama status plus basic CUDA info.
    Note: this does not trigger loading.
    """
    status = {
        "loaded":      is_loaded(),
        "model_name":  _loaded_model_name,
        "probe_ready": _au_probe_loaded,
    }
    try:
        import torch  # type: ignore

        status["cuda_available"]    = bool(torch.cuda.is_available())
        status["cuda_device_count"] = int(torch.cuda.device_count())
        status["cuda_device_name"]  = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        status["cuda_available"]    = None
        status["cuda_device_count"] = None
        status["cuda_device_name"]  = None
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

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current environment. "
            "Install a CUDA-enabled PyTorch build (torch+cuXX) or fix your CUDA setup."
        )

    device_map  = "auto"
    torch_dtype = torch.bfloat16

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tokenizer_kwargs = {"trust_remote_code": True}
    if token:
        tokenizer_kwargs["token"] = token

    _tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if getattr(_tokenizer, "pad_token", None) is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    model_kwargs = {
        "torch_dtype":    torch_dtype,
        "device_map":     device_map,
        "trust_remote_code": True,
        "output_hidden_states": True,   # needed for AU probe
    }
    if token:
        model_kwargs["token"] = token

    _model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    _model.eval()
    _loaded_model_name = model_name


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

    Args:
        probe_path: absolute path to the .pt probe file.
        layer:      Llama layer the probe was trained on (used in get_au_uncertainty).
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
        import torch
        data = torch.load(probe_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            # Support both 'weight'/'bias' (nn.Linear style) and 'w'/'b' (AU-Med style)
            weight_key = next(
                (k for k in data if k in ('w', 'weight') or 'weight' in k.lower()), None
            )
            bias_key = next(
                (k for k in data if k in ('b', 'bias') or ('bias' in k.lower() and k != weight_key)), None
            )
            if weight_key is None:
                print(f"WARNING: No weight key found in probe dict {list(data.keys())} – probe disabled.")
                return

            # Convert weight: could be torch.Tensor or ndarray or plain number
            w_val = data[weight_key]
            if hasattr(w_val, 'float'):          # torch.Tensor
                _au_probe_weights = w_val.float().squeeze()
            elif hasattr(w_val, 'astype'):       # numpy array
                _au_probe_weights = torch.from_numpy(w_val.astype('float32')).squeeze()
            else:                                # plain Python scalar
                _au_probe_weights = torch.tensor(float(w_val), dtype=torch.float32)

            # Convert bias: same – could be float, tensor, or ndarray
            if bias_key is not None:
                b_val = data[bias_key]
                if hasattr(b_val, 'float'):      # torch.Tensor
                    _au_probe_bias = b_val.float().squeeze()
                elif hasattr(b_val, 'astype'):   # numpy array
                    _au_probe_bias = torch.from_numpy(b_val.astype('float32')).squeeze()
                else:                            # plain Python float
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
            f"AU probe loaded ✓  path={probe_path}  layer={layer}  "
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
    Estimate aleatoric uncertainty of *prompt* using the loaded linear probe.

    Extracts the mean-pooled hidden state from layer `_au_probe_layer` of the
    locally loaded Llama model, then runs the linear probe:
        score = sigmoid(w · hidden + b)

    Returns a float in [0, 1]. Returns 0.0 if probe or model is not loaded.
    Threshold check (>= 0.8) lives in server.py.
    """
    if not _au_probe_loaded or _au_probe_weights is None:
        return 0.0

    if _model is None or _tokenizer is None:
        return 0.0

    try:
        import torch
        import torch.nn.functional as F

        # --- Tokenise the prompt ---
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
            inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        try:
            inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        except Exception:
            pass

        # --- Forward pass (no generation, just hidden states) ---
        with torch.no_grad():
            outputs = _model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # outputs.hidden_states is a tuple of (num_layers+1) tensors: (batch, seq, hidden)
        hidden_states = outputs.hidden_states  # tuple length = num_layers + 1

        # Layer indexing: hidden_states[0] = embedding, hidden_states[i] = layer i output
        layer_idx = min(_au_probe_layer, len(hidden_states) - 1)
        layer_hidden = hidden_states[layer_idx]   # (1, seq_len, hidden_dim)

        # Mean-pool across sequence dimension → (hidden_dim,)
        embedding = layer_hidden[0].mean(dim=0).float().cpu()

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

        score = float(torch.sigmoid(logit).item())
        print(f"DEBUG: AU uncertainty score = {score:.4f}")
        return score

    except Exception as e:
        print(f"DEBUG: AU uncertainty calculation error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Text generation
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
    Generate text locally and return only the newly generated part.

    For Instruct models (e.g. Llama-3.1-8B-Instruct), set use_chat_template=True
    so the prompt is wrapped as a user turn.
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

    try:
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    except Exception:
        pass

    is_phi3 = model_name.lower().find("phi") != -1

    with torch.no_grad():
        generate_kwargs = {
            **inputs,
            "max_new_tokens":    max_new_tokens,
            "temperature":       temperature,
            "top_p":             top_p,
            "do_sample":         True,
            "pad_token_id":      _tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if is_phi3:
            generate_kwargs["use_cache"] = False

        outputs = _model.generate(**generate_kwargs)

    # Decode only the new tokens (remove the prompt from the output).
    prompt_len    = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    return _tokenizer.decode(generated_ids, skip_special_tokens=True)
