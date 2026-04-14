"""
Local Llama/Ollama wrapper for PACT.

This module prioritizes Ollama for local inference if available, 
falling back to Transformers/Torch only if explicitly configured.
"""

from __future__ import annotations

import os
import requests
import json
from typing import Optional

# Configuration for Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

_loaded_model_name: Optional[str] = None
_use_ollama: bool = True # Default to true for your environment

def is_loaded() -> bool:
    if _use_ollama:
        try:
            # Check if the model is available in Ollama
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return any(OLLAMA_MODEL in m.get("name", "") for m in models)
        except Exception:
            return False
    return False

def get_status() -> dict:
    """
    Return current local llama status focused on Ollama.
    """
    status = {
        "loaded": is_loaded(),
        "model_name": OLLAMA_MODEL if _use_ollama else None,
        "backend": "ollama" if _use_ollama else "transformers",
        "ollama_url": OLLAMA_BASE_URL
    }
    
    # Basic health check for Ollama
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        status["ollama_connected"] = (resp.status_code == 200)
    except Exception:
        status["ollama_connected"] = False
        
    return status

def load_model(
    model_name: str = OLLAMA_MODEL,
    force_reload: bool = False,
) -> None:
    global _loaded_model_name
    
    if _use_ollama:
        # For Ollama, "loading" just means checking if it exists
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if not any(model_name in m.get("name", "") for m in models):
                raise RuntimeError(
                    f"Model '{model_name}' not found in Ollama. "
                    f"Please run 'ollama pull {model_name}' in your terminal."
                )
            _loaded_model_name = model_name
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not connect to Ollama at {OLLAMA_BASE_URL}: {e}")
    else:
        raise RuntimeError("Direct Transformers loading is disabled in favor of Ollama.")


def load_au_probe(probe_path: str, layer: int = 32) -> None:
    """Stub for backend compatibility; Ollama-only builds have no in-process probe."""
    return


def get_au_uncertainty(prompt: str) -> float:
    """No probe in Ollama-only builds; backend uses this for optional gating."""
    return 0.0


def generate_text(
    prompt: str,
    max_new_tokens: int = 220,
    temperature: float = 0.2,
    top_p: float = 0.9,
    model_name: str = OLLAMA_MODEL,
    use_chat_template: bool = False,
    repetition_penalty: float = 1.12,
) -> str:
    """
    Generate text using the Ollama API.
    """
    if not _use_ollama:
        raise RuntimeError("Ollama mode is active but attempted non-Ollama call.")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
        }
    }

    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        print(f"DEBUG: Ollama generating at {url} with model {model_name}")
        response = requests.post(
            url,
            json=payload,
            timeout=600 # Increased to 10 minutes for large document synthesis
        )
        if response.status_code != 200:
            print(f"DEBUG: Ollama error {response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"DEBUG: Ollama exception: {str(e)}")
        raise RuntimeError(f"Ollama generation failed: {e}")
