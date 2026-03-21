[Launch PACT Privacy Assistant](frontend/index.html)

**Local Llama smoke test (no server / no `GPT_API_KEY`):** from repo root, `python scripts/test_local_llama.py` (CUDA + HF token as for the backend). Use `--case modules` to print real detector candidates only (no model load). `--case synthesis` runs `collect_pipeline_inputs` then Llama.

**Chat UI:** expand **Pipeline: module masks → Local Llama → GPT** on each reply to see per-module candidates, full synthesis prompt, raw Llama output, and `final_prompt_to_gpt`.

**Synthesis:** Llama is optional. If you only need deterministic merging of module outputs, set `PACT_USE_LOCAL_LLAMA_FOR_SYNTHESIS=0` before starting the backend (skips GPU wait; uses financial-redacted line if present, else longest candidate).
