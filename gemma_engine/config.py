"""
gemma_engine/config.py — Environment-variable configuration for the VisualBox HF engine.
All settings are read once at import time.
Set VISUALBOX_MODEL to any HuggingFace model ID to use a different LLM.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Model identity
# VISUALBOX_MODEL accepts any HuggingFace model ID.
# Examples:
#   HuggingFaceTB/SmolLM2-135M-Instruct  (default — CPU-friendly, ~270MB)
#   google/gemma-3-4b-it                  (requires HF login)
#   mistralai/Mistral-7B-Instruct-v0.3
#   meta-llama/Llama-3.2-3B-Instruct      (requires HF login)
#   microsoft/Phi-3.5-mini-instruct
#   Qwen/Qwen2.5-7B-Instruct
#   gpt2                                  (no login, instant test)
# ---------------------------------------------------------------------------
# Accept both VISUALBOX_MODEL (new canonical name) and GEMMA_MODEL (legacy alias)
VISUALBOX_MODEL: str = (
    os.environ.get("VISUALBOX_MODEL")
    or os.environ.get("GEMMA_MODEL")
    or "unsloth/gemma-2-2b-it"
)
# Keep the old name as an alias for backward compatibility
GEMMA_MODEL = VISUALBOX_MODEL

# ---------------------------------------------------------------------------
# Hardware / dtype
# ---------------------------------------------------------------------------
DEVICE: str = os.environ.get("DEVICE", "cpu")   # "cpu" avoids disk-offloading on memory-constrained machines

# ---------------------------------------------------------------------------
# Generation defaults (all overridable per-request)
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS: int = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TOP_K: int = int(os.environ.get("TOP_K", "10"))        # top-k candidates to return in trace
TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P: float = float(os.environ.get("TOP_P", "0.9"))

# ---------------------------------------------------------------------------
# System prompt (short — keeps the chat fast)
# ---------------------------------------------------------------------------
# Gemma 2's chat template raises "System role not supported" if a system message is sent.
# Default to empty so no system turn is injected — override via VISUALBOX_SYSTEM_PROMPT if your model supports it.
SYSTEM_PROMPT: str = os.environ.get(
    "VISUALBOX_SYSTEM_PROMPT",
    os.environ.get("GEMMA_SYSTEM_PROMPT", ""),
)
