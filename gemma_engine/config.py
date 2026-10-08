"""
gemma_engine/config.py — Environment-variable configuration for the Gemma engine.
All settings are read once at import time.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
GEMMA_MODEL: str = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b-it")

# ---------------------------------------------------------------------------
# Hardware / dtype
# ---------------------------------------------------------------------------
DEVICE: str = os.environ.get("DEVICE", "auto")   # "auto" | "cpu" | "cuda" | "mps"

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
SYSTEM_PROMPT: str = os.environ.get(
    "GEMMA_SYSTEM_PROMPT",
    "You are a helpful, concise assistant.",
)
