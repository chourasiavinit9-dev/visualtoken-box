# gemma_engine — Model-agnostic HuggingFace inference engine for VisualBox
# Supports any AutoModelForCausalLM or AutoModelForImageTextToText model.
#
# Quick start:
#   from gemma_engine.loader import load_model
#   from gemma_engine.engine import VisualBoxEngine
#   engine = VisualBoxEngine(load_model())
#   for step in engine.chat("Hello!"):
#       print(step.token_str, end="", flush=True)
#
# Set VISUALBOX_MODEL env var to use your own model:
#   VISUALBOX_MODEL=mistralai/Mistral-7B-Instruct-v0.3 python run_server.py

from gemma_engine.engine import VisualBoxEngine, InferenceStep, GemmaEngine, GemmaStep
from gemma_engine.loader import ModelLoader, load_model, load_gemma, GemmaLoader

__all__ = [
    # Canonical (model-agnostic) names
    "VisualBoxEngine",
    "InferenceStep",
    "ModelLoader",
    "load_model",
    # Backward-compatible aliases
    "GemmaEngine",
    "GemmaStep",
    "GemmaLoader",
    "load_gemma",
]
