#!/usr/bin/env python3
"""
swap_model.py — Quick helper to change the model in gemma_engine/config.py.

Usage:
    python .agents/skills/glassbox-interpretability/scripts/swap_model.py \
        "mistralai/Mistral-7B-Instruct-v0.3"

This edits gemma_engine/config.py in place and restarts the server if it's running.
"""
import sys
import re
import subprocess
from pathlib import Path

# ── Preset shortcuts ──────────────────────────────────────────────────────────
PRESETS = {
    "smollm":    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b":"google/gemma-3-12b-it",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi3":      "microsoft/Phi-3.5-mini-instruct",
    "qwen7b":    "Qwen/Qwen2.5-7B-Instruct",
    "gpt2":      "gpt2",
}

REPO_ROOT = Path(__file__).resolve().parents[4]   # climb up from scripts/
CONFIG_PATH = REPO_ROOT / "gemma_engine" / "config.py"


def list_presets():
    print("\nAvailable presets:\n")
    for key, model_id in PRESETS.items():
        print(f"  {key:<12} → {model_id}")
    print()


def swap(model_id: str):
    if not CONFIG_PATH.exists():
        print(f"❌ Could not find config at: {CONFIG_PATH}")
        sys.exit(1)
    
    content = CONFIG_PATH.read_text()
    new_content = re.sub(
        r'(GEMMA_MODEL\s*=\s*)["\'].*?["\']',
        f'\\1"{model_id}"',
        content,
    )
    if new_content == content:
        print(f"⚠️  GEMMA_MODEL line not found in {CONFIG_PATH}")
        sys.exit(1)
    
    CONFIG_PATH.write_text(new_content)
    print(f"✅ Swapped model → {model_id}")
    print(f"   ({CONFIG_PATH})")
    print("\n🔄 Restart the server to apply: python run_server.py\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python swap_model.py <model-id-or-preset>")
        list_presets()
        sys.exit(0)
    
    arg = sys.argv[1].strip()
    model_id = PRESETS.get(arg, arg)   # resolve preset or use as-is
    swap(model_id)


if __name__ == "__main__":
    main()
