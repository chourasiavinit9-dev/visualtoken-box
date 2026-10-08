"""
Example: Using the GlassBox Python API directly (no web server needed).

This shows how to use GemmaEngine programmatically from your own Python code
to get per-token attention traces, logit-lens data, and top-k candidates.

Run:
    cd visualtoken-box
    source venv/bin/activate
    python .agents/skills/glassbox-interpretability/examples/python_api_usage.py
"""
from gemma_engine.loader import load_gemma
from gemma_engine.engine import GemmaEngine


def main():
    # ── 1. Load the model ────────────────────────────────────────────────────
    print("Loading model… (first time may download weights)")
    loader = load_gemma()   # uses GEMMA_MODEL from gemma_engine/config.py
    engine = GemmaEngine(loader)

    # ── 2. Generate with full trace access ───────────────────────────────────
    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt!r}")
    print("─" * 60)

    full_text = ""
    for step_idx, step in enumerate(engine.chat(prompt, max_new_tokens=10, greedy=True)):
        full_text += step.token_str

        # ── Token info ──────────────────────────────────────────────────────
        print(f"\nStep {step_idx}: token={step.token_str!r} | "
              f"{step.tokens_per_sec:.1f} tok/s | "
              f"entropy={step.entropy_bits:.2f} bits")

        # ── Top-3 candidates ─────────────────────────────────────────────────
        print("  Top-3 candidates:")
        for cand in step.top_k_candidates[:3]:
            print(f"    [{cand.token_str!r}] logit={cand.prob:.2f}")

        # ── Logit lens: which layer first "knew" the answer ──────────────────
        print("  Logit lens (first 5 layers):")
        for entry in step.lens[:5]:
            sw = "SW" if entry.is_sliding_window else "  "
            print(f"    Layer {entry.layer:2d} [{sw}] → {entry.token_str!r} ({entry.prob:.2%})")

        # ── Attention: avg weight on each token at layer 0 ──────────────────
        if step.attention_avg:
            layer0_attn = step.attention_avg[0]
            print(f"  Layer-0 avg attention ({len(layer0_attn)} positions): "
                  f"{[round(v, 3) for v in layer0_attn]}")

    print(f"\n\nFull response: {full_text!r}")

    # ── 3. Multi-turn chat ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Multi-turn example:")
    for step in engine.chat("What language do they speak there?", max_new_tokens=20, greedy=True):
        print(step.token_str, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
