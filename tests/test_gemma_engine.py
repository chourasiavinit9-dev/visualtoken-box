"""
tests/test_gemma_engine.py — Phase 1 test suite for gemma_engine/.

Tests (run with: pytest tests/test_gemma_engine.py -v)
------------------------------------------------------
(a) Model loads with eager attention — no crash, expected attributes present.
(b) Short prompt → non-empty text, token_str fields are strings.
(c) Shapes of logits/attentions/hidden_states are correct.
(d) Top-k probs sorted descending and sum ≤ 1.
(e) Greedy loop parity vs model.generate greedy.
(f) Second turn correctly uses first turn's history (answer refers to it).

IMPORTANT: These tests load the REAL model.  They require:
  - pip install torch transformers accelerate
  - huggingface-cli login  (or HF_TOKEN env var)
  - ~7–9 GB RAM
Model: unsloth/gemma-2-2b-it (default; override with GEMMA_MODEL env var)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import torch

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loader():
    """Load the model once for all tests in this module."""
    from gemma_engine.loader import GemmaLoader
    from gemma_engine.config import GEMMA_MODEL, DEVICE

    _loader = GemmaLoader()
    _loader.load(model_id=GEMMA_MODEL, device_str=DEVICE)
    return _loader


@pytest.fixture(scope="module")
def engine(loader):
    from gemma_engine.engine import GemmaEngine
    return GemmaEngine(loader)


# ── (a) Loader ───────────────────────────────────────────────────────────────

def test_a_loader_eager_attention(loader):
    """Model loads with eager attention; key attributes are present."""
    assert loader.loaded, "Loader must be in loaded state"
    assert loader.model is not None
    assert loader.processor is not None
    assert loader.config is not None

    # Check eager attention is actually active
    cfg = loader.config
    # The config should NOT request sdpa or flash_attention_2
    attn_impl = getattr(cfg, "_attn_implementation", None)
    # eager is the one we set; it may be stored differently across versions
    if attn_impl is not None:
        assert attn_impl == "eager", f"Expected eager, got {attn_impl}"

    # Model exposes num_hidden_layers
    assert hasattr(cfg, "num_hidden_layers")
    assert cfg.num_hidden_layers > 0

    print(f"\n✅ (a) Model loaded | layers={cfg.num_hidden_layers} | "
          f"dtype={loader.dtype} | device={loader.device} | "
          f"supports_vision={loader.supports_vision}")


# ── (b) Short chat → non-empty text ─────────────────────────────────────────

def test_b_short_chat_gives_text(engine):
    """Single-turn chat produces at least one non-empty token."""
    engine.reset()
    steps = list(engine.chat("Say hello.", max_new_tokens=20, greedy=True))
    assert len(steps) > 0, "Expected at least 1 generated token"
    full_text = "".join(s.token_str for s in steps)
    # strip model-specific special tokens like <end_of_turn>
    stripped = full_text.replace("<end_of_turn>", "").strip()
    assert len(stripped) > 0, f"Expected non-empty text, got: {repr(full_text)}"
    print(f"\n✅ (b) Generated {len(steps)} tokens: {repr(stripped[:80])}")


# ── (c) Shapes of traces ─────────────────────────────────────────────────────

def test_c_trace_shapes(loader, engine):
    """attention_avg, attention_per_head, and lens have correct dimensions."""
    engine.reset()
    steps = list(engine.chat("What is 2+2?", max_new_tokens=5, greedy=True))
    assert len(steps) > 0

    cfg = loader.config
    num_layers = cfg.num_hidden_layers

    for i, step in enumerate(steps):
        # Attention avg: num_layers × seq_len
        assert len(step.attention_avg) == num_layers, (
            f"step {i}: attention_avg has {len(step.attention_avg)} layers, "
            f"expected {num_layers}"
        )
        for layer_idx, row in enumerate(step.attention_avg):
            assert len(row) > 0, f"step {i} layer {layer_idx}: empty attention row"

        # Attention per-head: num_layers × num_heads × seq_len
        assert len(step.attention_per_head) == num_layers
        num_heads = cfg.num_attention_heads
        for layer_idx, heads in enumerate(step.attention_per_head):
            assert len(heads) == num_heads, (
                f"step {i} layer {layer_idx}: {len(heads)} heads, expected {num_heads}"
            )

        # Logit lens: one entry per layer
        assert len(step.lens) == num_layers, (
            f"step {i}: lens has {len(step.lens)} entries, expected {num_layers}"
        )

    print(f"\n✅ (c) Shapes correct for {len(steps)} steps | "
          f"layers={num_layers} | heads={num_heads}")


# ── (d) Top-k probabilities ──────────────────────────────────────────────────

def test_d_topk_probs_sorted_and_sum_le_one(engine):
    """Top-k candidates must be sorted descending, individual probs ≤ 1."""
    engine.reset()
    steps = list(engine.chat("Name one color.", max_new_tokens=10, greedy=True))
    assert len(steps) > 0

    for i, step in enumerate(steps):
        probs = [c.prob for c in step.top_k_candidates]
        # Sorted descending
        for j in range(len(probs) - 1):
            assert probs[j] >= probs[j + 1], (
                f"step {i}: probs not sorted at position {j}: {probs[j]:.4f} < {probs[j+1]:.4f}"
            )
        # Each prob ≤ 1
        for p in probs:
            assert 0.0 <= p <= 1.0 + 1e-6, f"step {i}: prob {p} out of range"
        # Sum ≤ 1 (top-k of full vocab, so sum may be < 1)
        assert sum(probs) <= 1.0 + 1e-6, f"step {i}: sum of top-k probs > 1: {sum(probs)}"

    print(f"\n✅ (d) Top-k probs sorted and ≤ 1 for {len(steps)} steps")


# ── (e) Greedy parity vs model.generate ──────────────────────────────────────

def test_e_greedy_parity_vs_model_generate(loader, engine):
    """Manual greedy loop must produce the same token sequence as model.generate."""
    from gemma_engine.config import SYSTEM_PROMPT

    processor = loader.processor
    model = loader.model
    device = loader.device

    user_msg = "What is the capital of France? Answer in one word."
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_msg})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    MAX_COMPARE = 10

    # Reference: model.generate greedy
    with torch.no_grad():
        ref_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_COMPARE,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    ref_new = ref_ids[0, inputs["input_ids"].shape[1]:].tolist()

    # Our loop: greedy
    engine.reset()
    steps = list(engine.chat(user_msg, max_new_tokens=MAX_COMPARE, greedy=True))
    our_ids = [s.token_id for s in steps]

    # Compare up to length of shorter sequence (engine stops on EOS, generate may not)
    n = min(len(ref_new), len(our_ids))
    assert n > 0, "No tokens generated"
    match = our_ids[:n] == ref_new[:n]
    if not match:
        for pos in range(n):
            ref_tok = processor.decode([ref_new[pos]])
            our_tok = processor.decode([our_ids[pos]])
            print(f"  pos {pos}: ref={repr(ref_tok)}({ref_new[pos]})  "
                  f"ours={repr(our_tok)}({our_ids[pos]})")
    assert match, (
        f"Greedy parity failed.\n"
        f"  Reference: {ref_new[:n]}\n"
        f"  Our loop:  {our_ids[:n]}"
    )
    ref_text = processor.decode(ref_new[:n], skip_special_tokens=True)
    our_text = processor.decode(our_ids[:n], skip_special_tokens=True)
    print(f"\n✅ (e) Greedy parity: first {n} tokens match")
    print(f"   Reference: {repr(ref_text)}")
    print(f"   Our loop:  {repr(our_text)}")


# ── (f) Multi-turn history ───────────────────────────────────────────────────

def test_f_multiturn_history(engine):
    """Second turn should have access to first turn's context."""
    engine.reset()

    # First turn: tell the model a fact
    t1 = list(engine.chat(
        "My favourite number is 42. Remember that.",
        max_new_tokens=30,
        greedy=True,
    ))
    assert len(t1) > 0

    # Second turn: ask about the fact
    t2 = list(engine.chat(
        "What is my favourite number?",
        max_new_tokens=20,
        greedy=True,
    ))
    assert len(t2) > 0

    # History should now have 2 user turns + 2 assistant turns
    assert len(engine._history) == 4, (
        f"Expected 4 history entries, got {len(engine._history)}"
    )

    full_t2 = "".join(s.token_str for s in t2)
    print(f"\n✅ (f) Second turn response: {repr(full_t2[:120])}")
    # The model should mention 42 somewhere
    assert "42" in full_t2, (
        f"Expected '42' in second-turn response, got: {repr(full_t2)}"
    )
