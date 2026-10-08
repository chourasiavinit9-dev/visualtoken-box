"""
GlassBox Correctness Harness — Verify NumPy forward pass against HuggingFace PyTorch.
Proves exact numerical accuracy of RoPE, Attention, RMSNorm, and SwiGLU. 🔮
Made with 💜 by Vivi
"""
from pathlib import Path
import sys
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbox.model import Transformer
from glassbox.tokenizer import Tokenizer
from glassbox.generate import generate_stream

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"


def test_forward_and_greedy_generation_match_huggingface():
    print("=" * 65)
    print("  🔮 GlassBox vs Hugging Face Correctness Test")
    print("  Verifying NumPy math against PyTorch reference implementation")
    print("=" * 65)
    print()

    # 1. Load HuggingFace Reference Model (Pure CPU PyTorch)
    print("🤗 Loading Hugging Face Reference Model (PyTorch)...")
    hf_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    print("   ✅ HF Model loaded!")

    # 2. Load GlassBox From-Scratch Model
    print("🔮 Loading GlassBox Engine (NumPy from-scratch)...")
    gb_tok = Tokenizer(MODEL_DIR)
    gb_model = Transformer(MODEL_DIR)
    print("   ✅ GlassBox Engine loaded!\n")

    # 3. Test Prompt
    prompt = "Artificial Intelligence is"
    print(f"💬 Test Prompt: '{prompt}'")

    # Encode with both
    hf_inputs = hf_tok(prompt, return_tensors="pt")
    gb_ids = np.array([gb_tok.encode(prompt)], dtype=np.int64)

    # 4. Run HF Forward Pass
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
        hf_logits = hf_outputs.logits.numpy()  # (1, seq_len, vocab_size)

    # 5. Run GlassBox Forward Pass
    gb_outputs = gb_model.forward(gb_ids)
    gb_logits = gb_outputs.logits  # (1, seq_len, vocab_size)

    # 6. Compare Logits
    print("\n🔍 Comparing Outputs:")
    print("-" * 50)
    
    # Check shape
    assert hf_logits.shape == gb_logits.shape, f"Shape mismatch: {hf_logits.shape} vs {gb_logits.shape}"
    print(f"  Shape check: {gb_logits.shape} ✅ MATCH")

    # Check maximum absolute difference
    max_diff = np.max(np.abs(hf_logits - gb_logits))
    mean_diff = np.mean(np.abs(hf_logits - gb_logits))
    print(f"  Max Logit Difference:  {max_diff:.6f}")
    print(f"  Mean Logit Difference: {mean_diff:.6f}")
    assert max_diff < 1e-3, f"Max logit difference {max_diff:.8g} exceeds 1e-3"

    # Check top-5 predictions for the next token
    hf_last_logits = hf_logits[0, -1, :]
    gb_last_logits = gb_logits[0, -1, :]

    hf_top5 = np.argsort(hf_last_logits)[-5:][::-1]
    gb_top5 = np.argsort(gb_last_logits)[-5:][::-1]

    print("\n🔮 Top-5 Token Prediction Comparison:")
    print(f"{'Rank':<6} | {'Hugging Face':<20} | {'GlassBox (NumPy)':<20} | {'Match?'}")
    print("-" * 65)

    all_matched = True
    for i in range(5):
        hf_word = hf_tok.decode([hf_top5[i]]).strip()
        gb_word = gb_tok.decode([gb_top5[i]]).strip()
        match = (hf_top5[i] == gb_top5[i])
        status = "✅ MATCH" if match else "❌ DIFF"
        if not match:
            all_matched = False
        print(f"#{i+1:<5} | '{hf_word}' ({hf_top5[i]}){' ':<8} | '{gb_word}' ({gb_top5[i]}){' ':<8} | {status}")

    print("-" * 65)
    
    assert all_matched, "Top-5 token IDs differ from Hugging Face"
    print(f"\nVerification passed: max logit diff {max_diff:.8g}; top-5 IDs match.")

    prompt = "The capital of France is"
    input_ids = torch.tensor([gb_tok.encode(prompt)])
    gb_tokens = [step.token_id for step in generate_stream(
        gb_model,
        gb_tok,
        prompt,
        max_new_tokens=25,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        use_cache=True,
    )]
    with torch.no_grad():
        hf_generated = hf_model.generate(input_ids, max_new_tokens=25, do_sample=False)
    hf_tokens = hf_generated[0, input_ids.shape[1]:].tolist()
    assert len(gb_tokens) == 25, f"GlassBox generated {len(gb_tokens)} greedy tokens, expected 25"
    assert gb_tokens == hf_tokens, "25-token greedy generation differs from Hugging Face"
    print("Verified 25 of 25 greedy generated tokens match Hugging Face.")