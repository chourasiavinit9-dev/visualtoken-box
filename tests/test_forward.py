"""Test full transformer forward pass with real prompt."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbox.model import Transformer
from glassbox.tokenizer import Tokenizer

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"


def test_forward_pass_produces_finite_logits():
    print("=" * 60)
    print("  🔮 GlassBox Forward Pass Test")
    print("  Testing full 30-layer transformer forward pass in NumPy")
    print("=" * 60)
    print()
    
    # 1. Initialize tokenizer and model
    print("📚 Loading Tokenizer...")
    tok = Tokenizer(MODEL_DIR)
    
    print("🧠 Loading Transformer Model & Weights...")
    model = Transformer(MODEL_DIR)
    print("   ✅ Model loaded successfully!")
    print()
    
    # 2. Test prompt
    prompt = "The capital of France is"
    print(f"💬 Prompt: '{prompt}'")
    
    input_ids = np.array(tok.encode(prompt), dtype=np.int64)
    print(f"🔢 Input IDs: {input_ids.tolist()}")
    
    # 3. Run forward pass
    print("⚡ Running Forward Pass across 30 layers in pure NumPy...")
    output = model.forward(input_ids, trace=True)
    assert output.logits.shape == (1, len(input_ids), model.config.vocab_size)
    assert np.isfinite(output.logits).all()
    
    # 4. Get logits for the LAST token position
    last_token_logits = output.logits[0, -1, :]  # Shape: (vocab_size,)
    
    # 5. Find top 5 predicted tokens
    top_5_indices = np.argsort(last_token_logits)[-5:][::-1]
    
    print("\n🔮 Top 5 next-token predictions from scratch:")
    print("-" * 40)
    for rank, token_id in enumerate(top_5_indices, 1):
        token_str = tok.decode(int(token_id))
        logit_val = last_token_logits[token_id]
        print(f"  #{rank}: '{token_str}' (id: {token_id}, logit: {logit_val:.2f})")
    
    print("-" * 40)
    top_predicted = tok.decode(int(top_5_indices[0]))
    print(f"🎯 Model predicts: '{prompt} {top_predicted.strip()}'")
    print()
    print("🎉 FULL FORWARD PASS COMPLETE AND WORKING! 💜")