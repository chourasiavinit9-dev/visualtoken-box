"""Quick smoke test for loader and ops."""
from pathlib import Path
import sys

# Auto-detect project root directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from glassbox.loader import ModelConfig, load_all_weights
from glassbox.ops import rmsnorm, softmax, silu, precompute_rope, apply_rope, causal_mask

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"


def test_config():
    print("🔮 Testing config loader...")
    cfg = ModelConfig.from_dir(MODEL_DIR)
    print(cfg)
    assert cfg.hidden_size == 576, f"Expected 576, got {cfg.hidden_size}"
    assert cfg.head_dim == 64, f"Expected 64, got {cfg.head_dim}"
    assert cfg.num_kv_groups == 3, f"Expected 3, got {cfg.num_kv_groups}"
    print("   ✅ Config loaded correctly!\n")


def test_weights():
    print("📦 Testing weight loader...")
    weights = load_all_weights(MODEL_DIR)
    
    # Check some expected tensors
    assert "model.embed_tokens.weight" in weights
    embed = weights["model.embed_tokens.weight"]
    print(f"   Embedding shape: {embed.shape} (expect ~49152 x 576)")
    print(f"   Embedding dtype: {embed.dtype} (should be float32)")
    print(f"   Total tensors: {len(weights)}")
    print(f"   Total params: {sum(w.size for w in weights.values()):,}")
    print("   ✅ Weights loaded correctly!\n")
def test_ops():
    print("⚡ Testing core ops...")
    
    # RMSNorm
    x = np.random.randn(1, 10, 576).astype(np.float32)
    w = np.ones(576, dtype=np.float32)
    out = rmsnorm(x, w, eps=1e-5)
    assert out.shape == x.shape
    print("   ✅ RMSNorm works")
    
    # Softmax
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    probs = softmax(logits)
    assert abs(probs.sum() - 1.0) < 1e-5
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]
    print("   ✅ Softmax works")
    
    # SiLU
    x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    out = silu(x)
    assert abs(out[0]) < 1e-6  # silu(0) = 0
    print("   ✅ SiLU works")
    
    # RoPE
    cos_c, sin_c = precompute_rope(64, 128, 10000.0)
    assert cos_c.shape == (128, 64)
    q = np.random.randn(1, 9, 10, 64).astype(np.float32)
    k = np.random.randn(1, 3, 10, 64).astype(np.float32)
    q_out, k_out = apply_rope(q, k, cos_c, sin_c)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    print("   ✅ RoPE works")
    
    # Causal mask
    mask = causal_mask(5)
    assert mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, 0, 1] == -np.inf  # Can't attend to future
    assert mask[0, 0, 1, 0] == 0.0      # Can attend to past
    print("   ✅ Causal mask works")
    
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  🔮 GlassBox Foundation Test Suite")
    print("  Made with 🔮 by Vivi")
    print("=" * 50)
    print()
    
    test_config()
    test_weights()
    test_ops()
    
    print("🎉 ALL TESTS PASSED! Foundation is solid!")
    print("   Next: Forward pass + generation loop!")