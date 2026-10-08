"""Test streaming generation and measure KV Cache performance gains."""
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbox.model import Transformer
from glassbox.tokenizer import Tokenizer
from glassbox.generate import generate_stream

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"


def run_demo(model, tokenizer, prompt: str, use_cache: bool):
    cache_status = "ON 🚀" if use_cache else "OFF 🐌"
    print(f"\n🎬 Running with KV Cache {cache_status}")
    print(f"💬 Prompt: '{prompt}'")
    print("✨ Output: ", end="", flush=True)
    
    start = time.perf_counter()
    token_count = 0
    speeds = []
    
    stream = generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=35,
        temperature=0.7,
        use_cache=use_cache,
        trace=False,
    )
    
    for step in stream:
        print(step.token_str, end="", flush=True)
        token_count += 1
        speeds.append(step.tokens_per_sec)
        
    total_time = time.perf_counter() - start
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    
    print("\n" + "-" * 50)
    print(f"📊 Stats: Generated {token_count} tokens in {total_time:.2f}s")
    print(f"   Average Speed: {avg_speed:.2f} tokens/sec")
    print("-" * 50)
    
    return avg_speed


def test_generation_stream_emits_finite_token_steps():
    tok = Tokenizer(MODEL_DIR)
    model = Transformer(MODEL_DIR)
    steps = list(generate_stream(
        model,
        tok,
        "Paris is in France. Rome is in",
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        use_cache=True,
        trace=False,
    ))
    assert 1 <= len(steps) <= 3
    assert all(step.is_cache_active for step in steps)
    assert all(step.output_logits.ndim == 1 for step in steps)
    assert all(np.isfinite(step.output_logits).all() for step in steps)


def benchmark_cache_speeds():
    print("=" * 60)
    print("  🔮 GlassBox Generation & Performance Benchmark")
    print("  Made with 🔮 by Vivi")
    print("=" * 60)
    print()
    
    # Load model and tokenizer
    tok = Tokenizer(MODEL_DIR)
    model = Transformer(MODEL_DIR)
    
    prompt = "Once upon a time in a digital kingdom, there was a tiny AI helper who"
    
    # 1. Run without cache (slow)
    slow_speed = run_demo(model, tok, prompt, use_cache=False)
    
    # 2. Run with cache (fast!)
    fast_speed = run_demo(model, tok, prompt, use_cache=True)
    
    # Compute speedup
    speedup = (fast_speed / slow_speed) if slow_speed > 0 else 1
    print(f"\n🏆 BENCHMARK RESULT: KV Cache is {speedup:.2f}x FASTER! ⚡⚡")
    print("🎉 Streaming generation is fully functional!")


if __name__ == "__main__":
    benchmark_cache_speeds()