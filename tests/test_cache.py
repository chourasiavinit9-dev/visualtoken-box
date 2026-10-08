"""KV cache test: cache on vs off must give identical tokens and (near) identical logits."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbox.model import Transformer
from glassbox.tokenizer import Tokenizer
from glassbox.generate import generate_stream

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"


def _run(model, tok, prompt, use_cache, n):
    # top_k=1 makes sampling greedy (always the argmax token)
    return list(generate_stream(
        model, tok, prompt, max_new_tokens=n,
        temperature=1.0, top_k=1, top_p=1.0, use_cache=use_cache,
    ))


def _check(prompt, n):
    tok = Tokenizer(MODEL_DIR)
    model = Transformer(MODEL_DIR)

    on = _run(model, tok, prompt, True, n)
    off = _run(model, tok, prompt, False, n)

    ids_on = [s.token_id for s in on]
    ids_off = [s.token_id for s in off]
    first_bad = next((i for i, (a, b) in enumerate(zip(ids_on, ids_off)) if a != b), None)
    assert first_bad is None, f"tokens first differ at step {first_bad}"
    assert len(on) == len(off)

    diffs = [float(np.max(np.abs(a.output_logits - b.output_logits))) for a, b in zip(on, off)]
    print(f"{len(on)} steps, max logit diff cache vs no-cache: {max(diffs):.8f}")
    assert max(diffs) < 1e-3, f"logits drift, worst at step {int(np.argmax(diffs))}"


def test_cache_matches_no_cache_short_prompt():
    _check("The capital of France is", 20)


def test_cache_matches_no_cache_long_prompt():
    # ~120 tokens, so RoPE and the cache are exercised at larger positions
    _check(" ".join(["The quick brown fox jumps over the lazy dog."] * 12), 8)


if __name__ == "__main__":
    test_cache_matches_no_cache_short_prompt()
    test_cache_matches_no_cache_long_prompt()