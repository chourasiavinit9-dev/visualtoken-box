"""Logit lens test: the last layer's lens must reproduce the model's real logits."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbox.model import Transformer
from glassbox.tokenizer import Tokenizer
from glassbox.trace import logit_lens, rmsnorm

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"
N_LAYERS = 30  # SmolLM2-135M


def test_logit_lens_last_layer_matches_real_logits():
    tok = Tokenizer(MODEL_DIR)
    model = Transformer(MODEL_DIR)
    ids = np.array(tok.encode("The capital of France is"), dtype=np.int64)

    out = model.forward(ids, trace=True)
    real = out.logits[0, -1, :]

    assert len(out.traces) == N_LAYERS, f"expected {N_LAYERS} traces, got {len(out.traces)}"

    # Redo the lens math on the last layer, exactly as logit_lens() does
    h = out.traces[-1].hidden_state[0, -1, :]
    x = rmsnorm(h[np.newaxis, :], model.final_norm_weight,
                eps=model.config.rms_norm_eps)[0]
    lens_logits = x @ model.lm_head.T

    max_diff = float(np.max(np.abs(lens_logits - real)))
    print(f"max diff, last-layer lens vs real logits: {max_diff:.8f}")
    assert max_diff < 1e-4, f"lens differs from real logits by {max_diff}"

    # And check the real function's output for the last layer
    results = logit_lens(model, tok, out.traces)
    assert len(results) == N_LAYERS
    print(f"lens top: {results[-1].top_token_str!r}   real top: {tok.decode(int(np.argmax(real)))!r}")
    assert results[-1].top_token_id == int(np.argmax(real))


if __name__ == "__main__":
    test_logit_lens_last_layer_matches_real_logits()