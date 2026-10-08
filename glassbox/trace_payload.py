"""
GlassBox Trace Payloads — Build compact JSON payloads for streaming to UI.
Only sends what the browser needs; the browser does the heavy UI math.
Made with 🔮 by Vinit Chaurasia
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class StepPayload:
    """A single generation step payload for the frontend."""
    step: int                              # Which generation step (0-indexed)
    token_id: int                          # The generated token ID
    token_str: str                         # Decoded string
    tokens_per_sec: float
    cache_active: bool
    entropy_bits: float
    # Top-N raw logits so client can re-softmax instantly on slider change
    top_logit_ids: list[int]               # Length N (e.g., 200)
    top_logit_values: list[float]
    # Attention: only the NEW row per layer per head (new token vs all prev)
    # Shape: [num_layers][num_heads][total_seq_len]
    attention_new_rows: list[list[list[float]]] | None
    # Logit lens: top-1 per layer at this position
    lens_predictions: list[dict] | None    # [{layer, token_str, prob}]


def build_step_payload(
    step_idx: int,
    token_id: int,
    token_str: str,
    tokens_per_sec: float,
    cache_active: bool,
    logits: np.ndarray,          # (vocab_size,)
    traces: list,                 # list of LayerTrace
    tokenizer,
    model,
    top_n: int = 200,
    include_attention: bool = True,
    include_lens: bool = True,
) -> dict:
    """
    Build a compact JSON-serializable payload for the frontend.
    
    - Sends only top_n logits (client re-softmaxes on temperature/top-p change)
    - Sends only the NEW attention row per layer (not full matrix)
    - Sends only top-1 lens prediction per layer
    """
    # 1. Top-N logits (ids + raw values)
    top_indices = np.argpartition(logits, -top_n)[-top_n:]
    # Sort descending by logit value
    top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
    top_values = logits[top_indices]

    # Model uncertainty over the full vocabulary at temperature 1, in bits
    z = logits.astype(np.float64)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    nz = p[p > 0]
    entropy_bits = float(-(nz * np.log2(nz)).sum())

    payload = {
        "step": step_idx,
        "token_id": int(token_id),
        "token_str": token_str,
        "tokens_per_sec": float(tokens_per_sec),
        "cache_active": bool(cache_active),
        "entropy_bits": round(entropy_bits, 3),
        "top_logit_ids": [int(i) for i in top_indices],
        "top_logit_values": [float(v) for v in top_values],
        "top_logit_strs": [tokenizer.decode(int(i)) for i in top_indices],
    }
    
    # 2. Attention: NEW token's row only (new token vs all prior tokens)
    if include_attention and traces:
        attn_data = []
        for trace in traces:
            if trace.attn_weights is None:
                attn_data.append(None)
                continue
            # trace.attn_weights shape: (batch, num_heads, seq_len, total_seq_len)
            # The "new row" is the LAST row of the attention matrix for this step
            # Shape: (num_heads, total_seq_len)
            new_row = trace.attn_weights[0, :, -1, :]  # (num_heads, total_seq_len)
            # Round to 4 decimals and convert to lists
            layer_data = [
                [round(float(v), 4) for v in head_row]
                for head_row in new_row
            ]
            attn_data.append(layer_data)
        payload["attention_new_rows"] = attn_data
    else:
        payload["attention_new_rows"] = None
    
    # 3. Logit Lens: top-1 per layer
    if include_lens and traces:
        from glassbox.trace import logit_lens
        lens_results = logit_lens(model, tokenizer, traces, position=-1)
        payload["lens_predictions"] = [
            {
                "layer": r.layer_idx,
                "token_str": r.top_token_str,
                "prob": round(r.top_prob, 4),
            }
            for r in lens_results
        ]
    else:
        payload["lens_predictions"] = None
    
    return payload