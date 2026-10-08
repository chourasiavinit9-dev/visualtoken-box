# Architecture Reference

## System Overview

```
User Prompt
    │
    ▼
FastAPI Server (glassbox/server/app.py)
    │ HTTP POST /api/generate
    │
    ▼
GemmaEngine.chat() — gemma_engine/engine.py
    │
    │  For each generated token:
    │  ┌─────────────────────────────────────────────┐
    │  │ model.forward(                               │
    │  │   output_attentions=True,                    │
    │  │   output_hidden_states=True                  │
    │  │ )                                            │
    │  │                                              │
    │  │ ── Attention traces ──────────────────────── │
    │  │   out.attentions[layer]                      │
    │  │   → (batch, heads, seq_q, seq_k)             │
    │  │   → avg over heads → attention_avg           │
    │  │   → per-head → attention_per_head            │
    │  │                                              │
    │  │ ── Logit Lens ────────────────────────────── │
    │  │   out.hidden_states[layer+1][:, -1, :]       │
    │  │   → final_norm → lm_head → softmax → top-1   │
    │  │                                              │
    │  │ ── Sampling ─────────────────────────────── │
    │  │   logits → temperature → top-p nucleus       │
    │  │   → multinomial → next_token_id              │
    │  └─────────────────────────────────────────────┘
    │
    ▼
GemmaStep dataclass (JSON-serializable)
    │
    ▼
WebSocket stream → Browser
    │
    ▼
Vanilla JS Dashboard (glassbox/web/)
  ├── heatmap.js    — Canvas-based attention heatmap renderer
  ├── lens.js       — Logit lens bar chart
  └── index.html    — Main dashboard layout
```

---

## Data Structures

### `GemmaStep` (one per generated token)

```python
@dataclass
class GemmaStep:
    token_id: int
    token_str: str              # The generated token as a string
    tokens_per_sec: float       # Inference speed

    top_k_candidates: list[TopKCandidate]  # Top-10 next-token options

    # Attention: indexed as [layer][token_position]
    attention_avg: list[list[float]]

    # Per-head: [layer][head][token_position]
    attention_per_head: list[list[list[float]]]

    # Logit Lens: one entry per transformer layer
    lens: list[LensEntry]

    entropy_bits: float         # Uncertainty over full vocabulary
```

### `LensEntry` (one per layer per step)
```python
@dataclass
class LensEntry:
    layer: int
    token_str: str      # Top-1 prediction from this layer's hidden state
    prob: float         # Probability of that prediction
    is_sliding_window: bool  # Whether this is a SW-attention layer (Gemma 3)
```

---

## Why `output_attentions=True` Is Required

Standard HuggingFace `model.generate()` does **not** return attention tensors
unless explicitly requested. GlassBox bypasses `model.generate()` entirely and
runs a **manual token-by-token loop** in `engine.py` where every `model()`
call passes `output_attentions=True` and `output_hidden_states=True`.

This is the **core architectural decision** that makes interpretability possible.

---

## KV Cache Usage

GlassBox uses the HuggingFace native `past_key_values` KV cache:
- **Prefill step** (step_idx == 0): full prompt runs through all layers
- **Decode steps** (step_idx > 0): only the new token is processed; previous
  KV pairs are reused from `past_key_values`

This gives full inference speed while still capturing per-token traces.

---

## WebSocket Protocol

Messages sent from server → browser per token:

```json
{
  "type": "token",
  "data": {
    "token_id": 1234,
    "token_str": " Paris",
    "tokens_per_sec": 12.4,
    "entropy_bits": 3.21,
    "top_k_candidates": [...],
    "attention_avg": [[...], ...],
    "attention_per_head": [[[...]], ...],
    "lens": [{"layer": 0, "token_str": "...", "prob": 0.12}, ...]
  }
}
```

Final message: `{"type": "done"}`
