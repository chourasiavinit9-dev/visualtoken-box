"""
gemma_engine/engine.py — Custom token-by-token generation loop for the Gemma engine.

Design goals
------------
* Manual generation loop (NOT model.generate) using the KV cache explicitly.
* apply_chat_template is called every turn so multi-turn history is always
  formatted correctly.
* Per-token yields a GemmaStep with: token text, id, top-k candidates,
  per-layer attention (last query position, avg over heads), logit lens
  (per layer top-1 after final norm + lm_head), and a logit soft-cap if the
  model config specifies one.
* output_hidden_states=True and output_attentions=True are passed to every
  forward call so we can read traces without monkey-patching.
* Sliding-window layers are detected from the model config if present.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TopKCandidate:
    token_id: int
    token_str: str
    prob: float


@dataclass
class LensEntry:
    layer: int
    token_str: str
    prob: float
    is_sliding_window: bool


@dataclass
class GemmaStep:
    """Everything the frontend needs for one generated token."""
    token_id: int
    token_str: str
    tokens_per_sec: float
    # Top-k next-token candidates (after temperature/top-p)
    top_k_candidates: list[TopKCandidate]
    # Attention: [num_layers][total_seq_len] — last query position, avg over heads
    attention_avg: list[list[float]]
    # Per-head attention: [num_layers][num_heads][total_seq_len]
    attention_per_head: list[list[list[float]]]
    # Logit lens: one entry per layer
    lens: list[LensEntry]
    # Entropy over full vocab (bits)
    entropy_bits: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Gemma 2+ logit soft-capping: logits = cap * tanh(logits / cap)."""
    return cap * torch.tanh(logits / cap)


def _logit_lens_entry(
    hidden: torch.Tensor,          # (1, hidden_size)
    final_norm,
    lm_head,
    processor,
    layer_idx: int,
    is_sw: bool,
    softcap: float | None,
) -> LensEntry:
    """Project a hidden state through final norm + lm_head, return top-1."""
    with torch.no_grad():
        normed = final_norm(hidden)                  # (1, hidden_size)
        logits = lm_head(normed).squeeze(0)          # (vocab_size,)
        if softcap:
            logits = _apply_softcap(logits, softcap)
        probs = torch.softmax(logits.float(), dim=-1)
        top_id = int(probs.argmax())
        token_str = processor.decode([top_id], skip_special_tokens=False)
    return LensEntry(
        layer=layer_idx,
        token_str=token_str,
        prob=float(probs[top_id]),
        is_sliding_window=is_sw,
    )


def _entropy_bits(probs: torch.Tensor) -> float:
    nz = probs[probs > 0]
    return float(-(nz * torch.log2(nz)).sum())


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class GemmaEngine:
    """Stateful chat engine with a multi-turn history.

    Usage
    -----
    engine = GemmaEngine(loader)
    for step in engine.chat("Hello, who are you?"):
        print(step.token_str, end="", flush=True)
    for step in engine.chat("What was my last question?"):
        ...
    engine.reset()
    """

    def __init__(self, loader) -> None:
        self._loader = loader
        self._history: list[dict] = []   # list of {role, content}
        self._detect_model_properties()

    def _detect_model_properties(self) -> None:
        cfg = self._loader.config
        # Logit soft-capping (Gemma 2 / Gemma 3 have this)
        self._softcap: float | None = getattr(cfg, "final_logit_softcapping", None)
        if self._softcap == 0.0:
            self._softcap = None

        # Number of layers
        self._num_layers: int = cfg.num_hidden_layers

        # Sliding-window layers: Gemma 3 stores sliding_window_pattern as a list
        # e.g. [True, True, False, True, True, False, ...]
        sw_pattern = getattr(cfg, "sliding_window_pattern", None)
        if isinstance(sw_pattern, (list, tuple)) and len(sw_pattern) == self._num_layers:
            self._sliding_window: list[bool] = [bool(v) for v in sw_pattern]
        elif isinstance(sw_pattern, int):
            # Older format: every `sw_pattern`-th layer is global
            self._sliding_window = [
                (i % sw_pattern != sw_pattern - 1) for i in range(self._num_layers)
            ]
        else:
            self._sliding_window = [False] * self._num_layers

        logger.info(
            "Gemma properties: layers=%d, softcap=%s, SW_layers=%d/%d",
            self._num_layers,
            self._softcap,
            sum(self._sliding_window),
            self._num_layers,
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    def chat(
        self,
        user_message: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k_trace: int = 10,
        greedy: bool = False,
        image_bytes: bytes | None = None,
    ) -> Generator[GemmaStep, None, None]:
        """Yield one GemmaStep per generated token."""
        import io
        from PIL import Image
        from gemma_engine.config import (
            MAX_NEW_TOKENS, TEMPERATURE, TOP_P, SYSTEM_PROMPT,
        )

        loader = self._loader
        loader.assert_loaded()
        model = loader.model
        processor = loader.processor
        device = loader.device

        max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        temperature = temperature if temperature is not None else TEMPERATURE
        top_p = top_p if top_p is not None else TOP_P
        if greedy:
            temperature = 0.0

        # Build full message list for apply_chat_template
        messages: list[dict] = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.extend(self._history)
        
        # Handle vision if an image is provided and model supports it
        images = None
        if image_bytes and loader.supports_vision:
            images = [Image.open(io.BytesIO(image_bytes)).convert("RGB")]
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_message}
                ]
            }
        else:
            user_turn = {"role": "user", "content": user_message}
            
        messages.append(user_turn)

        # Tokenise the full history every turn
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **({"images": images} if images else {})
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # Detect end-of-turn token id(s)
        eot_ids: set[int] = set()
        for attr in ("eos_token_id", "pad_token_id"):
            val = getattr(processor, attr, None) or getattr(
                getattr(processor, "tokenizer", None), attr, None
            )
            if val is not None:
                if isinstance(val, list):
                    eot_ids.update(val)
                else:
                    eot_ids.add(int(val))
        # Gemma uses <end_of_turn> as a stop token
        eot_str = processor.convert_tokens_to_ids("<end_of_turn>")
        if eot_str and eot_str != processor.unk_token_id:
            eot_ids.add(int(eot_str))

        # Helpers for the logit-lens projection
        try:
            # Most HF causal-LM models expose these at the top level
            final_norm = model.model.norm
            lm_head = model.lm_head
        except AttributeError:
            final_norm = None
            lm_head = None

        past_key_values = None
        generated_ids: list[int] = []
        full_answer: str = ""

        for step_idx in range(max_new_tokens):
            t0 = time.perf_counter()

            with torch.no_grad():
                if step_idx == 0:
                    # Prefill: full prompt
                    model_kwargs = {
                        "input_ids": input_ids,
                        "past_key_values": past_key_values,
                        "use_cache": True,
                        "output_attentions": True,
                        "output_hidden_states": True,
                        "return_dict": True,
                    }
                    if "pixel_values" in inputs:
                        model_kwargs["pixel_values"] = inputs["pixel_values"]
                    if "pixel_attention_mask" in inputs:
                        model_kwargs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
                    
                    out = model(**model_kwargs)
                else:
                    # Decode: single new token
                    out = model(
                        input_ids=torch.tensor([[next_token_id]], device=device),
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )

            past_key_values = out.past_key_values

            # ── Logits ──────────────────────────────────────────────────────
            logits: torch.Tensor = out.logits[0, -1, :].float()  # (vocab,)
            if self._softcap:
                logits = _apply_softcap(logits, self._softcap)

            # ── Sampling ────────────────────────────────────────────────────
            if greedy or temperature == 0.0:
                next_token_id = int(logits.argmax())
            else:
                scaled = logits / max(temperature, 1e-6)
                probs_full = torch.softmax(scaled, dim=-1)
                # top-p nucleus
                sorted_probs, sorted_ids = torch.sort(probs_full, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                cutoff = (cumulative - sorted_probs) < top_p
                sorted_probs[~cutoff] = 0.0
                sorted_probs /= sorted_probs.sum()
                chosen_pos = torch.multinomial(sorted_probs, 1).item()
                next_token_id = int(sorted_ids[chosen_pos])

            elapsed = time.perf_counter() - t0
            tokens_per_sec = 1.0 / elapsed if elapsed > 0 else 0.0

            token_str: str = processor.decode(
                [next_token_id], skip_special_tokens=False
            )
            generated_ids.append(next_token_id)
            full_answer += token_str

            # ── Top-k candidates ────────────────────────────────────────────
            topk_vals, topk_ids = torch.topk(logits, k=min(top_k_trace, len(logits)))
            top_k_candidates = [
                TopKCandidate(
                    token_id=int(tid),
                    token_str=processor.decode([int(tid)], skip_special_tokens=False),
                    prob=float(pv), # This is now actually the raw logit, despite the field name
                )
                for tid, pv in zip(topk_ids.tolist(), topk_vals.tolist())
            ]

            # ── Entropy ─────────────────────────────────────────────────────
            probs_for_trace = torch.softmax(logits, dim=-1)
            entropy = _entropy_bits(probs_for_trace)

            # ── Attention traces ─────────────────────────────────────────────
            # out.attentions: tuple of (batch, heads, seq_q, seq_k) per layer
            attention_avg: list[list[float]] = []
            attention_per_head: list[list[list[float]]] = []
            if out.attentions is not None:
                for layer_attn in out.attentions:
                    # layer_attn: (1, heads, seq_q, seq_k)
                    last_row = layer_attn[0, :, -1, :]      # (heads, seq_k)
                    avg_row = last_row.mean(dim=0)           # (seq_k,)
                    attention_avg.append(
                        [round(float(v), 4) for v in avg_row.tolist()]
                    )
                    attention_per_head.append(
                        [
                            [round(float(v), 4) for v in head_row.tolist()]
                            for head_row in last_row.tolist()
                        ]
                    )

            # ── Logit lens ──────────────────────────────────────────────────
            lens: list[LensEntry] = []
            if out.hidden_states is not None and final_norm is not None and lm_head is not None:
                # hidden_states: tuple of (batch, seq, hidden) — index 0 = embedding, 1..N = layers
                for layer_idx in range(self._num_layers):
                    hs = out.hidden_states[layer_idx + 1][0, -1:, :]  # (1, hidden)
                    entry = _logit_lens_entry(
                        hs, final_norm, lm_head, processor,
                        layer_idx=layer_idx,
                        is_sw=self._sliding_window[layer_idx],
                        softcap=self._softcap,
                    )
                    lens.append(entry)

            yield GemmaStep(
                token_id=next_token_id,
                token_str=token_str,
                tokens_per_sec=tokens_per_sec,
                top_k_candidates=top_k_candidates,
                attention_avg=attention_avg,
                attention_per_head=attention_per_head,
                lens=lens,
                entropy_bits=round(entropy, 3),
            )

            if next_token_id in eot_ids:
                break

        # Append this turn to history
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": full_answer})
