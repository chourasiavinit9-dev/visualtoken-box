"""
GlassBox Trace — Interpretability utilities for peeking inside the model.
Logit Lens: Watch the model's answer form layer by layer!
Attention Analysis: Explore what each head is looking at!
Made with 🔮 by Vivi
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from glassbox.model import Transformer, LayerTrace
from glassbox.ops import rmsnorm, softmax


@dataclass
class LogitLensResult:
    """A single layer's prediction from the Logit Lens."""
    layer_idx: int
    top_token_id: int
    top_token_str: str
    top_prob: float
    top5_tokens: list[tuple[str, float]]  # List of (token_str, probability)


def logit_lens(
    model: Transformer,
    tokenizer,
    traces: list[LayerTrace],
    position: int = -1,
) -> list[LogitLensResult]:
    """
    Project every layer's hidden state through the final norm and output head
    to see what the model would predict if it stopped at each layer!
    
    This is a famous AI interpretability technique known as "Logit Lens".
    
    Args:
        model: Our Transformer engine
        tokenizer: Our tokenizer
        traces: List of LayerTrace objects from a forward pass (with trace=True)
        position: Which sequence position to analyze (-1 = last token)
    
    Returns:
        List of LogitLensResult, one per layer.
    """
    results = []
    
    for trace in traces:
        if trace.hidden_state is None:
            continue
            
        # 1. Extract hidden state for the chosen position
        # Shape: (batch, seq_len, hidden_size) → (hidden_size,)
        hidden = trace.hidden_state[0, position, :]
        
        # 2. Apply the final RMSNorm (reusing model's learned weights)
        hidden_normed = rmsnorm(
            hidden[np.newaxis, :],
            model.final_norm_weight,
            eps=model.config.rms_norm_eps,
        )[0]
        
        # 3. Project to vocabulary logits using LM head
        logits = hidden_normed @ model.lm_head.T  # Shape: (vocab_size,)
        
        # 4. Softmax to probabilities
        probs = softmax(logits)
        
        # 5. Extract top-1 and top-5
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_tokens = [
            (tokenizer.decode(int(idx)), float(probs[idx]))
            for idx in top5_indices
        ]
        
        top_id = int(top5_indices[0])
        results.append(LogitLensResult(
            layer_idx=trace.layer_idx,
            top_token_id=top_id,
            top_token_str=tokenizer.decode(top_id),
            top_prob=float(probs[top_id]),
            top5_tokens=top5_tokens,
        ))
    
    return results


def summarize_attention(trace: LayerTrace, head_idx: int | None = None) -> dict:
    """
    Summarize an attention pattern for a single layer.
    
    Args:
        trace: LayerTrace containing attn_weights
        head_idx: If given, analyze a specific head. If None, average across heads.
    
    Returns:
        Dictionary with summary stats.
    """
    if trace.attn_weights is None:
        return {}
    
    # trace.attn_weights shape: (batch, num_heads, seq_len, total_seq_len)
    attn = trace.attn_weights[0]  # Drop batch dim
    
    if head_idx is not None:
        attn = attn[head_idx:head_idx+1]  # Keep single head
    
    # Average across heads if multiple
    avg_attn = np.mean(attn, axis=0)  # (seq_len, total_seq_len)
    
    # Compute entropy (how focused is attention?)
    # Low entropy = focused on 1-2 tokens, High entropy = spread uniformly
    eps = 1e-9
    entropy = -np.sum(avg_attn * np.log(avg_attn + eps), axis=-1).mean()
    
    return {
        "layer_idx": trace.layer_idx,
        "num_heads": trace.attn_weights.shape[1],
        "seq_len": avg_attn.shape[0],
        "mean_entropy": float(entropy),
        "max_attention_score": float(np.max(avg_attn)),
    }