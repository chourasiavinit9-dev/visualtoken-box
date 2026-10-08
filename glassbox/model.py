"""
GlassBox Model — From-scratch Transformer Forward Pass with KV Cache support.
Supports Llama architecture (SmolLM2-135M) with GQA, RoPE, and SwiGLU.
Made with 🔮 by Vinit chaursia
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from glassbox.loader import ModelConfig, load_all_weights
from glassbox.ops import rmsnorm, silu, softmax, precompute_rope, apply_rope, repeat_kv, causal_mask
from glassbox.cache import KVCache


@dataclass
class LayerTrace:
    """Stores inspectability data for visualizer panels."""
    layer_idx: int
    attn_weights: np.ndarray | None = None  # (batch, num_heads, seq_len, total_seq_len)
    hidden_state: np.ndarray | None = None  # (batch, seq_len, hidden_size)


@dataclass
class ForwardOutput:
    """Output of transformer forward pass."""
    logits: np.ndarray                      # (batch, seq_len, vocab_size)
    traces: list[LayerTrace] = field(default_factory=list)


class Transformer:
    """From-scratch NumPy Transformer engine with KV Cache support."""
    
    def __init__(self, model_dir: str | Path, max_seq_len: int = 2048):
        self.model_dir = Path(model_dir)
        self.config = ModelConfig.from_dir(self.model_dir)
        self.weights = load_all_weights(self.model_dir)
        
        # Precompute RoPE frequency tables
        self.cos_cache, self.sin_cache = precompute_rope(
            head_dim=self.config.head_dim,
            max_seq_len=max_seq_len,
            theta=self.config.rope_theta,
        )
        
        # Unpack embedding & output projection matrices
        self.embed_tokens = self.weights["model.embed_tokens.weight"]  # (vocab_size, hidden_size)
        self.final_norm_weight = self.weights["model.norm.weight"]
        
        if "lm_head.weight" in self.weights:
            self.lm_head = self.weights["lm_head.weight"]
        else:
            self.lm_head = self.embed_tokens

    def _attention(
        self,
        x: np.ndarray,
        layer_idx: int,
        cache: KVCache | None = None,
        capture_trace: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Multi-Head Attention with Grouped Query Attention (GQA), RoPE, and optional KVCache.
        """
        b, seq_len, _ = x.shape
        cfg = self.config
        
        # 1. Linear projections
        wq = self.weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
        wk = self.weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
        wv = self.weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
        wo = self.weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
        
        q = x @ wq.T  # (b, seq_len, num_heads * head_dim)
        k = x @ wk.T  # (b, seq_len, num_kv_heads * head_dim)
        v = x @ wv.T  # (b, seq_len, num_kv_heads * head_dim)
        
        # 2. Reshape into heads: (b, heads, seq_len, head_dim)
        q = q.reshape(b, seq_len, cfg.num_attention_heads, cfg.head_dim).swapaxes(1, 2)
        k = k.reshape(b, seq_len, cfg.num_key_value_heads, cfg.head_dim).swapaxes(1, 2)
        v = v.reshape(b, seq_len, cfg.num_key_value_heads, cfg.head_dim).swapaxes(1, 2)
        
        # 3. Apply RoPE (Offset RoPE frequencies if using cache during decoding)
        if cache is not None and cache.k_cache[layer_idx] is not None:
            pos_offset = cache.k_cache[layer_idx].shape[2]
            layer_cos = self.cos_cache[pos_offset : pos_offset + seq_len]
            layer_sin = self.sin_cache[pos_offset : pos_offset + seq_len]
        else:
            layer_cos = self.cos_cache[:seq_len]
            layer_sin = self.sin_cache[:seq_len]
            
        q, k = apply_rope(q, k, layer_cos, layer_sin)
        
        # 4. KV Cache Update
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
            
        total_seq_len = k.shape[2]
        
        # 5. GQA: Repeat KV heads to match Q heads
        k = repeat_kv(k, cfg.num_kv_groups)  # (b, num_heads, total_seq_len, head_dim)
        v = repeat_kv(v, cfg.num_kv_groups)  # (b, num_heads, total_seq_len, head_dim)
        
        # 6. Scaled Dot-Product Attention: (Q @ K.T) / sqrt(head_dim)
        scale = 1.0 / math.sqrt(cfg.head_dim)
        scores = (q @ k.swapaxes(-1, -2)) * scale  # (b, num_heads, seq_len, total_seq_len)
        
        # 7. Apply causal mask (only needed if generating multiple tokens at once)
        if seq_len > 1:
            mask = causal_mask(seq_len)
            scores = scores + mask
            
        # 8. Softmax over last dimension
        attn_probs = softmax(scores, axis=-1)
        
        # 9. Compute context
        context = attn_probs @ v  # (b, num_heads, seq_len, head_dim)
        
        # 10. Concatenate and project output
        context = context.swapaxes(1, 2).reshape(b, seq_len, cfg.hidden_size)
        out = context @ wo.T
        
        attn_trace = attn_probs if capture_trace else None
        return out, attn_trace

    def _mlp(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        w_gate = self.weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        w_up = self.weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        w_down = self.weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
        
        gate = silu(x @ w_gate.T)
        up = x @ w_up.T
        return (gate * up) @ w_down.T

    def forward(
        self,
        input_ids: np.ndarray,
        cache: KVCache | None = None,
        trace: bool = False,
    ) -> ForwardOutput:
        """
        Full Transformer forward pass. Supports KV Cache.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
            
        cfg = self.config
        traces = []
        
        # Embeddings lookup
        x = self.embed_tokens[input_ids]
        
        # Run through layers
        for i in range(cfg.num_hidden_layers):
            # Attention Block
            norm_attn = self.weights[f"model.layers.{i}.input_layernorm.weight"]
            x_norm = rmsnorm(x, norm_attn, eps=cfg.rms_norm_eps)
            attn_out, attn_weights = self._attention(
                x_norm,
                layer_idx=i,
                cache=cache,
                capture_trace=trace,
            )
            x = x + attn_out
            
            # MLP Block
            norm_mlp = self.weights[f"model.layers.{i}.post_attention_layernorm.weight"]
            x_norm = rmsnorm(x, norm_mlp, eps=cfg.rms_norm_eps)
            mlp_out = self._mlp(x_norm, layer_idx=i)
            x = x + mlp_out
            
            if trace:
                traces.append(LayerTrace(
                    layer_idx=i,
                    attn_weights=attn_weights,
                    hidden_state=x.copy(),
                ))
                
        # Final Norm & Projection
        x = rmsnorm(x, self.final_norm_weight, eps=cfg.rms_norm_eps)
        logits = x @ self.lm_head.T
        
        return ForwardOutput(logits=logits, traces=traces)