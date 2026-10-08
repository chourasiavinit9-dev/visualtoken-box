"""
GlassBox KV Cache — Store keys and values of past tokens to speed up decoding.
Made with 🔮 by Vinit chaurasia
"""
from __future__ import annotations
import numpy as np


class KVCache:
    """Stores key/value states per layer for token-by-token generation."""
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # Create empty caches for each layer: (K_cache, V_cache)
        self.k_cache: list[np.ndarray | None] = [None] * num_layers
        self.v_cache: list[np.ndarray | None] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        new_k: np.ndarray,
        new_v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Append new key & value states to the existing cache for a given layer.
        
        Args:
            layer_idx: Current transformer layer
            new_k: Key state of current step (batch, kv_heads, seq_len_new, head_dim)
            new_v: Value state of current step (batch, kv_heads, seq_len_new, head_dim)
        
        Returns:
            Concatenated (all_k, all_v) cache matrices up to current token.
        """
        if self.k_cache[layer_idx] is None:
            # First token (prefill phase)
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
        else:
            # Decoding subsequent tokens: concatenate along sequence length dimension (axis 2)
            self.k_cache[layer_idx] = np.concatenate([self.k_cache[layer_idx], new_k], axis=2)
            self.v_cache[layer_idx] = np.concatenate([self.v_cache[layer_idx], new_v], axis=2)
            
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        """Reset the cache for a new generation run."""
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers

    @property
    def current_length(self) -> int:
        """Get the number of tokens cached in layer 0."""
        if self.k_cache[0] is None:
            return 0
        return self.k_cache[0].shape[2]