"""
GlassBox Ops — Pure NumPy implementations of transformer primitives & sampling.
Every function here is unit-testable on its own. No torch. No magic. 🔮
Made with 🔮 by Vivi
"""
import numpy as np


# ─── Normalization ──────────────────────────────────────────────

def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    RMS Normalization.
    Formula: x / sqrt(mean(x²) + eps) * weight
    """
    variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight


# ─── Activation & Normalization Functions ──────────────────────

def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Swish) activation: x * sigmoid(x).
    Used in the SwiGLU MLP of Llama-style models.
    """
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax (subtracts max to avoid overflow).
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ─── Rotary Position Embedding (RoPE) ─────────────────────────

def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute cos and sin caches for RoPE (Llama-style split-halves).
    """
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (2.0 * np.arange(0, half_dim, dtype=np.float32) / head_dim))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    
    cos_half = np.cos(angles)
    sin_half = np.sin(angles)
    
    # Duplicate to match full head_dim (Llama split-halves convention)
    cos_cache = np.concatenate([cos_half, cos_half], axis=-1)
    sin_cache = np.concatenate([sin_half, sin_half], axis=-1)
    
    return cos_cache, sin_cache


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    cos_cache: np.ndarray,
    sin_cache: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Rotary Position Embedding to Q and K tensors.
    Uses Llama-style split-halves rotation.
    """
    def rotate_half(x: np.ndarray) -> np.ndarray:
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return np.concatenate([-x2, x1], axis=-1)
    
    seq_len = q.shape[2]
    cos = cos_cache[:seq_len]
    sin = sin_cache[:seq_len]
    
    # Reshape for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos[np.newaxis, np.newaxis, :, :]
    sin = sin[np.newaxis, np.newaxis, :, :]
    
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    
    return q_out, k_out


# ─── Attention Helpers ─────────────────────────────────────────

def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """
    Repeat KV heads for Grouped Query Attention (GQA).
    """
    if n_rep == 1:
        return x
    
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, np.newaxis, :, :]
    x = np.broadcast_to(x, (batch, n_kv_heads, n_rep, seq_len, head_dim))
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


def causal_mask(seq_len: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Create an upper-triangular causal attention mask.
    """
    mask = np.full((seq_len, seq_len), -np.inf, dtype=dtype)
    mask = np.triu(mask, k=1)
    return mask[np.newaxis, np.newaxis, :, :]


# ─── Sampling Strategies ───────────────────────────────────────

def sample_next_token(
    logits: np.ndarray,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    """
    Apply Temperature, Top-K, and Top-P filtering to logits and sample next token.
    """
    if temperature <= 1e-6:
        # Greedy sampling
        return int(np.argmax(logits))
    
    # 1. Temperature scaling
    scaled_logits = logits / temperature
    
    # 2. Top-K filtering
    if top_k > 0 and top_k < len(scaled_logits):
        threshold = np.partition(scaled_logits, -top_k)[-top_k]
        scaled_logits[scaled_logits < threshold] = -np.inf
        
    # 3. Compute Softmax Probabilities
    probs = softmax(scaled_logits)
    
    # 4. Top-P (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Mask out tokens exceeding cumulative threshold
        mask_to_remove = cumulative_probs > top_p
        mask_to_remove[1:] = mask_to_remove[:-1].copy()
        mask_to_remove[0] = False
        
        indices_to_remove = sorted_indices[mask_to_remove]
        probs[indices_to_remove] = 0.0
        
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs = probs / sum_probs
        else:
            return int(np.argmax(logits))
            
    # 5. Categorical random sample
    try:
        return int(np.random.choice(len(probs), p=probs))
    except Exception:
        return int(np.argmax(logits))