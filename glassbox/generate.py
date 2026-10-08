"""
GlassBox Generator — Token-by-token text generator with KV Cache toggles.
Made with by Vinit chaurasia
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Generator
import numpy as np

from glassbox.model import Transformer, ForwardOutput
from glassbox.cache import KVCache
from glassbox.ops import sample_next_token


@dataclass
class GenerationStep:
    """Contains metadata for a single generated token step."""
    token_id: int
    token_str: str
    tokens_per_sec: float
    is_cache_active: bool
    output_logits: np.ndarray  # For the logit lens & prob bar visualization
    traces: list


def generate_stream(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    use_cache: bool = True,
    trace: bool = False,
) -> Generator[GenerationStep, None, None]:
    """
    Generate text streaming one token at a time.
    Can run with or without KV cache to showcase speed comparison.
    """
    input_ids = tokenizer.encode(prompt)
    curr_ids = list(input_ids)
    
    # Initialize cache if requested
    cache = KVCache(model.config.num_hidden_layers) if use_cache else None
    
    # Prefill phase (process the whole prompt)
    start_time = time.perf_counter()
    
    prompt_array = np.array([curr_ids], dtype=np.int64)
    out: ForwardOutput = model.forward(prompt_array, cache=cache, trace=trace)
    
    # Sample first token
    next_logits = out.logits[0, -1, :]
    next_token = sample_next_token(next_logits, temperature, top_k, top_p)
    
    curr_ids.append(next_token)
    
    step_time = time.perf_counter() - start_time
    yield GenerationStep(
        token_id=next_token,
        token_str=tokenizer.decode(next_token),
        tokens_per_sec=1.0 / step_time if step_time > 0 else 0.0,
        is_cache_active=use_cache,
        output_logits=next_logits,
        traces=out.traces if trace else [],
    )
    
    # Decoding phase (token-by-token loop)
    for _ in range(max_new_tokens - 1):
        if next_token == tokenizer.eos_token_id:
            break
            
        step_start = time.perf_counter()
        
        if use_cache:
            # KV Cache Active: Only pass the single NEW token!
            input_feed = np.array([[next_token]], dtype=np.int64)
        else:
            # KV Cache Inactive: Pass the entire accumulated sequence!
            input_feed = np.array([curr_ids], dtype=np.int64)
            
        out = model.forward(input_feed, cache=cache, trace=trace)
        
        next_logits = out.logits[0, -1, :]
        next_token = sample_next_token(next_logits, temperature, top_k, top_p)
        
        curr_ids.append(next_token)
        
        step_time = time.perf_counter() - step_start
        
        yield GenerationStep(
            token_id=next_token,
            token_str=tokenizer.decode(next_token),
            tokens_per_sec=1.0 / step_time if step_time > 0 else 0.0,
            is_cache_active=use_cache,
            output_logits=next_logits,
            traces=out.traces if trace else [],
        )