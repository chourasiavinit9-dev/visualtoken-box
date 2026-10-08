"""
GlassBox Loader — Parse safetensors files and model configs from scratch.
No external ML libraries. Just bytes, JSON, and NumPy. 🔮
"""
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Dict


# ─── Model Config ───────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Reads config.json so we never hardcode model dimensions."""
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    head_dim: int          # computed: hidden_size // num_attention_heads
    num_kv_groups: int     # computed: num_attention_heads // num_key_value_heads

    @classmethod
    def from_dir(cls, model_dir: Union[str, Path]) -> "ModelConfig":
        config_path = Path(model_dir) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        hidden = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        n_kv = cfg["num_key_value_heads"]

        return cls(
            hidden_size=hidden,
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            intermediate_size=cfg["intermediate_size"],
            vocab_size=cfg["vocab_size"],
            rope_theta=cfg.get("rope_theta", 10000.0),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            head_dim=hidden // n_heads,
            num_kv_groups=n_heads // n_kv,
        )

    def __repr__(self) -> str:
        return (
            f"ModelConfig(\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  layers={self.num_hidden_layers},\n"
            f"  heads={self.num_attention_heads} (kv={self.num_key_value_heads}, groups={self.num_kv_groups}),\n"
            f"  head_dim={self.head_dim},\n"
            f"  intermediate={self.intermediate_size},\n"
            f"  vocab={self.vocab_size},\n"
            f"  rope_theta={self.rope_theta},\n"
            f"  tied_embeddings={self.tie_word_embeddings}\n"
            f")"
        )


# ─── Safetensors Parser ────────────────────────────────────────

DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "I32": np.int32,
    "I64": np.int64,
    "I16": np.int16,
    "I8":  np.int8,
    "U8":  np.uint8,
    "BOOL": np.bool_,
}


def _bfloat16_to_float32(raw_bytes: bytes) -> np.ndarray:
    """
    Convert bfloat16 raw bytes to float32 NumPy array.
    
    bfloat16 is the TOP 16 bits of a float32.
    So we: read as uint16 → cast to uint32 → shift left 16 → view as float32.
    """
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def load_safetensors(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Parse a .safetensors file from scratch.
    
    Format:
      [8 bytes: header length (uint64 LE)]
      [N bytes: JSON header]
      [rest: raw tensor data]
    
    Returns dict mapping tensor name → float32 NumPy array.
    """
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    
    with open(filepath, "rb") as f:
        # 1. Read header length (8 bytes, little-endian uint64)
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        
        # 2. Read and parse JSON header
        header_json = f.read(header_len)
        header = json.loads(header_json)
        
        # 3. Data section starts right after header
        data_start = 8 + header_len
        
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            
            dtype_str = info["dtype"]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            
            # Read raw bytes for this tensor
            f.seek(data_start + start)
            raw = f.read(end - start)
            
            # Convert to NumPy array
            if dtype_str == "BF16":
                arr = _bfloat16_to_float32(raw).reshape(shape)
            elif dtype_str in DTYPE_MAP:
                arr = np.frombuffer(raw, dtype=DTYPE_MAP[dtype_str]).reshape(shape)
                if dtype_str == "F16":
                    arr = arr.astype(np.float32)
            else:
                raise ValueError(f"Unsupported dtype '{dtype_str}' for tensor '{name}'")
            
            tensors[name] = arr
    
    return tensors


def load_all_weights(model_dir: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load all weights from a model directory.
    Handles both single file and sharded (index.json) layouts.
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    
    if index_path.exists():
        # Sharded model: read index to find which file has which tensor
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        # Find unique shard files
        shard_files = sorted(set(index["weight_map"].values()))
        print(f"📦 Loading {len(shard_files)} weight shards...")
        
        all_tensors = {}
        for shard in shard_files:
            shard_path = model_dir / shard
            tensors = load_safetensors(shard_path)
            all_tensors.update(tensors)
            print(f"   ✅ {shard} ({len(tensors)} tensors)")
        
        return all_tensors
    else:
        # Single file
        safetensors_path = model_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"No model.safetensors found in {model_dir}. "
                f"Did you run download_model.py?"
            )
        print(f"📦 Loading weights from {safetensors_path.name}...")
        tensors = load_safetensors(safetensors_path)
        print(f"   ✅ {len(tensors)} tensors loaded")
        return tensors