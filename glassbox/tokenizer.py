"""
GlassBox Tokenizer — Tokenizer wrapper for encoding text and decoding tokens.
Made with 🔮 by Vinit chaurasia
"""
from __future__ import annotations
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    """Wrapper around HuggingFace tokenizer.json file."""
    
    def __init__(self, model_dir: str | Path):
        tokenizer_path = Path(model_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        
        self.tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        self.bos_token_id = self.tokenizer.token_to_id("<|im_start|>") or 1
        self.eos_token_id = self.tokenizer.token_to_id("<|im_end|>") or 0
    
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode a string into a list of token IDs."""
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        return tokens
    
    def decode(self, token_ids: list[int] | int) -> str:
        """Decode token ID(s) back into a string."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids)