from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbox.model import Transformer
from glassbox.tokenizer import Tokenizer
from glassbox.generate import generate_stream

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"
tok = Tokenizer(MODEL_DIR)
model = Transformer(MODEL_DIR)
hf = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32).eval()

prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 12)
kw = dict(max_new_tokens=3, temperature=1.0, top_k=1, top_p=1.0)

on = list(generate_stream(model, tok, prompt, use_cache=True, **kw))
off = list(generate_stream(model, tok, prompt, use_cache=False, **kw))

prompt_ids = tok.encode(prompt)
gen_ids = [s.token_id for s in on]
for k in range(len(on)):
    seq = prompt_ids + gen_ids[:k]          # the sequence whose last position predicts token k
    with torch.no_grad():
        ref = hf(torch.tensor([seq])).logits[0, -1].numpy()
    d_on = float(np.abs(on[k].output_logits - ref).max())
    d_off = float(np.abs(off[k].output_logits - ref).max())
    print(f"step {k}: cache vs HF {d_on:.5f} | no-cache vs HF {d_off:.5f}")