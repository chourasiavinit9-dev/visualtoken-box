from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glassbox.model import Transformer
from glassbox.tokenizer import Tokenizer
from glassbox.trace import rmsnorm

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"
tok = Tokenizer(MODEL_DIR)
model = Transformer(MODEL_DIR)
hf = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32).eval()

ids = tok.encode("Paris is in France. Rome is in")
out = model.forward(np.array([ids], dtype=np.int64), trace=True)
with torch.no_grad():
    hs = hf(torch.tensor([ids]), output_hidden_states=True).hidden_states  # embeddings + one per block

n = len(out.traces)
for i, tr in enumerate(out.traces):
    ours = tr.hidden_state[0]                      # (seq, hidden)
    ref = hs[i + 1][0].numpy()
    if i == n - 1:                                 # HF's last entry already has the final norm applied
        ours = rmsnorm(ours, model.final_norm_weight, eps=model.config.rms_norm_eps)
    diff = float(np.abs(ours - ref).max())
    scale = float(np.abs(ref).max())
    print(f"block {i+1:2d}: max diff {diff:.6f}  (relative {diff/scale:.2e})")
    