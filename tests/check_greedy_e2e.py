from pathlib import Path
import sys
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

prompt = "The capital of France is"
ours = [s.token_id for s in generate_stream(
    model, tok, prompt, max_new_tokens=25,
    temperature=1.0, top_k=1, top_p=1.0, use_cache=True)]

ids = torch.tensor([tok.encode(prompt)])
with torch.no_grad():
    out = hf.generate(ids, max_new_tokens=25, do_sample=False)
theirs = out[0, ids.shape[1]:].tolist()

n = min(len(ours), len(theirs))
print("identical:", ours[:n] == theirs[:n], "| lengths:", len(ours), len(theirs))
if ours[:n] != theirs[:n]:
    i = next(k for k in range(n) if ours[k] != theirs[k])
    print("first difference at step", i, ":", ours[i], "vs", theirs[i])