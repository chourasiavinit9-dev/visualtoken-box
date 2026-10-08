# Swapping the LLM — Model-Specific Config Guide

GlassBox works with **any HuggingFace `AutoModelForCausalLM`** model.
Just change one line in `gemma_engine/config.py`.

---

## Quick Config Snippets

### SmolLM2-135M (Default — CPU-friendly, ~270 MB)
```python
GEMMA_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "auto"
```

### Gemma 3 4B / 12B (Requires HF login)
```python
GEMMA_MODEL = "google/gemma-3-4b-it"
DEVICE = "auto"        # uses bfloat16 on CUDA/MPS automatically
MAX_NEW_TOKENS = 300
```
> Run `huggingface-cli login` first and accept the license at https://huggingface.co/google/gemma-3-4b-it

### Mistral 7B Instruct
```python
GEMMA_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "auto"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
```

### Llama 3.2 3B Instruct (Requires HF login)
```python
GEMMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "auto"
```
> Accept license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### Phi-3.5 Mini (Microsoft, great on laptop)
```python
GEMMA_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEVICE = "auto"
```

### Qwen2.5 7B Instruct
```python
GEMMA_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "auto"
```

### GPT-2 (No login, instant test)
```python
GEMMA_MODEL = "gpt2"
DEVICE = "cpu"
MAX_NEW_TOKENS = 100
```
> Note: GPT-2 does not use a chat template. The system prompt will be ignored.

### Falcon 7B Instruct
```python
GEMMA_MODEL = "tiiuae/falcon-7b-instruct"
DEVICE = "auto"
```
> If loading fails, add `trust_remote_code=True` to loader.py's `from_pretrained` calls.

---

## How the Loader Auto-Detects Vision Models

[`gemma_engine/loader.py`](../../gemma_engine/loader.py) checks the model config
`architectures` field for vision-related keywords:
`"vision"`, `"multimodal"`, `"paligemma"`, `"gemma3"`, `"siglip"`, `"llava"`.

If detected **and** the config has a `vision_config` sub-object, it uses
`AutoModelForImageTextToText`. Otherwise it uses `AutoModelForCausalLM`.

**You do not need to set anything for vision models** — they are auto-detected.

---

## For Gated Models (Gemma, Llama 3, etc.)

```bash
# One-time setup
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

Or set the environment variable:
```bash
export HF_TOKEN="hf_your_token_here"
python run_server.py
```

---

## Memory Requirements (Rough Guide)

| Model Size | FP32 RAM | BF16 VRAM (GPU) |
|---|---|---|
| 135M | ~0.5 GB | ~0.3 GB |
| 1-3B | 4-12 GB | 2-6 GB |
| 7B | 28 GB | 14 GB |
| 12B | 48 GB | 24 GB |

> **Tip**: On Apple Silicon Macs, `DEVICE="auto"` picks MPS automatically, giving 2-4× speedup over CPU with shared unified memory.
