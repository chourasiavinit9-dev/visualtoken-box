# Troubleshooting Guide

---

## Model Loading Errors

### ❌ `Model 'xyz' is gated`
**Cause**: The model requires accepting a license on HuggingFace.

**Fix**:
```bash
# 1. Accept the license at: https://huggingface.co/<model-id>
# 2. Log in:
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

Or set the env variable:
```bash
export HF_TOKEN="hf_your_token_here"
```

---

### ❌ `CUDA out of memory`
**Cause**: Model is too large for your GPU.

**Fix options**:
1. Use a smaller model (e.g., SmolLM2-135M instead of Llama-7B)
2. Force CPU: set `DEVICE = "cpu"` in `gemma_engine/config.py`
3. Use 4-bit quantization (advanced — requires `bitsandbytes`):
   ```python
   # In gemma_engine/loader.py, add to load_kwargs:
   load_kwargs["load_in_4bit"] = True
   ```

---

### ❌ `attention_weights is None` / empty heatmap
**Cause**: The model is using Flash Attention or SDPA which does not return
attention tensors.

**Fix**: `attn_implementation="eager"` must be set. Verify in
`gemma_engine/loader.py`:
```python
load_kwargs = {
    "attn_implementation": "eager",   # <-- this must be present
    ...
}
```

---

### ❌ `trust_remote_code` error (Falcon, some custom models)
**Fix**: In `gemma_engine/loader.py`, add to the `from_pretrained` call:
```python
self.model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,   # add this
    **load_kwargs
)
```

---

### ❌ Chat template error / `apply_chat_template` fails
**Cause**: Some models (GPT-2, older BERT-style) don't have a chat template.

**Fix**: Open `gemma_engine/config.py` and set:
```python
SYSTEM_PROMPT = ""   # disable system prompt for non-instruct models
```

If it still fails, the model may not support `apply_chat_template`. For GPT-2,
the raw prompt is used without any template formatting.

---

## Server Errors

### ❌ Port 8000 already in use
```bash
lsof -ti:8000 | xargs kill -9
python run_server.py
```

### ❌ WebSocket disconnects immediately
**Cause**: Model is not yet loaded when the WebSocket connects.

**Fix**: Wait for the console to print `"Model loaded successfully"` before
sending a prompt. The UI should show a loading spinner automatically.

---

## Performance Issues

### Very slow generation (< 1 token/sec)
- On CPU this is normal for models > 1B parameters
- Try enabling MPS (Apple Silicon): `DEVICE = "mps"` in config
- Try a smaller model (SmolLM2-135M runs at 30+ tok/s on CPU)

### Dashboard is laggy
- Large attention matrices (many layers × many heads × long sequences) are
  computationally expensive to render
- Reduce `MAX_NEW_TOKENS` or shorten your prompts
- Use the "Average Heads" view instead of per-head view

---

## Testing

Run the self-test suite to diagnose issues:
```bash
pytest tests/ -v
python test_gemma.py          # basic engine smoke test
```
