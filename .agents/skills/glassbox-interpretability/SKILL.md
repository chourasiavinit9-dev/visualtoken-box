---
name: glassbox-interpretability
description: >-
  GlassBox LLM Interpretability Skill — adds live mechanistic interpretability
  to ANY HuggingFace-compatible language model. Use this skill when the user
  wants to visualize attention heatmaps, logit-lens projections, next-token
  entropy, top-k candidate tokens, or activation steering for their own LLM.
  Also use when the user asks to "see inside" a model, run GlassBox, swap the
  model to a different one, or add interpretability tooling to a project.
---

# GlassBox Interpretability Skill

GlassBox is a **model-agnostic** mechanistic interpretability engine + real-time
web dashboard. It works with **any HuggingFace `AutoModelForCausalLM`** or
`AutoModelForImageTextToText` model — not just Gemma.

---

## What This Skill Covers

| Feature | What It Shows |
|---|---|
| 🔥 Attention Heatmap | Per-layer, per-head token→token attention weights |
| 🔭 Logit Lens | How each layer's hidden state predicts the final token |
| 📊 Top-K Candidates | Which tokens the model was "thinking about" at each step |
| ⚡ Entropy Meter | Model's uncertainty (bits) over the vocabulary |
| 🔀 What-If Branching | Force any candidate token and re-route generation |

---

## Quick Tasks

- **[Swap LLM model →](./references/swap_model.md)**
- **[Run setup + server →](./references/setup.md)**
- **[Understand the architecture →](./references/architecture.md)**
- **[Interpret what you see →](./references/interpreting_output.md)**
- **[Troubleshooting →](./references/troubleshooting.md)**

---

## Step-by-Step: Adding GlassBox to Any Project

### Step 1 — Clone & Install

```bash
git clone https://github.com/chourasiavinit9-dev/visualtoken-box.git
cd visualtoken-box
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Verify the install:
```bash
python -c "from gemma_engine.loader import GemmaLoader; print('OK')"
```

### Step 2 — Point to Your Model (one line change)

**Option A — Environment variable (recommended, no code edit):**
```bash
# Copy the template and edit it
cp .env.example .env
# Then edit .env and set GLASSBOX_MODEL to your model ID
```

**Option B — Edit the config file directly:**

Open [`gemma_engine/config.py`](../../gemma_engine/config.py) and change:

```python
GLASSBOX_MODEL = "your-org/your-model-id"   # e.g. "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "auto"                              # "cuda" | "mps" | "cpu" | "auto"
```

### Step 3 — Launch the Server

```bash
python run_server.py
```

Open **http://localhost:8000** in any browser. Type a prompt and watch GlassBox
stream attention heatmaps and logit-lens projections live.

### Step 4 — Verify It's Working

Run the built-in health check:
```bash
python scripts/health_check.py
```

Expected output: `✅ Server healthy | model loaded | WebSocket OK`

---

## Supported Model Families

Read [`references/swap_model.md`](./references/swap_model.md) for model-specific
config snippets. Quick compatibility table:

| Model Family | Works? | Notes |
|---|---|---|
| Gemma 2 / 3 (text) | ✅ Yes | Original target, best tested |
| SmolLM2 / SmolLM3 | ✅ Yes | Lightweight, runs on CPU |
| Mistral / Mixtral | ✅ Yes | Set `GEMMA_MODEL` in config |
| Llama 3.x Instruct | ✅ Yes | Requires HF login (gated) |
| Phi-3 / Phi-4 | ✅ Yes | Set `attn_implementation="eager"` |
| Falcon | ✅ Yes | May need `trust_remote_code=True` |
| Qwen 2.5 | ✅ Yes | |
| PaliGemma / LLaVA | ✅ Yes | Vision support auto-detected |
| GPT-2 / DistilGPT-2 | ✅ Yes | Great for quick local tests |

---

## Key Files

```
gemma_engine/
  config.py        ← Change GEMMA_MODEL here to use your LLM
  engine.py        ← Core token-by-token generation + trace capture
  loader.py        ← HuggingFace AutoModel loader (model-agnostic)
glassbox/
  server/app.py    ← FastAPI + WebSocket server
  web/             ← Vanilla JS dashboard (no build step)
run_server.py      ← Entrypoint
```

---

## Architecture Decision: Why `attn_implementation="eager"`

GlassBox **requires** `attn_implementation="eager"` in the loader because:
- Flash Attention 2 and SDPA do not return intermediate attention weight tensors
- `eager` mode returns full `(batch, heads, seq_q, seq_k)` tensors per layer
- This is set automatically in [`gemma_engine/loader.py`](../../gemma_engine/loader.py)

You do **not** need to change this — it is already configured correctly.
