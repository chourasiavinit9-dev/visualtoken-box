# 🔮 VisualBox (VisualToken Box): See Inside the Model

**By Vinit Chaurasia & Avijit** — An interactive, real-time mechanistic interpretability platform for Large Language Models, turning opaque model inference into a transparent, living circuit diagram.

Powered by **Google Gemma 2** (`unsloth/gemma-2-2b-it`) as its core flagship model, alongside an educational zero-black-box pure-NumPy inference engine, VisualBox lets you perform a live "MRI scan" on a neural network while it generates every single token.

---

## ✨ Why VisualBox?

Most modern language model interfaces treat LLMs as black-box APIs: prompt goes in, text comes out. VisualBox acts as an **interactive X-ray machine** for AI:
- **Google Gemma Powered**: Step-by-step autoregressive generation on Google Gemma 2 with full tensor observability across all attention heads and transformer layers.
- **Logit Lens in Real Time**: Intercept residual stream hidden states at intermediate layers and project them straight through the unembedding head — literally watch the model "change its mind" mid-computation as reasoning forms.
- **Attention Matrix Telemetry**: Unroll multi-head and sliding-window attention weights per token, per layer, and per head over live WebSockets.
- **Dual-Engine Architecture**: Run high-fidelity HuggingFace models (**Google Gemma**, Qwen, Mistral, Llama) or inspect an educational, zero-PyTorch pure-NumPy Transformer engine.

---

## 🔬 The 3-Stage VisualBox Pipeline

### 1️⃣ Tokenization Transformation Pipeline
Before intelligence emerges, raw text is dissected into subword representations across Gemma’s massive 256k-token vocabulary. VisualBox exposes the exact moment human semantics are translated into discrete token IDs and high-dimensional vector representations — showing how language models navigate coordinate spaces rather than reading words.

### 2️⃣ The Transformer Engine (Powered by Google Gemma)
Standard generation pipelines run inference blindly behind closed APIs. VisualBox bypasses black-box routines to run a custom step-by-step decoding loop on **Google Gemma**:
* **Live Hidden State Extraction**: Intercepting residual stream vectors layer-by-layer (`Layer 0 → Layer N`).
* **Logit-Lens Projections**: Projecting intermediate hidden states into vocabulary logits *before* the model finishes computing. Observe crude initial guesses in early layers sharpen into coherent, contextual reasoning in deeper layers.
* **Attention Extraction**: Unrolling multi-head & sliding-window attention matrices to trace exactly which past tokens each attention head is attending to.
* **Entropy & Uncertainty Dynamics**: Measuring vocabulary entropy in bits at every generation step to pinpoint the exact moment the model hesitates or experiences high uncertainty.

### 3️⃣ WebSocket Telemetry & Real-Time Visualization
Inference data streams instantly over high-speed WebSockets into a bespoke, GPU-accelerated Canvas/CSS dashboard:
* 🔥 **Dynamic Attention Heatmaps**: Watch query-key affinities flash in real time. Switch between all layers and individual heads, or view head-averaged attention.
* 🔭 **Logit-Lens Predictions**: Inspect top candidate tokens layer-by-layer as they battle for dominance before a token is sampled.
* 📊 **Next-Token Probability Radar**: Explore the top candidate tokens considered at each generation step, alongside their entropy and softmax probabilities.
* 🔀 **What-If Branching**: Click any candidate token to force its selection, re-route the generation tree, and compare alternative model paths.
* ⚡ **KV Cache Diagnostics**: Toggle KV caching on/off dynamically to compare inference latency, tokens-per-second, and memory footprint.

---

## 🧭 Flagship Model: Google Gemma 2

VisualBox is optimized out-of-the-box for **Google Gemma 2 (2B-IT)** (`unsloth/gemma-2-2b-it`):
- **Architecture**: 26 Transformer Layers, 8 Query Heads, 4 KV Heads (Grouped-Query Attention), Hidden Size 2304.
- **Vocabulary**: 256,000 subword tokens with byte-level fallback.
- **Attention Mechanics**: Alternating local sliding-window attention (4096 tokens) and global attention.
- **Activation Function**: GeGLU / RMSNorm with logit soft-capping.
- **Native Prompting**: Direct user-turn formatting without unsupported system roles, preventing prompt rejection.

> 💡 **Model Agnostic**: You can also run any other HuggingFace CausalLM model or the lightweight pure-NumPy **SmolLM2-135M** engine by simply changing `.env`.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- 4GB+ RAM (Google Gemma 2 2B runs efficiently in 4-bit / 8-bit / bfloat16)

### Installation & Launch

```bash
# 1. Clone the repository
git clone https://github.com/chourasiavinit9-dev/visualtoken-box.git
cd visualtoken-box

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the visualizer server
python run_server.py
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser, enter your prompt, and watch Gemma think!

---

## ⚙️ Configuration & Model Switching

Configure the active engine and generation hyperparameters via `.env` or environment variables:

```bash
# Google Gemma 2 (Default Flagship)
VISUALBOX_MODEL=unsloth/gemma-2-2b-it

# Hardware device ("auto", "cpu", "cuda", "mps")
DEVICE=auto

# Generation settings
MAX_NEW_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=10
```

To switch to other models:
```bash
# Pure NumPy educational engine (SmolLM2-135M)
python download_model.py  # Download weights (~270MB)
# Select "NumPy (SmolLM2-135M)" from the dashboard dropdown

# Or specify any HuggingFace model:
VISUALBOX_MODEL=Qwen/Qwen2.5-0.5B-Instruct python run_server.py
```

---

## 🏗️ Architecture & Tracing Flow

```text
User Prompt ──> Tokenizer (Gemma 256k Vocab)
                     │
         [Token IDs & High-Dim Embeddings]
                     │
          Gemma 2 Transformer Blocks (0 to 25)
      ┌───────────────────────────────────────────┐
      │  • RMSNorm & RoPE                         │
      │  • Sliding-Window & Global GQA Attention  │
      │  • GeGLU Feed-Forward Network             │
      │  • Logit Soft-Capping                     │
      └───────────────────────────────────────────┘
          │ (Residual Streams Intercepted)
          ├────────────────────────────────────────┐
          ▼                                        ▼
   Final RMSNorm & LM Head                Logit Lens Projections
          │                               (Layer 0..N -> Vocabulary)
          ▼                                        ▼
   Top-k Token Sampling & Entropy          Candidate Battle Per Layer
          │                                        │
          └──────────────────┬─────────────────────┘
                             ▼
              FastAPI WebSocket Telemetry
                             │
                             ▼
          Real-Time HTML5 / Canvas Instrument Panel
```

---

## 🗂️ Project Layout

```text
visualbox-main/
├── gemma_engine/        Google Gemma & HuggingFace inference & tracing engine
│   ├── engine.py        Step-by-step decoding loop, attention extraction, Logit Lens
│   ├── loader.py        Hugging Face model & tokenizer loader with device mapping
│   ├── config.py        Environment & hyperparameter configuration
│   └── __init__.py      Public engine APIs
├── visualbox/           Pure-NumPy model core & visualization backend
│   ├── model.py         Pure NumPy transformer implementation with GQA & RoPE
│   ├── ops.py           NumPy tensor operations (rmsnorm, silu, softmax, rope)
│   ├── cache.py         KV cache implementation for low-latency autoregressive decoding
│   ├── trace.py         Logit Lens projection and attention statistics
│   ├── server/          FastAPI WebSocket server streaming generation traces
│   └── web/             Vanilla JS/Canvas real-time visualizer dashboard
├── download_model.py    Downloader for SmolLM2-135M safetensors (NumPy engine)
├── run_server.py        Unified server runner
└── requirements.txt     Project dependencies
```

---

## 🌐 Community Wisdom & Provenance

Grounding model interpretability and transformer visualization in real-world engineering practices:

### 🌐 Community Wisdom: [Attention Mechanism Explained Visually: The Innovation That Made Modern AI Possible](https://dev.to/mangeshmandlik/attention-mechanism-explained-visually-the-innovation-that-made-modern-ai-possible-25cg)
> **Source**: [mangeshmandlik](https://dev.to/mangeshmandlik)
>
> Modern LLM interpretability requires visualizing attention weights as dynamic token-to-token matrices rather than black-box embeddings. By exposing internal Query-Key-Value interactions step-by-step, developers can debug context decay and hallucination patterns in real time.
>
> 🔗 [Read Full Discussion](https://dev.to/mangeshmandlik/attention-mechanism-explained-visually-the-innovation-that-made-modern-ai-possible-25cg)

### 🌐 Community Wisdom: [Under the Hood of Transformer Mechanics, Attention Math, and Memory Bottlenecks](https://dev.to/abhishekninja_writer/under-the-hood-of-transformer-mechanics-attention-math-and-memory-bottlenecks-2b23)
> **Source**: [abhishekninja_writer](https://dev.to/abhishekninja_writer)
>
> Moving past high-level abstractions to inspect raw tensor operations reveals how KV caching and grouped-query attention trade memory bandwidth for decoding throughput. A pure NumPy implementation provides unparalleled transparency into the memory mechanics powering modern autoregressive models.
>
> 🔗 [Read Full Discussion](https://dev.to/abhishekninja_writer/under-the-hood-of-transformer-mechanics-attention-math-and-memory-bottlenecks-2b23)

---

## 📄 License & Credits

- **Authors**: Vinit Chaurasia & Avijit
- **License**: [MIT License](LICENSE)
- **Flagship Model**: [Google Gemma 2](https://huggingface.co/google/gemma-2-2b-it) by Google DeepMind (`unsloth/gemma-2-2b-it`).
- **Educational Engine**: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) by Hugging Face (HuggingFaceTB).