# 🔮 VisualToken Box: See Inside the Model

**By Vinit Chaurasia** — An interactive, zero-black-box Transformer inference engine built in pure NumPy, featuring a real-time web visualizer for internal token mechanics.

VisualToken Box runs the open-weight **SmolLM2-135M** locally and demystifies what happens during autoregressive generation: where attention focuses, which next tokens the model considers, and how predictions evolve across layers via Logit Lens. With **zero PyTorch runtime dependency**, the engine is designed for clarity, exploration, and interpretability.

---

## ✨ Why VisualToken Box?

Most modern language model interfaces treat LLMs as black-box APIs: prompt goes in, text comes out. VisualToken Box acts as an **interactive X-ray machine** for AI:
- **Inspectable Core**: Built from the ground up in pure Python & NumPy with Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU activations, and KV Caching.
- **Real-Time Instrumentation**: Streams internal layer traces and attention weights over WebSockets into a live dashboard without requiring heavy frontend build steps.
- **Interactive Probing**: Test hypotheses on the fly by branching generation paths, toggling attention sinks, and observing layer-by-layer representations.

---

## 🧭 Key Features

- 🔥 **Attention Heatmap**: Inspect token-to-token attention across all 30 layers and 9 attention heads. Toggle first-token attention sinks, normalize rows, or average across heads.
- 📊 **Next-Token Probabilities**: Explore the top 200 candidate tokens considered at each generation step, alongside their entropy and softmax probabilities.
- 🔀 **What-If Branching**: Click any candidate token to force its selection, re-route the generation tree, and compare alternative model paths.
- 🔭 **Logit Lens**: Project intermediate hidden states across all 30 transformer blocks through the unembedding head to see predictions form in real time.
- ⚡ **KV Cache Diagnostics**: Toggle KV caching on/off dynamically to compare inference latency, tokens-per-second, and memory footprint.
- 🌐 **Zero-Build Web UI**: Fast local FastAPI server streaming real-time visualization directly to vanilla HTML5/Canvas/CSS.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- ~270 MB disk space for model weights

### Installation & Launch

```bash
# Clone the repository
git clone https://github.com/chourasiavinit9-dev/visualtoken-box.git
cd visualtoken-box

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install lightweight dependencies
pip install -r requirements.txt

# Download model weights (~270MB SmolLM2-135M from Hugging Face)
python download_model.py

# Launch the visualizer server
python run_server.py
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser, enter your prompt, and watch the model generate!

---

## 🏗️ Architecture Overview

```text
Prompt -> Tokenizer -> Embeddings
              │
        30 Transformer Blocks
     [RMSNorm -> RoPE -> GQA Attention -> SwiGLU MLP]
              │
        Final RMSNorm -> LM Head Projection -> Logits
              │
        Temperature / Top-k / Top-p Sampling -> Selected Token
              │
   KV Cache (Cached Keys & Values for Step-by-Step Generation)
              │
   Trace Data (Attention Weights & Hidden States) -> FastAPI WebSocket
              └──> Canvas Heatmap & Logit Lens Dashboard
```

- **Parameters**: ~134.5 Million
- **Layers**: 30 Transformer Blocks
- **Hidden Dimension**: 576
- **Heads**: 9 Query Heads, 3 Key/Value Heads (Grouped-Query Attention)
- **Vocabulary Size**: 49,152 tokens

---

## 🌐 Community Wisdom & Provenance

Grounding model interpretability and transformer visualization in real-world engineering practices:

### 🌐 Community Wisdom: [Attention Mechanism Explained Visually: The Innovation That Made Modern AI Possible](https://dev.to/mangeshmandlik/attention-mechanism-explained-visually-the-innovation-that-made-modern-ai-possible-25cg)
> **Source**: [mangeshmandlik](https://dev.to/mangeshmandlik)
> **Tags**: `ai`, `llm`, `machinelearning`, `beginners`
>
> Modern LLM interpretability requires visualizing attention weights as dynamic token-to-token matrices rather than black-box embeddings. By exposing internal Query-Key-Value interactions step-by-step, developers can debug context decay and hallucination patterns in real time.
>
> 🔗 [Read Full Discussion](https://dev.to/mangeshmandlik/attention-mechanism-explained-visually-the-innovation-that-made-modern-ai-possible-25cg)

### 🌐 Community Wisdom: [Under the Hood of Transformer Mechanics, Attention Math, and Memory Bottlenecks](https://dev.to/abhishekninja_writer/under-the-hood-of-transformer-mechanics-attention-math-and-memory-bottlenecks-2b23)
> **Source**: [abhishekninja_writer](https://dev.to/abhishekninja_writer)
> **Tags**: `machinelearning`, `artificialintelligen`, `deeplearning`, `python`
>
> Moving past high-level abstractions to inspect raw tensor operations reveals how KV caching and grouped-query attention trade memory bandwidth for decoding throughput. A pure NumPy implementation provides unparalleled transparency into the memory mechanics powering modern autoregressive models.
>
> 🔗 [Read Full Discussion](https://dev.to/abhishekninja_writer/under-the-hood-of-transformer-mechanics-attention-math-and-memory-bottlenecks-2b23)

---

## 🗂️ Project Layout

```text
glassbox/          NumPy model core, tokenizer, generation, and tracing engine
  ├── model.py     Transformer forward pass with GQA, RoPE, and SwiGLU
  ├── ops.py       Raw NumPy numerical operations (rmsnorm, silu, softmax, RoPE)
  ├── cache.py     KV cache implementation for low-latency autoregressive decoding
  ├── trace.py     Logit Lens projection and attention statistics
  ├── server/      FastAPI WebSocket server streaming generation traces
  └── web/         Vanilla JS/Canvas real-time visualizer dashboard
download_model.py  Hugging Face snapshot downloader for SmolLM2-135M safetensors
run_server.py      Entrypoint to start the backend visualizer
```

---

## 📄 License & Credits

- **Author**: Vinit Chaurasia
- **License**: [MIT License](LICENSE)
- **Base Model**: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) by Hugging Face (HuggingFaceTB).