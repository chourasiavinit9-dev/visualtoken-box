🧩 Adding a New Model Architecture to VisualBox
VisualBox is designed so that adding a new model architecture (e.g., Qwen2, Gemma, Phi) is a single, self-contained file contribution!

Architecture Registry Layout
Each architecture lives in visualbox/arch/:

visualbox/arch/
├── init.py
├── base.py (Abstract base class)
├── llama.py (Llama / SmolLM2 reference implementation)
└── qwen2.py (Proposed contribution!)

How to add a model:
Inherit from BaseArchitecture in visualbox/arch/base.py.
Implement forward_layer(x, layer_idx) handling model-specific quirks (e.g., Qwen's attention biases).
Register your architecture in ARCH_REGISTRY.
Add a unit test comparing output against HuggingFace transformers in tests/test_against_hf.py.
Good first issues for hackathon contributors! 💜