"""
gemma_engine/loader.py — Load tokenizer/processor and model for the Gemma engine.

Rules:
- Uses attn_implementation="eager" (required for per-layer hidden-state access).
- bf16 on CUDA/MPS, fp32 on CPU (fp16 avoided: MPS doesn't support it reliably).
- Exposes `supports_vision` so engine.py can gate image handling.
- Gives a clear human-readable error if the model is gated and the user is not
  logged in to Hugging Face.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> str:
    """Return a concrete device string (never 'auto')."""
    if device_str != "auto":
        return device_str
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _resolve_dtype(device: str):
    """Choose dtype based on device capability."""
    try:
        import torch
        if device in ("cuda", "mps"):
            return torch.bfloat16
    except ImportError:
        pass
    return None  # let transformers default to float32


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

class GemmaLoader:
    """Holds the loaded model + processor/tokenizer.

    Attributes
    ----------
    model      : the Hugging Face model (AutoModelForCausalLM or vision-LM)
    processor  : AutoProcessor (covers both text-only and multimodal models)
    tokenizer  : same object as processor (for text-only access convenience)
    device     : resolved concrete device string
    dtype      : torch dtype used for the model
    supports_vision : True if the model accepts image inputs
    model_id   : the Hugging Face repo id that was loaded
    config     : the model's config object (for logit_softcapping, layer counts, etc.)
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.device: str = "cpu"
        self.dtype = None
        self.supports_vision: bool = False
        self.model_id: str = ""
        self.config: Any = None
        self._loaded: bool = False

    def load(self, model_id: str, device_str: str = "auto") -> None:
        """Load the model and processor.  Raises on auth / not-found errors."""
        try:
            import torch
            from transformers import AutoProcessor, AutoConfig
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for the Gemma engine.\n"
                "Install them with:  pip install torch transformers accelerate"
            ) from exc

        self.model_id = model_id
        self.device = _resolve_device(device_str)
        self.dtype = _resolve_dtype(self.device)

        logger.info("Loading config for %s …", model_id)
        try:
            cfg = AutoConfig.from_pretrained(model_id)
        except OSError as exc:
            msg = str(exc)
            if "gated" in msg or "401" in msg or "403" in msg or "token" in msg.lower():
                raise RuntimeError(
                    f"Model '{model_id}' is gated.\n"
                    "1. Accept the license at https://huggingface.co/google/gemma-3-4b-it\n"
                    "2. Run:  huggingface-cli login\n"
                    "   or set the HF_TOKEN environment variable."
                ) from exc
            raise

        self.config = cfg

        # Detect vision support from config architecture name
        arch = getattr(cfg, "architectures", [])
        arch_str = " ".join(arch).lower() if arch else ""
        model_type = getattr(cfg, "model_type", "").lower()
        self.supports_vision = any(
            kw in arch_str or kw in model_type
            for kw in ("vision", "multimodal", "paligemma", "gemma3", "siglip", "llava")
        )
        # gemma-3 text-only models also have model_type="gemma3" but no vision tower;
        # verify by checking for the vision config sub-object
        if self.supports_vision and not hasattr(cfg, "vision_config"):
            self.supports_vision = False

        logger.info(
            "Supports vision: %s | device: %s | dtype: %s",
            self.supports_vision, self.device, self.dtype,
        )

        # Load processor (handles both text-only and multimodal)
        logger.info("Loading processor …")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor  # convenience alias

        # Load model — choose the right Auto class
        logger.info("Loading model weights (this may download ~several GB) …")
        load_kwargs: dict = {
            "attn_implementation": "eager",   # required for hidden-state traces
            "output_hidden_states": True,     # not a load kwarg but confirmed below
        }
        if self.dtype is not None:
            load_kwargs["torch_dtype"] = self.dtype
        if self.device == "cpu":
            load_kwargs["device_map"] = "cpu"
        else:
            load_kwargs["device_map"] = "auto"

        if self.supports_vision:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **load_kwargs
            )
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, **load_kwargs
            )

        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully on device=%s.", self.device)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("GemmaLoader: model has not been loaded yet.")


# ---------------------------------------------------------------------------
# Module-level singleton (populated on first call to load_gemma())
# ---------------------------------------------------------------------------
_singleton: GemmaLoader | None = None


def load_gemma(model_id: str | None = None, device_str: str = "auto") -> GemmaLoader:
    """Load once and cache. Subsequent calls return the same instance."""
    global _singleton
    if _singleton is not None and _singleton.loaded:
        return _singleton
    from gemma_engine.config import GEMMA_MODEL, DEVICE
    _singleton = GemmaLoader()
    _singleton.load(
        model_id=model_id or GEMMA_MODEL,
        device_str=device_str or DEVICE,
    )
    return _singleton
