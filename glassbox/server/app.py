"""
GlassBox Server — FastAPI app with WebSocket streaming + static UI mount.
Run with: glassbox serve (or uvicorn glassbox.server.app:app)
Made with by Vinit Chaurasia
"""
import json
import time
from pathlib import Path
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from glassbox.model import Transformer
from glassbox.tokenizer import Tokenizer
from glassbox.generate import generate_stream
from glassbox.trace_payload import build_step_payload


# ─── Global Model Instance ──────────────────────────────────────
STATE = {
    "model": None,
    "tokenizer": None,
    "model_dir": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on server startup."""
    print("🔮 GlassBox Server starting...")
    model_dir = Path(__file__).parent.parent.parent / "models" / "SmolLM2-135M"
    print(f"📦 Loading model from {model_dir}...")
    STATE["model_dir"] = model_dir
    STATE["tokenizer"] = Tokenizer(model_dir)
    STATE["model"] = Transformer(model_dir)
    print("✅ Model loaded! Server ready at http://localhost:8000")
    yield
    print("👋 Server shutting down.")


app = FastAPI(title="GlassBox 🔮", lifespan=lifespan)


# ─── WebSocket: Generation Stream ──────────────────────────────
@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    try:
        # Receive generation params from client
        params = await ws.receive_json()
        prompt = params.get("prompt", "The quick brown fox jumps over the lazy dog.")
        max_new_tokens = int(params.get("max_new_tokens", 30))
        temperature = float(params.get("temperature", 0.7))
        top_k = int(params.get("top_k", 40))
        top_p = float(params.get("top_p", 0.9))
        use_cache = bool(params.get("use_cache", True))
        trace = bool(params.get("trace", True))

        model = STATE["model"]
        tok = STATE["tokenizer"]

        # Send initial metadata
        await ws.send_json({
            "type": "init",
            "config": {
                "num_layers": model.config.num_hidden_layers,
                "num_heads": model.config.num_attention_heads,
                "num_kv_heads": model.config.num_key_value_heads,
                "hidden_size": model.config.hidden_size,
                "vocab_size": model.config.vocab_size,
            },
            "prompt": prompt,
            "prompt_token_strs": [tok.decode(t) for t in tok.encode(prompt)],
        })

        # Stream generation
        step_idx = 0
        for step in generate_stream(
            model=model,
            tokenizer=tok,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
            trace=trace,
        ):
            payload = build_step_payload(
                step_idx=step_idx,
                token_id=step.token_id,
                token_str=step.token_str,
                tokens_per_sec=step.tokens_per_sec,
                cache_active=step.is_cache_active,
                logits=step.output_logits,
                traces=step.traces,
                tokenizer=tok,
                model=model,
                top_n=200,
                include_attention=trace,
                include_lens=trace,
            )
            payload["type"] = "step"
            await ws.send_json(payload)
            step_idx += 1

        await ws.send_json({"type": "done", "total_steps": step_idx})

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ─── Static Files: Serve the UI ────────────────────────────────
WEB_DIR = Path(__file__).parent.parent / "web"

@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")

# Mount everything else as static
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ─── Health Check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": STATE["model"] is not None}