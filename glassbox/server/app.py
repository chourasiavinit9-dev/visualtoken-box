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
    model_dir = Path(__file__).parent.parent.parent / "models" / "SmolLM2-135M-Instruct"
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
        engine_type = params.get("engine", "glassbox")

        image_b64 = params.get("image_b64")
        image_bytes = None
        if image_b64:
            import base64
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)

        if engine_type == "gemma":
            try:
                from gemma_engine.loader import load_gemma
                from gemma_engine.engine import GemmaEngine
            except ImportError:
                await ws.send_json({"type": "error", "message": "Gemma engine not installed."})
                return

            gemma_loader = load_gemma()
            gemma_engine = GemmaEngine(gemma_loader)

            await ws.send_json({
                "type": "init",
                "config": {
                    "num_layers": gemma_loader.config.num_hidden_layers,
                    "num_heads": getattr(gemma_loader.config, "num_attention_heads", 0),
                    "num_kv_heads": getattr(gemma_loader.config, "num_key_value_heads", 0),
                    "hidden_size": gemma_loader.config.hidden_size,
                    "vocab_size": gemma_loader.config.vocab_size,
                },
                "prompt": prompt,
                "prompt_token_strs": [],
            })

            step_idx = 0
            for step in gemma_engine.chat(
                user_message=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k_trace=top_k,
                greedy=(temperature == 0.0),
                image_bytes=image_bytes,
            ):
                payload = {
                    "type": "step",
                    "step": step_idx,
                    "token_id": step.token_id,
                    "token_str": step.token_str,
                    "tokens_per_sec": step.tokens_per_sec,
                    "cache_active": True,
                    "entropy_bits": step.entropy_bits,
                    "top_logit_ids": [c.token_id for c in step.top_k_candidates],
                    "top_logit_values": [c.prob for c in step.top_k_candidates],
                    "top_logit_strs": [c.token_str for c in step.top_k_candidates],
                    "attention_new_rows": step.attention_per_head if trace else None,
                    "lens_predictions": [
                        {"layer": l.layer, "token_str": l.token_str, "prob": l.prob, "is_sliding_window": getattr(l, "is_sliding_window", False)}
                        for l in step.lens
                    ] if trace else None
                }
                await ws.send_json(payload)
                step_idx += 1

            await ws.send_json({"type": "done", "total_steps": step_idx})
            return

        # --- Default GlassBox Engine ---
        model = STATE["model"]
        tok = STATE["tokenizer"]

        # Wrap in SmolLM2-Instruct chat template so the model answers questions
        # instead of doing raw text completion.
        SYSTEM = "You are a helpful AI assistant. Answer questions clearly and concisely."
        formatted_prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Send initial metadata (show the original user prompt to UI, not the template)
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

        # Stream generation using the formatted prompt
        # stop on <|im_end|> (token 2) to prevent looping
        step_idx = 0
        for step in generate_stream(
            model=model,
            tokenizer=tok,
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
            trace=trace,
            stop_token_ids={2},  # <|im_end|> stops the assistant turn
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