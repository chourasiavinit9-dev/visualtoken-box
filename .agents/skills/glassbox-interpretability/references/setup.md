# Setup & Running the Server

## Prerequisites

- Python 3.9+ (3.11 recommended)
- ~500 MB disk for default SmolLM2-135M model
- Internet access for first-time model download

---

## Full Setup

```bash
# 1. Clone
git clone https://github.com/chourasiavinit9-dev/visualtoken-box.git
cd visualtoken-box

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: .\\venv\\Scripts\\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install dev tools for testing
pip install -r requirements-dev.txt
```

---

## Launching the Server

```bash
python run_server.py
```

The server starts on **http://localhost:8000** by default.

To use a different port:
```bash
python run_server.py --port 9000
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `GLASSBOX_PORT` | `8000` | Server port |
| `GLASSBOX_HOST` | `0.0.0.0` | Bind address |

Example:
```bash
HF_TOKEN=hf_xxx python run_server.py
```

---

## Health Check

After starting the server, verify everything works:

```bash
python .agents/skills/glassbox-interpretability/scripts/health_check.py
```

Or with curl:
```bash
curl http://localhost:8000/api/health
# Expected: {"status": "ok", "model_loaded": true}
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just the engine (no server needed)
pytest tests/ -k "engine" -v
```

---

## Docker

```bash
# Build
docker build -t glassbox .

# Run (CPU)
docker run -p 8000:8000 glassbox

# Run (GPU)
docker run --gpus all -p 8000:8000 glassbox
```

---

## Stopping the Server

```bash
# If running in foreground: Ctrl+C

# If running in background:
lsof -ti:8000 | xargs kill -9
```
