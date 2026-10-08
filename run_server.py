"""
GlassBox Server Runner — Launch FastAPI backend and web visualizer.
Made with 🔮 by Vivi
"""
import sys
from pathlib import Path
import uvicorn

# Ensure the root directory is on the Python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    print("=" * 60)
    print("  🔮 Visual Box Instrument Panel Visualizer")
    print("  Serving at: http://localhost:8000")
    print("  Made with 🔮 by Vivi")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "glassbox.server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Single process mode prevents Windows spawn path bugs
    )