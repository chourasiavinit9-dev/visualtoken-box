"""Download SmolLM2-135M model weights from HuggingFace."""
from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
LOCAL_DIR = Path(__file__).parent / "models" / "SmolLM2-135M"

def main():
    print(f"🔮 Downloading {MODEL_ID}...")
    print(f"   Target: {LOCAL_DIR}")
    print(f"   Size: ~270MB (one-time download)")
    print()
    
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        # Only download what we need (skip .bin if .safetensors exists)
        ignore_patterns=["*.bin", "*.h5", "*.ot"],
    )
    
    print()
    print("✅ Model downloaded successfully!")
    print(f"   Config: {LOCAL_DIR / 'config.json'}")
    print(f"   Weights: {LOCAL_DIR / 'model.safetensors'}")

if __name__ == "__main__":
    main()