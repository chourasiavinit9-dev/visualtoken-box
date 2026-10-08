"""
Test script for running Google Gemma multimodal (image + text) with Transformers.
"""
import sys

def main():
    print("=" * 60)
    print("🔍 Checking Environment & Hardware Acceleration...")
    print("=" * 60)

    try:
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please ensure: pip install --default-timeout=1000 torch torchvision transformers accelerate pillow")
        sys.exit(1)

    print(f"✅ PyTorch Version:      {torch.__version__}")
    print(f"✅ Transformers Version: {transformers.__version__}")

    # Determine device: MPS (Apple Silicon GPU), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
        print("⚡ Hardware Acceleration: Apple Metal (MPS) GPU detected")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print("⚡ Hardware Acceleration: NVIDIA CUDA GPU detected")
    else:
        device = "cpu"
        dtype = torch.float32
        print("ℹ️ Hardware Acceleration: CPU mode")

    print()
    # Recommended default multimodal Gemma model
    # (4B parameters fits in Mac unified memory)
    model_id = "google/gemma-3-4b-it"

    print(f"📦 Loading Processor & Model: {model_id}...")
    print("   (First run will download model weights from Hugging Face)")

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device if device != "mps" else None,
            torch_dtype=dtype,
        )
        if device == "mps":
            model = model.to(device)
    except Exception as err:
        print(f"❌ Error loading model: {err}")
        print("\nNote: Make sure you accepted the terms on https://huggingface.co/google/gemma-3-4b-it")
        return

    # Multimodal test message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy and what color is it?"},
            ],
        },
    ]

    print("\n💬 Preparing input prompt and image...")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    print("🤖 Generating response...")
    outputs = model.generate(**inputs, max_new_tokens=128)

    response_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = processor.decode(response_tokens, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("✨ Model Response:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    main()
