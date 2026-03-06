#!/usr/bin/env python3
"""
Pre-download Florence-2-large model on DGX.

Run this ONCE on the DGX (outside Docker) to cache the model:
    python3 scripts/download_florence2.py

The model will be saved to ~/.cache/huggingface and mounted into Docker container.
"""
import os

# Must be online to download — clear any offline flags
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.setdefault("USE_TF", "0")

import torch


def download_florence2():
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "microsoft/Florence-2-large"
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    print(f"Downloading Florence-2-large to {cache_dir}...")
    print(f"Model ID: {model_id}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Download processor
    print("\n1. Downloading processor...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("   Processor downloaded successfully")
    
    # Download model
    # attn_implementation="eager" bypasses _supports_sdpa check (Florence-2 custom code lacks it)
    print("\n2. Downloading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager"  # Fixes: 'Florence2ForConditionalGeneration' has no attribute '_supports_sdpa'
    )
    print("   Model downloaded successfully")
    
    # Quick test
    print("\n3. Testing model load...")
    if torch.cuda.is_available():
        model = model.eval().cuda()
        print(f"   Model loaded on CUDA: {next(model.parameters()).device}")
    else:
        model = model.eval()
        print("   Model loaded on CPU")
    
    print("\n✅ Florence-2-large cached successfully!")
    print(f"   Cache location: {cache_dir}")
    print("\n   The model will be available in Docker via mounted volume.")

if __name__ == "__main__":
    download_florence2()
