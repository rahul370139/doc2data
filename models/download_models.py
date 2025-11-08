"""
Auto-download scripts for models (LayoutParser, Detectron2, PaddleOCR).
"""
import os
from pathlib import Path
from utils.config import Config


def download_layout_models():
    """Download LayoutParser/Detectron2 PubLayNet models."""
    print("Downloading LayoutParser PubLayNet model...")
    try:
        import layoutparser as lp
        
        # This will auto-download on first use
        model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
        )
        print("✓ LayoutParser model ready")
        return True
    except Exception as e:
        print(f"✗ Error downloading LayoutParser model: {e}")
        return False


def download_paddleocr_models():
    """Download PaddleOCR models."""
    print("Downloading PaddleOCR models...")
    try:
        from paddleocr import PaddleOCR
        
        # This will auto-download on first use
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("✓ PaddleOCR models ready")
        return True
    except Exception as e:
        print(f"✗ Error downloading PaddleOCR models: {e}")
        return False


def check_ollama_models():
    """Check if Ollama models are available."""
    print("Checking Ollama models...")
    try:
        from src.vlm.ollama_client import get_ollama_client
        
        client = get_ollama_client()
        
        # Check SLM model
        slm_model = Config.OLLAMA_MODEL_SLM
        if client.is_model_available(slm_model):
            print(f"✓ {slm_model} is available")
        else:
            print(f"⚠ {slm_model} not found. Pull with: ollama pull {slm_model}")
        
        # Check VLM model
        vlm_model = Config.OLLAMA_MODEL_VLM
        if client.is_model_available(vlm_model):
            print(f"✓ {vlm_model} is available")
        else:
            print(f"⚠ {vlm_model} not found. Pull with: ollama pull {vlm_model}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking Ollama models: {e}")
        return False


def download_all_models():
    """Download all required models."""
    print("=" * 50)
    print("Downloading all models...")
    print("=" * 50)
    
    results = []
    
    results.append(("LayoutParser", download_layout_models()))
    results.append(("PaddleOCR", download_paddleocr_models()))
    results.append(("Ollama", check_ollama_models()))
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    return all([r[1] for r in results])


if __name__ == "__main__":
    download_all_models()

