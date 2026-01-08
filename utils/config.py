"""
Configuration management for document processing pipeline.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for application settings."""
    
    # Ollama Configuration
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost:11434")
    OLLAMA_MODEL_SLM: str = os.getenv("OLLAMA_MODEL_SLM", "llama3.2:3b")  # Fast, good for structured extraction
    OLLAMA_MODEL_VLM: str = os.getenv("OLLAMA_MODEL_VLM", "llama3.2:3b")  # Fallback to SLM if no VLM
    ENABLE_SLM: bool = os.getenv("ENABLE_SLM", "true").lower() == "true"  # Enabled by default
    ENABLE_VLM: bool = os.getenv("ENABLE_VLM", "true").lower() == "true"  # Enabled by default
    
    # Model Paths
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "models/weights"))
    LAYOUT_MODEL: str = os.getenv("LAYOUT_MODEL", "publaynet")
    OCR_MODEL: str = os.getenv("OCR_MODEL", "paddleocr")
    # YOLO model paths - check container path first, then local
    @staticmethod
    def _get_yolo_path() -> Optional[str]:
        """Find YOLO model in container or local paths."""
        # Environment override takes priority
        env_path = os.getenv("YOLO_MODEL_PATH")
        if env_path and Path(env_path).exists():
            return env_path
        
        # Container path (Docker)
        container_path = Path("/app/models/yolo/cms1500_best.pt")
        if container_path.exists():
            return str(container_path)
        
        # Local development paths
        local_paths = [
            "runs/detect/cms1500_reducto_v1/weights/best.pt",
            "models/yolo/cms1500_reducto_v1.pt",
            "runs/detect/cms1500_yolo_run13/weights/best.pt",
            "runs/detect/cms1500_yolo_run1/weights/best.pt",
            "models/yolo/cms1500_best.pt",
        ]
        for p in local_paths:
            if Path(p).exists():
                return p
        
        return None
    
    YOLO_MODEL_PATH: Optional[str] = None  # Set dynamically
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
    YOLO_IOU: float = float(os.getenv("YOLO_IOU", "0.6"))
    
    # Processing Configuration
    DPI: int = int(os.getenv("DPI", "300"))
    DESKEW_ENABLED: bool = os.getenv("DESKEW_ENABLED", "true").lower() == "true"
    DENOISE_ENABLED: bool = os.getenv("DENOISE_ENABLED", "true").lower() == "true"
    
    # Cache Configuration
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "cache"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "logs"))
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # GPU Configuration (optional, for future use)
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    CUDA_VISIBLE_DEVICES: Optional[str] = os.getenv("CUDA_VISIBLE_DEVICES")
    # TATR model - use HuggingFace model name directly
    TATR_MODEL_PATH: Optional[str] = os.getenv("TATR_MODEL_PATH", "microsoft/table-transformer-structure-recognition")

    # Document type hint for adaptive preprocessing (generic|form|cms1500|ub04|ncpdp|handwritten)
    DOC_TYPE_HINT: str = os.getenv("DOC_TYPE_HINT", "generic").lower()
    
    # NVIDIA DGX Configuration (optional, for future use)
    DGX_HOST: Optional[str] = os.getenv("DGX_HOST")
    DGX_PORT: Optional[str] = os.getenv("DGX_PORT")
    
    # Project root directory
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        (cls.PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (cls.PROJECT_ROOT / "validation" / "results").mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_ollama_url(cls) -> str:
        """Get Ollama API URL."""
        return f"http://{cls.OLLAMA_HOST}"


# Initialize directories and dynamic config
Config.ensure_directories()
Config.YOLO_MODEL_PATH = Config._get_yolo_path()
print(f"[Config] YOLO model path: {Config.YOLO_MODEL_PATH}")
