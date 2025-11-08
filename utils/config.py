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
    
    # Ollama Configuration (stubbed for local testing)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost:11434")
    OLLAMA_MODEL_SLM: str = os.getenv("OLLAMA_MODEL_SLM", "qwen2.5:7b-instruct")
    OLLAMA_MODEL_VLM: str = os.getenv("OLLAMA_MODEL_VLM", "qwen2-vl:7b")
    ENABLE_SLM: bool = os.getenv("ENABLE_SLM", "false").lower() == "true"
    ENABLE_VLM: bool = os.getenv("ENABLE_VLM", "false").lower() == "true"
    
    # Model Paths
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "models/weights"))
    LAYOUT_MODEL: str = os.getenv("LAYOUT_MODEL", "publaynet")
    OCR_MODEL: str = os.getenv("OCR_MODEL", "paddleocr")
    
    # Processing Configuration
    DPI: int = int(os.getenv("DPI", "300"))
    DESKEW_ENABLED: bool = os.getenv("DESKEW_ENABLED", "true").lower() == "true"
    DENOISE_ENABLED: bool = os.getenv("DENOISE_ENABLED", "true").lower() == "true"
    
    # Cache Configuration
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
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


# Initialize directories
Config.ensure_directories()

