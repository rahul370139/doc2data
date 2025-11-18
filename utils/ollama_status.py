"""Helpers to check Ollama server availability."""
from __future__ import annotations
import requests
from utils.config import Config


def is_ollama_available(timeout: float = 2.0) -> bool:
    """Return True if Ollama HTTP endpoint responds to /api/tags."""
    try:
        url = f"{Config.get_ollama_url().rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False
