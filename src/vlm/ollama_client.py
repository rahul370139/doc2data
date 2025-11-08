"""
Ollama client for SLM and VLM inference.
"""
import requests
import json
from typing import Dict, Any, Optional, List
from utils.config import Config


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API (default: from Config)
        """
        self.base_url = base_url or Config.get_ollama_url()
        self.session = requests.Session()
    
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        format: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama.
        
        Args:
            model: Model name
            prompt: User prompt
            system: System prompt (optional)
            images: List of base64-encoded images for VLM (optional)
            format: Response format (e.g., "json")
            stream: Whether to stream response
            
        Returns:
            Response dictionary
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if system:
            payload["system"] = system
        
        if images:
            payload["images"] = images
        
        if format:
            payload["format"] = format
        
        try:
            response = self.session.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            raise
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        format: Optional[str] = "json",
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Chat completion using Ollama.
        
        Args:
            model: Model name
            messages: List of message dicts with "role" and "content"
            format: Response format (e.g., "json")
            stream: Whether to stream response
            
        Returns:
            Response dictionary
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if format:
            payload["format"] = format
        
        try:
            response = self.session.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error calling Ollama chat API: {e}")
            raise
    
    def is_model_available(self, model: str) -> bool:
        """
        Check if model is available in Ollama.
        
        Args:
            model: Model name
            
        Returns:
            True if model is available
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(model in name for name in model_names)
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False
    
    def pull_model(self, model: str) -> bool:
        """
        Pull model from Ollama registry.
        
        Args:
            model: Model name
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/api/pull"
            payload = {"name": model}
            response = self.session.post(url, json=payload, stream=True, timeout=600)
            response.raise_for_status()
            
            # Stream response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("status") == "success":
                            return True
                    except:
                        continue
            
            return False
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False


# Global instance
_ollama_client = None


def get_ollama_client() -> OllamaClient:
    """Get global Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client

