"""
SHA256-based artifact caching system.
"""
import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Optional
from utils.config import Config


def compute_hash(data: Any) -> str:
    """
    Compute SHA256 hash of data.
    
    Args:
        data: Data to hash (can be string, bytes, or any serializable object)
        
    Returns:
        SHA256 hash as hex string
    """
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        # Serialize object to JSON string
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_bytes = data_str.encode('utf-8')
    
    return hashlib.sha256(data_bytes).hexdigest()


def get_cache_path(cache_key: str, suffix: str = ".pkl") -> Path:
    """
    Get cache file path for a given key.
    
    Args:
        cache_key: Cache key (typically SHA256 hash)
        suffix: File suffix (default: .pkl)
        
    Returns:
        Path to cache file
    """
    cache_dir = Config.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}{suffix}"


def save_to_cache(data: Any, cache_key: str) -> Path:
    """
    Save data to cache.
    
    Args:
        data: Data to cache
        cache_key: Cache key (typically SHA256 hash)
        
    Returns:
        Path to saved cache file
    """
    if not Config.CACHE_ENABLED:
        return None
    
    cache_path = get_cache_path(cache_key)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return cache_path
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")
        return None


def load_from_cache(cache_key: str) -> Optional[Any]:
    """
    Load data from cache.
    
    Args:
        cache_key: Cache key (typically SHA256 hash)
        
    Returns:
        Cached data or None if not found
    """
    if not Config.CACHE_ENABLED:
        return None
    
    cache_path = get_cache_path(cache_key)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load from cache: {e}")
        return None


def cache_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of file.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hash as hex string
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def clear_cache():
    """Clear all cached files."""
    cache_dir = Config.CACHE_DIR
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.pkl"):
            cache_file.unlink()


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    cache_dir = Config.CACHE_DIR
    if not cache_dir.exists():
        return {
            "enabled": Config.CACHE_ENABLED,
            "total_files": 0,
            "total_size": 0
        }
    
    cache_files = list(cache_dir.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    return {
        "enabled": Config.CACHE_ENABLED,
        "total_files": len(cache_files),
        "total_size": total_size,
        "total_size_mb": total_size / (1024 * 1024)
    }

