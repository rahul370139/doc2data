"""
Image Registration Module for Template Loading.

PURPOSE: Loads reference templates (CMS-1500, UB-04) and computes ORB features
for alignment. For CMS-1500, delegates to cms1500_register for deterministic
template loading. Caches template data to avoid re-computation.

USE CASE: Called by template_alignment agent and multi_agent_pipeline when
alignment or zone matching needs template keypoints/descriptors.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import os

from src.pipelines.registration import get_cms1500_registrar

# Cache for template features to avoid re-computing
_TEMPLATE_CACHE: Dict[str, Dict[str, Any]] = {}


def get_reference_image_path(template_name: str) -> Optional[str]:
    """Resolve path to reference image for a template. Use when loading template for alignment."""
    name = (template_name or "").lower().strip()
    project_root = Path(__file__).parent.parent.parent

    if name in {"cms-1500", "cms1500"}:
        # Prefer user-provided canonical template in data/raw.
        candidates = [
            project_root / "data" / "raw" / "cms1500_template.pdf",
            project_root / "data" / "raw" / "cms1500_template.png",
            project_root / "data" / "raw" / "cms1500_template.jpg",
            project_root / "data" / "sample_docs" / "cms1500_blank.pdf",
            project_root / "data" / "sample_docs" / "cms1500_blank.png",
        ]
        env_path = os.getenv("CMS1500_TEMPLATE_PATH")
        if env_path:
            candidates.insert(0, Path(env_path))
        for p in candidates:
            try:
                if p.exists():
                    return str(p)
            except Exception:
                continue
        return None

    # Generic template resolution fallback.
    base_dir = project_root / "data" / "sample_docs"
    mapping = {
        "ub-04": "ub04_clean.pdf",
        "ub04": "ub04_clean.pdf",
    }
    filename = mapping.get(name)
    if not filename:
        return None
    path = base_dir / filename
    return str(path) if path.exists() else None

def load_and_process_reference(template_name: str) -> Optional[Dict[str, Any]]:
    """
    Load reference template and compute ORB features. Use for alignment and zone matching.
    Returns dict with keypoints, descriptors, shape. Cached for reuse.
    """
    global _TEMPLATE_CACHE
    
    if template_name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[template_name]

    # CMS-1500 uses the deterministic registrar template loader.
    if (template_name or "").lower().strip() in {"cms-1500", "cms1500"}:
        try:
            registrar = get_cms1500_registrar()
            data = registrar.get_template_data()
            out = {
                "keypoints": data.get("keypoints"),
                "descriptors": data.get("descriptors"),
                "shape": data.get("shape"),
                "image": data.get("image"),
                "image_rgb": data.get("image_rgb"),
                "line_mask": data.get("line_mask"),
            }
            _TEMPLATE_CACHE[template_name] = out
            return out
        except Exception as e:
            print(f"Error loading CMS-1500 reference template: {e}")
            return None
    
    path = get_reference_image_path(template_name)
    if not path:
        return None
    
    try:
        # Handle PDF reference
        if path.lower().endswith('.pdf'):
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            page = doc[0]
            # Render at 300 DPI for good feature detection
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            doc.close()
        else:
            # Handle Image reference
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
        if img is None:
            return None
            
        # Initialize ORB detector
        # Increase features for better matching on dense forms
        orb = cv2.ORB_create(nfeatures=5000)
        kp, des = orb.detectAndCompute(img, None)
        
        if des is None:
            return None
            
        data = {
            "keypoints": kp,
            "descriptors": des,
            "shape": img.shape, # h, w
            "image": img # Keep image for visualization/debugging if needed
        }
        _TEMPLATE_CACHE[template_name] = data
        return data
        
    except Exception as e:
        print(f"Error loading reference template {template_name}: {e}")
        return None
