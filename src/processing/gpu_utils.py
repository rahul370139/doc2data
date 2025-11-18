"""Utility helpers for optional OpenCV CUDA acceleration."""
from __future__ import annotations
from typing import Tuple
from utils.config import Config

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover - OpenCV not installed in some environments
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False


class GPUUtils:
    """Small helper for OpenCV CUDA ops (falls back to CPU when unavailable)."""

    _cuda_available: bool | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Return True if CUDA path can be used (env + OpenCV support)."""
        if cls._cuda_available is None:
            cls._cuda_available = False
            if not Config.USE_GPU or not _CV2_AVAILABLE:
                return cls._cuda_available
            try:
                if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    cls._cuda_available = True
            except Exception:
                cls._cuda_available = False
        return cls._cuda_available

    @classmethod
    def clahe(cls, gray, clip_limit=2.0, tile_grid_size=(8, 8)):
        if not cls.is_available():
            raise RuntimeError("CUDA CLAHE unavailable")
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(gray)
        clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        gpu_res = clahe.apply(gpu_mat)
        return gpu_res.download()

    @classmethod
    def gaussian_blur(cls, gray, ksize=(3, 3), sigma=0):
        if not cls.is_available():
            raise RuntimeError("CUDA Gaussian unavailable")
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(gray)
        filter_ = cv2.cuda.createGaussianFilter(gpu_src.type(), gpu_src.type(), ksize, sigma)
        gpu_dst = filter_.apply(gpu_src)
        return gpu_dst.download()

    @classmethod
    def bilateral_filter(cls, image, diameter=5, sigma_color=50, sigma_space=50):
        if not cls.is_available():
            raise RuntimeError("CUDA bilateral unavailable")
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(image)
        gpu_dst = cv2.cuda.bilateralFilter(gpu_src, diameter, sigma_color, sigma_space)
        return gpu_dst.download()

    @classmethod
    def morphology(cls, image, op, kernel, iterations=1):
        if not cls.is_available():
            raise RuntimeError("CUDA morphology unavailable")
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(image)
        morph = cv2.cuda.createMorphologyFilter(op, gpu_src.type(), kernel)
        gpu_dst = gpu_src
        for _ in range(max(1, iterations)):
            gpu_dst = morph.apply(gpu_dst)
        return gpu_dst.download()

    @classmethod
    def threshold(cls, gray, thresh_type=cv2.THRESH_BINARY | cv2.THRESH_OTSU):
        if not cls.is_available():
            raise RuntimeError("CUDA threshold unavailable")
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(gray)
        _, gpu_dst = cv2.cuda.threshold(gpu_src, 0, 255, thresh_type)
        return gpu_dst.download()

