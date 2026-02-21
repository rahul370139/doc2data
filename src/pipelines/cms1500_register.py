"""
Deterministic CMS-1500 template registration (classical CV only).

This module aligns a scanned CMS-1500 page to a canonical template space using:
1) Color-aware line masking (dropout-red aware)
2) AKAZE/ORB feature matching with RANSAC homography
3) Geometric fallback using detected form boundary
4) Warp to canonical template dimensions

No model training is required for this alignment layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import json

import cv2
import numpy as np

from utils.config import Config


CMS1500_CANONICAL_SIZE = (2550, 3300)  # width, height at 300 DPI


@dataclass
class CMS1500RegistrationResult:
    success: bool
    homography_input_to_template: Optional[np.ndarray]
    aligned_image: Optional[np.ndarray]
    quality: float
    method: str
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _TemplateData:
    path: Path
    image_rgb: np.ndarray
    image_gray: np.ndarray
    line_mask: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    shape: Tuple[int, int]  # (h, w)


@dataclass
class RegistrationThresholds:
    """
    Tunable thresholds for CMS-1500 alignment.
    Values are tuned for noisy handwritten scans while preserving machine-filled stability.
    """

    # Dropout-red mask thresholds (HSV).
    red_s_min: float = 0.34
    red_v_min: float = 0.45
    red_hue_low_deg: int = 345
    red_hue_high_deg: int = 15
    red_ratio_switch: float = 0.0012  # when to trust red rails in line-mask fusion

    # Feature matching thresholds.
    match_ratio_default: float = 0.78
    match_ratio_handwritten: float = 0.84
    min_keypoints_default: int = 20
    min_keypoints_handwritten: int = 14
    min_matches_default: int = 12
    min_matches_handwritten: int = 9

    # RANSAC thresholds.
    ransac_reproj_default: float = 3.5
    ransac_reproj_handwritten: float = 5.5
    min_feature_quality: float = 0.33

    # Quad fallback gates.
    quad_min_score_default: float = 0.52
    quad_min_score_handwritten: float = 0.34
    quad_prefer_margin: float = 0.02

    # Translation refinement gates.
    refine_max_shift_px: int = 12
    refine_min_response: float = 0.04

    @staticmethod
    def _get_env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    @staticmethod
    def _get_env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except Exception:
            return default

    @classmethod
    def from_env(cls) -> "RegistrationThresholds":
        out = cls()
        out.red_s_min = cls._get_env_float("CMS1500_RED_S_MIN", out.red_s_min)
        out.red_v_min = cls._get_env_float("CMS1500_RED_V_MIN", out.red_v_min)
        out.red_hue_low_deg = cls._get_env_int("CMS1500_RED_HUE_LOW_DEG", out.red_hue_low_deg)
        out.red_hue_high_deg = cls._get_env_int("CMS1500_RED_HUE_HIGH_DEG", out.red_hue_high_deg)
        out.red_ratio_switch = cls._get_env_float("CMS1500_RED_RATIO_SWITCH", out.red_ratio_switch)
        out.match_ratio_default = cls._get_env_float("CMS1500_MATCH_RATIO_DEFAULT", out.match_ratio_default)
        out.match_ratio_handwritten = cls._get_env_float("CMS1500_MATCH_RATIO_HANDWRITTEN", out.match_ratio_handwritten)
        out.min_keypoints_default = cls._get_env_int("CMS1500_MIN_KEYPOINTS_DEFAULT", out.min_keypoints_default)
        out.min_keypoints_handwritten = cls._get_env_int("CMS1500_MIN_KEYPOINTS_HANDWRITTEN", out.min_keypoints_handwritten)
        out.min_matches_default = cls._get_env_int("CMS1500_MIN_MATCHES_DEFAULT", out.min_matches_default)
        out.min_matches_handwritten = cls._get_env_int("CMS1500_MIN_MATCHES_HANDWRITTEN", out.min_matches_handwritten)
        out.ransac_reproj_default = cls._get_env_float("CMS1500_RANSAC_REPROJ_DEFAULT", out.ransac_reproj_default)
        out.ransac_reproj_handwritten = cls._get_env_float("CMS1500_RANSAC_REPROJ_HANDWRITTEN", out.ransac_reproj_handwritten)
        out.min_feature_quality = cls._get_env_float("CMS1500_MIN_FEATURE_QUALITY", out.min_feature_quality)
        out.quad_min_score_default = cls._get_env_float("CMS1500_QUAD_MIN_SCORE_DEFAULT", out.quad_min_score_default)
        out.quad_min_score_handwritten = cls._get_env_float("CMS1500_QUAD_MIN_SCORE_HANDWRITTEN", out.quad_min_score_handwritten)
        out.quad_prefer_margin = cls._get_env_float("CMS1500_QUAD_PREFER_MARGIN", out.quad_prefer_margin)
        out.refine_max_shift_px = cls._get_env_int("CMS1500_REFINE_MAX_SHIFT_PX", out.refine_max_shift_px)
        out.refine_min_response = cls._get_env_float("CMS1500_REFINE_MIN_RESPONSE", out.refine_min_response)
        return out


def _resolve_template_path() -> Optional[Path]:
    """
    Resolve canonical CMS-1500 template from user/project paths.
    """
    env = os.getenv("CMS1500_TEMPLATE_PATH")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env))

    candidates.extend(
        [
            Config.PROJECT_ROOT / "data" / "raw" / "cms1500_template.pdf",
            Config.PROJECT_ROOT / "data" / "raw" / "cms1500_template.png",
            Config.PROJECT_ROOT / "data" / "raw" / "cms1500_template.jpg",
            Config.PROJECT_ROOT / "data" / "sample_docs" / "cms1500_blank.pdf",
            Config.PROJECT_ROOT / "data" / "sample_docs" / "cms1500_blank.png",
        ]
    )

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return None


def _render_path_to_rgb(path: Path, dpi: int = 300) -> np.ndarray:
    """
    Load image/PDF into RGB numpy array.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz

        doc = fitz.open(str(path))
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif pix.n == 3:
            # PyMuPDF pixmap samples are RGB for 3-channel output.
            rgb = arr
        else:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        doc.close()
        return rgb

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read template image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _sauvola_binarize(gray: np.ndarray, window_size: int = 31, k: float = 0.2, R: float = 0.5) -> np.ndarray:
    """
    Sauvola-like adaptive threshold implemented with OpenCV box filters.
    Returns binary text mask (foreground=255).
    """
    if window_size % 2 == 0:
        window_size += 1
    f = gray.astype(np.float32) / 255.0
    mean = cv2.boxFilter(f, ddepth=-1, ksize=(window_size, window_size), borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.boxFilter(f * f, ddepth=-1, ksize=(window_size, window_size), borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(sqmean - mean * mean, 0.0)
    std = np.sqrt(var)
    thresh = mean * (1.0 + k * ((std / max(R, 1e-6)) - 1.0))
    out = (f < thresh).astype(np.uint8) * 255
    return out


def _dropout_red_mask(
    image_rgb: np.ndarray,
    s_min: float = 0.34,
    v_min: float = 0.45,
    hue_low_deg: int = 345,
    hue_high_deg: int = 15,
) -> np.ndarray:
    """
    Build a red line mask in HSV:
      H in [345°,15°], S > 0.4, V > 0.6  (OpenCV hue scale adapted).
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # OpenCV hue: 0..179 maps to 0..360 degrees.
    # Defaults map 345..360 and 0..15 degrees.
    deg_to_cv = lambda d: int(round((d % 360) / 2.0))
    low2 = deg_to_cv(hue_low_deg)
    high1 = deg_to_cv(hue_high_deg)

    lower1 = np.array([0, int(np.clip(s_min, 0.0, 1.0) * 255), int(np.clip(v_min, 0.0, 1.0) * 255)], dtype=np.uint8)
    upper1 = np.array([high1, 255, 255], dtype=np.uint8)
    lower2 = np.array([low2, int(np.clip(s_min, 0.0, 1.0) * 255), int(np.clip(v_min, 0.0, 1.0) * 255)], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # Clean and connect line fragments.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return mask


def _structural_line_mask(gray: np.ndarray) -> np.ndarray:
    """
    Structural grid/line mask from grayscale for scans where red dropout is weak.
    """
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    h, w = bw.shape[:2]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 25)))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
    out = cv2.bitwise_or(horiz, vert)
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return out


def _build_masks(
    image_rgb: np.ndarray,
    thresholds: RegistrationThresholds,
    profile: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      line_mask: geometry/alignment mask
      text_mask: OCR-friendly text mask
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    red = _dropout_red_mask(
        image_rgb,
        s_min=thresholds.red_s_min,
        v_min=thresholds.red_v_min,
        hue_low_deg=thresholds.red_hue_low_deg,
        hue_high_deg=thresholds.red_hue_high_deg,
    )
    structural = _structural_line_mask(gray)

    red_ratio = float(np.count_nonzero(red)) / float(max(red.size, 1))
    dynamic_switch = thresholds.red_ratio_switch
    if profile and profile.get("handwritten_likely"):
        # Handwritten/fax scans often weaken red rails; lower gate slightly.
        dynamic_switch *= 0.75

    # Blend red + structural so we preserve red rails but remain robust on B/W scans.
    if red_ratio > dynamic_switch:
        line_mask = cv2.bitwise_or(red, structural)
    else:
        line_mask = structural

    text_mask = _sauvola_binarize(gray, window_size=31, k=0.2, R=0.5)
    return line_mask, text_mask


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_outer_quad(line_mask: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Detect outer form boundary from line mask.
    """
    h, w = line_mask.shape[:2]
    img_area = float(h * w)

    # 1) Hough-based coarse rectangle from dominant horizontal/vertical lines.
    hough_score = 0.0
    hough_quad = None
    lines = cv2.HoughLinesP(
        line_mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=120,
        minLineLength=max(120, min(h, w) // 5),
        maxLineGap=20,
    )
    if lines is not None and len(lines) >= 8:
        xs: List[float] = []
        ys: List[float] = []
        h_count = 0
        v_count = 0
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, l.tolist())
            dx = x2 - x1
            dy = y2 - y1
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            if ang < 15 or ang > 165:
                ys.extend([(y1 + y2) * 0.5])
                h_count += 1
            elif 75 < ang < 105:
                xs.extend([(x1 + x2) * 0.5])
                v_count += 1
        if len(xs) >= 2 and len(ys) >= 2:
            left = float(np.percentile(xs, 5))
            right = float(np.percentile(xs, 95))
            top = float(np.percentile(ys, 5))
            bottom = float(np.percentile(ys, 95))
            if right - left > w * 0.45 and bottom - top > h * 0.45:
                hough_quad = np.array(
                    [[left, top], [right, top], [right, bottom], [left, bottom]],
                    dtype=np.float32,
                )
                line_support = min(1.0, (h_count + v_count) / 80.0)
                area_ratio = ((right - left) * (bottom - top)) / max(img_area, 1.0)
                hough_score = 0.6 * min(1.0, area_ratio / 0.75) + 0.4 * line_support

    # 2) Contour fallback.
    contour_quad = None
    contour_score = 0.0
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < img_area * 0.12:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            rect = cv2.minAreaRect(cnt)
            approx = cv2.boxPoints(rect).reshape(4, 1, 2)
        quad = _order_quad_points(approx.reshape(4, 2))
        area_ratio = area / max(img_area, 1.0)
        score = min(1.0, area_ratio / 0.75)
        if score > contour_score:
            contour_score = score
            contour_quad = quad

    if hough_quad is not None and hough_score >= contour_score:
        return hough_quad, float(hough_score)
    if contour_quad is not None:
        return contour_quad, float(contour_score)
    return None, 0.0


def _estimate_scan_profile(gray: np.ndarray, structural_line_mask: np.ndarray) -> Dict[str, Any]:
    """
    Estimate if scan resembles handwritten/noisy fax conditions.
    """
    h, w = gray.shape[:2]
    area = float(max(h * w, 1))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = float(np.clip(1.0 - min(lap_var, 280.0) / 280.0, 0.0, 1.0))
    line_density = float(np.count_nonzero(structural_line_mask)) / area

    # Dark-pixel stroke estimate from adaptive threshold.
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    stroke_density = float(np.count_nonzero(bw)) / area

    handwritten_likely = bool((stroke_density > 0.03 and line_density < 0.07) or blur_score > 0.55)
    return {
        "lap_var": lap_var,
        "blur_score": blur_score,
        "line_density": line_density,
        "stroke_density": stroke_density,
        "handwritten_likely": handwritten_likely,
    }


def _detect_features(
    gray: np.ndarray,
    line_mask: np.ndarray,
    min_akaze_kp: int = 60,
    min_orb_kp: int = 40,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray], str]:
    """
    Detect features focusing near grid lines first, then full image fallback.
    """
    focus = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)

    # AKAZE first (robust binary descriptors).
    try:
        akaze = cv2.AKAZE_create()
        kp, des = akaze.detectAndCompute(gray, focus)
        if des is not None and len(kp) >= min_akaze_kp:
            return kp, des, "akaze"
    except Exception:
        pass

    # ORB fallback.
    orb = cv2.ORB_create(nfeatures=9000, fastThreshold=7)
    kp, des = orb.detectAndCompute(gray, focus)
    if des is not None and len(kp) >= min_orb_kp:
        return kp, des, "orb"
    kp, des = orb.detectAndCompute(gray, None)
    return kp or [], des, "orb_full"


def _match_descriptors(
    des_src: np.ndarray,
    des_dst: np.ndarray,
    ratio: float = 0.78,
) -> List[cv2.DMatch]:
    if des_src is None or des_dst is None:
        return []
    norm = cv2.NORM_HAMMING if des_src.dtype == np.uint8 else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(des_src, des_dst, k=2)
    good: List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _homography_quality(
    H: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    inlier_mask: np.ndarray,
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
) -> float:
    """
    Score homography quality 0..1 using inliers + reprojection + corner sanity.
    """
    if H is None or inlier_mask is None or src_pts.shape[0] == 0:
        return 0.0

    inliers = inlier_mask.ravel().astype(bool)
    if not np.any(inliers):
        return 0.0

    inlier_ratio = float(np.count_nonzero(inliers)) / float(max(len(inliers), 1))
    src_in = src_pts[inliers].reshape(-1, 1, 2).astype(np.float32)
    dst_in = dst_pts[inliers].reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(src_in, H)
    err = np.linalg.norm(proj.reshape(-1, 2) - dst_in.reshape(-1, 2), axis=1)
    reproj = float(np.mean(err)) if err.size else 99.0
    reproj_score = max(0.0, 1.0 - min(reproj, 12.0) / 12.0)

    sh, sw = src_shape[:2]
    dh, dw = dst_shape[:2]
    corners = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not np.isfinite(warped).all():
        return 0.0

    area = abs(float(cv2.contourArea(warped.astype(np.float32))))
    dst_area = float(dw * dh)
    area_ratio = area / max(dst_area, 1.0)
    area_score = 1.0 - min(abs(area_ratio - 1.0), 0.7) / 0.7

    pad_x = dw * 0.2
    pad_y = dh * 0.2
    inside = bool(
        warped[:, 0].min() >= -pad_x
        and warped[:, 1].min() >= -pad_y
        and warped[:, 0].max() <= dw + pad_x
        and warped[:, 1].max() <= dh + pad_y
    )
    inside_score = 1.0 if inside else 0.0

    return float(0.50 * inlier_ratio + 0.25 * reproj_score + 0.15 * area_score + 0.10 * inside_score)


class CMS1500Registrar:
    """
    Classical CV registrar for CMS-1500.
    """

    def __init__(self, template_path: Optional[Path] = None, template_dpi: int = 300):
        self._template_path_override = template_path
        self._template_dpi = template_dpi
        self._template: Optional[_TemplateData] = None
        self.thresholds = RegistrationThresholds.from_env()

    def _load_template(self) -> _TemplateData:
        if self._template is not None:
            return self._template

        path = self._template_path_override or _resolve_template_path()
        if path is None:
            raise RuntimeError("CMS-1500 template not found. Put cms1500_template.pdf in data/raw or set CMS1500_TEMPLATE_PATH.")

        rgb = _render_path_to_rgb(path, dpi=self._template_dpi)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        line_mask, _ = _build_masks(rgb, self.thresholds, profile={"handwritten_likely": False})
        kp, des, detector = _detect_features(
            gray,
            line_mask,
            min_akaze_kp=max(30, self.thresholds.min_keypoints_default),
            min_orb_kp=max(22, int(self.thresholds.min_keypoints_default * 0.7)),
        )
        self._template = _TemplateData(
            path=path,
            image_rgb=rgb,
            image_gray=gray,
            line_mask=line_mask,
            keypoints=kp,
            descriptors=des,
            shape=gray.shape[:2],
        )
        return self._template

    def get_template_data(self) -> Dict[str, Any]:
        t = self._load_template()
        return {
            "path": str(t.path),
            "image_rgb": t.image_rgb,
            "image": t.image_gray,
            "shape": t.shape,
            "line_mask": t.line_mask,
            "keypoints": t.keypoints,
            "descriptors": t.descriptors,
        }

    def register(self, input_image: np.ndarray) -> CMS1500RegistrationResult:
        template = self._load_template()
        if input_image is None or not hasattr(input_image, "shape"):
            return CMS1500RegistrationResult(False, None, None, 0.0, "invalid_input")

        scan_rgb = input_image
        if input_image.ndim == 2:
            scan_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        scan_gray = cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2GRAY)
        structural_scan = _structural_line_mask(scan_gray)
        profile = _estimate_scan_profile(scan_gray, structural_scan)
        scan_line_mask, scan_text_mask = _build_masks(scan_rgb, self.thresholds, profile=profile)

        debug: Dict[str, Any] = {
            "template_path": str(template.path),
            "scan_shape": tuple(scan_gray.shape[:2]),
            "template_shape": tuple(template.shape),
            "profile": profile,
        }
        debug["thresholds"] = {
            "red_s_min": self.thresholds.red_s_min,
            "red_v_min": self.thresholds.red_v_min,
            "red_ratio_switch": self.thresholds.red_ratio_switch,
            "match_ratio_default": self.thresholds.match_ratio_default,
            "match_ratio_handwritten": self.thresholds.match_ratio_handwritten,
            "ransac_reproj_default": self.thresholds.ransac_reproj_default,
            "ransac_reproj_handwritten": self.thresholds.ransac_reproj_handwritten,
            "quad_min_score_default": self.thresholds.quad_min_score_default,
            "quad_min_score_handwritten": self.thresholds.quad_min_score_handwritten,
        }

        handwritten_like = bool(profile.get("handwritten_likely", False))
        match_ratio = self.thresholds.match_ratio_handwritten if handwritten_like else self.thresholds.match_ratio_default
        min_keypoints = self.thresholds.min_keypoints_handwritten if handwritten_like else self.thresholds.min_keypoints_default
        min_matches = self.thresholds.min_matches_handwritten if handwritten_like else self.thresholds.min_matches_default
        ransac_trials = [self.thresholds.ransac_reproj_default]
        if handwritten_like:
            ransac_trials.append(self.thresholds.ransac_reproj_handwritten)
        quad_min_score = self.thresholds.quad_min_score_handwritten if handwritten_like else self.thresholds.quad_min_score_default

        # A) Feature-based homography.
        kp_scan, des_scan, det_name = _detect_features(
            scan_gray,
            scan_line_mask,
            min_akaze_kp=max(14, min_keypoints),
            min_orb_kp=max(12, int(min_keypoints * 0.7)),
        )
        debug["feature_detector"] = det_name
        debug["scan_keypoints"] = len(kp_scan)
        debug["template_keypoints"] = len(template.keypoints)

        best_H = None
        best_quality = 0.0
        best_method = "none"

        if des_scan is not None and template.descriptors is not None and len(kp_scan) >= min_keypoints and len(template.keypoints) >= min_keypoints:
            matches = _match_descriptors(des_scan, template.descriptors, ratio=match_ratio)
            debug["good_matches"] = len(matches)
            if len(matches) >= min_matches:
                src_pts = np.float32([kp_scan[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                dst_pts = np.float32([template.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                feature_candidates: List[Tuple[np.ndarray, float, float]] = []
                for reproj in ransac_trials:
                    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(reproj))
                    if H is None or inliers is None:
                        continue
                    q = _homography_quality(H, src_pts, dst_pts, inliers, scan_gray.shape, template.image_gray.shape)
                    feature_candidates.append((H, q, float(reproj)))
                if feature_candidates:
                    Hf, qf, reproj_used = max(feature_candidates, key=lambda t: t[1])
                    debug["feature_quality"] = float(qf)
                    debug["feature_ransac_reproj_used"] = float(reproj_used)
                    if qf >= self.thresholds.min_feature_quality:
                        best_H = Hf
                        best_quality = qf
                        best_method = "feature_ransac"

        # B) Geometric fallback from outer boundary.
        quad, quad_score = _detect_outer_quad(scan_line_mask)
        debug["quad_score"] = quad_score
        if quad is not None:
            th, tw = template.shape[:2]
            dst_quad = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
            H_quad = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_quad)
            q_quad = float(min(max(quad_score, 0.0), 1.0))
            if H_quad is not None and q_quad >= quad_min_score:
                # Prefer quad when feature result is missing/weak or quad is clearly better.
                if (
                    best_H is None
                    or best_quality < self.thresholds.min_feature_quality
                    or q_quad > best_quality + self.thresholds.quad_prefer_margin
                ):
                    best_H = H_quad
                    best_quality = q_quad
                    best_method = "outer_quad"

        if best_H is None:
            return CMS1500RegistrationResult(
                success=False,
                homography_input_to_template=None,
                aligned_image=None,
                quality=0.0,
                method="failed",
                debug=debug,
            )

        th, tw = template.shape[:2]
        aligned = cv2.warpPerspective(scan_rgb, best_H, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Optional micro-refinement: tiny translation via phase correlation on line masks.
        try:
            warped_line = cv2.warpPerspective(scan_line_mask, best_H, (tw, th), flags=cv2.INTER_LINEAR)
            tline = template.line_mask
            shift, response = cv2.phaseCorrelate(np.float32(tline), np.float32(warped_line))
            sx, sy = float(shift[0]), float(shift[1])
            if (
                abs(sx) <= float(self.thresholds.refine_max_shift_px)
                and abs(sy) <= float(self.thresholds.refine_max_shift_px)
                and response > float(self.thresholds.refine_min_response)
            ):
                M = np.array([[1.0, 0.0, -sx], [0.0, 1.0, -sy], [0.0, 0.0, 1.0]], dtype=np.float32)
                best_H = M @ best_H
                aligned = cv2.warpPerspective(scan_rgb, best_H, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                best_quality = min(1.0, best_quality + 0.03)
                debug["translation_refine"] = {"sx": sx, "sy": sy, "response": float(response)}
        except Exception:
            pass

        debug["final_quality"] = best_quality
        debug["method"] = best_method
        debug["text_mask_density"] = float(np.count_nonzero(scan_text_mask)) / float(max(scan_text_mask.size, 1))

        return CMS1500RegistrationResult(
            success=True,
            homography_input_to_template=best_H,
            aligned_image=aligned,
            quality=float(best_quality),
            method=best_method,
            debug=debug,
        )

    @staticmethod
    def transform_bbox(
        bbox: Tuple[float, float, float, float],
        H: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Transform bbox by homography matrix H.
        """
        x0, y0, x1, y1 = bbox
        pts = np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]])
        warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return (
            float(np.min(warped[:, 0])),
            float(np.min(warped[:, 1])),
            float(np.max(warped[:, 0])),
            float(np.max(warped[:, 1])),
        )

    @staticmethod
    def snap_bbox_to_gridlines(
        bbox: Tuple[float, float, float, float],
        line_mask: np.ndarray,
        max_adjust_px: int = 8,
    ) -> Tuple[float, float, float, float]:
        """
        Snap bbox edges to nearest strong line within a small tolerance.
        """
        h, w = line_mask.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        if x1 <= x0 or y1 <= y0:
            return bbox

        def _snap_vertical(x: int) -> int:
            lo = max(0, x - max_adjust_px)
            hi = min(w - 1, x + max_adjust_px)
            col_scores = np.sum(line_mask[:, lo : hi + 1] > 0, axis=0)
            idx = int(np.argmax(col_scores))
            best = lo + idx
            if col_scores[idx] <= 0:
                return x
            return best

        def _snap_horizontal(y: int) -> int:
            lo = max(0, y - max_adjust_px)
            hi = min(h - 1, y + max_adjust_px)
            row_scores = np.sum(line_mask[lo : hi + 1, :] > 0, axis=1)
            idx = int(np.argmax(row_scores))
            best = lo + idx
            if row_scores[idx] <= 0:
                return y
            return best

        sx0 = _snap_vertical(x0)
        sx1 = _snap_vertical(x1)
        sy0 = _snap_horizontal(y0)
        sy1 = _snap_horizontal(y1)
        if sx1 <= sx0:
            sx0, sx1 = x0, x1
        if sy1 <= sy0:
            sy0, sy1 = y0, y1
        return (float(sx0), float(sy0), float(sx1), float(sy1))


@lru_cache(maxsize=1)
def get_cms1500_registrar() -> CMS1500Registrar:
    return CMS1500Registrar()


def load_canonical_cms1500_boxes(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load canonical template-space field boxes.
    Supports delegation through `source_schema`.
    """
    p = path or (Config.PROJECT_ROOT / "data" / "templates" / "cms1500_boxes.json")
    payload: Dict[str, Any] = {"fields": []}
    try:
        if p.exists():
            payload = json.loads(p.read_text())
    except Exception:
        payload = {"fields": []}

    fields = payload.get("fields", [])
    if fields:
        return payload

    source_schema = payload.get("source_schema")
    if source_schema:
        sp = Path(source_schema)
        if not sp.is_absolute():
            sp = Config.PROJECT_ROOT / sp
        try:
            if sp.exists():
                source = json.loads(sp.read_text())
                payload["fields"] = source.get("fields", [])
                return payload
        except Exception:
            pass

    fallback = Config.PROJECT_ROOT / "data" / "schemas" / "cms-1500.json"
    try:
        if fallback.exists():
            source = json.loads(fallback.read_text())
            payload["fields"] = source.get("fields", [])
    except Exception:
        pass
    return payload


def project_canonical_boxes_to_input(
    fields: List[Dict[str, Any]],
    H_template_to_input: np.ndarray,
    template_size: Tuple[int, int],
    snap_line_mask: Optional[np.ndarray] = None,
    max_snap_px: int = 8,
) -> List[Dict[str, Any]]:
    """
    Project normalized template-space boxes back to original input coordinates.
    """
    registrar = get_cms1500_registrar()
    tw, th = template_size
    out: List[Dict[str, Any]] = []
    for field in fields:
        norm = field.get("bbox_norm") or field.get("bbox")
        if not norm or len(norm) < 4:
            continue
        x0 = float(norm[0]) * float(tw)
        y0 = float(norm[1]) * float(th)
        x1 = float(norm[2]) * float(tw)
        y1 = float(norm[3]) * float(th)
        bbox = registrar.transform_bbox((x0, y0, x1, y1), H_template_to_input)
        if snap_line_mask is not None:
            bbox = registrar.snap_bbox_to_gridlines(bbox, snap_line_mask, max_adjust_px=max_snap_px)
        item = dict(field)
        item["bbox_input"] = [float(v) for v in bbox]
        out.append(item)
    return out


def detect_checkbox_state(crop: np.ndarray) -> Tuple[bool, float]:
    """
    Classical checkbox state detector using contour geometry + fill ratio.
    """
    if crop is None or crop.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    best_checked = False
    area_total = float(gray.shape[0] * gray.shape[1])
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        area = float(w * h)
        if area < area_total * 0.02 or area > area_total * 0.7:
            continue
        ratio = w / float(max(h, 1))
        if ratio < 0.65 or ratio > 1.35:
            continue
        roi = bw[y : y + h, x : x + w]
        fill_ratio = float(np.count_nonzero(roi)) / float(max(roi.size, 1))
        checked = fill_ratio > 0.18
        score = min(1.0, abs(fill_ratio - 0.18) * 3.0 + 0.35)
        if score > best_score:
            best_score = score
            best_checked = checked
    return best_checked, best_score


def link_labels_to_values(
    label_boxes: List[Tuple[float, float, float, float]],
    value_boxes: List[Tuple[float, float, float, float]],
) -> List[Tuple[int, int]]:
    """
    Hungarian-style label/value pairing constrained by geometric distance.
    Falls back to greedy matching if scipy is unavailable.
    """
    if not label_boxes or not value_boxes:
        return []

    def _center(b):
        return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

    L = len(label_boxes)
    V = len(value_boxes)
    cost = np.full((L, V), 1e6, dtype=np.float32)
    for i, lb in enumerate(label_boxes):
        lx, ly = _center(lb)
        for j, vb in enumerate(value_boxes):
            vx, vy = _center(vb)
            if vx < lx - 5:
                continue
            dx = vx - lx
            dy = abs(vy - ly)
            cost[i, j] = dx + 2.0 * dy

    pairs: List[Tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        ridx, cidx = linear_sum_assignment(cost)
        for r, c in zip(ridx.tolist(), cidx.tolist()):
            if cost[r, c] < 1e5:
                pairs.append((int(r), int(c)))
        return pairs
    except Exception:
        used_vals = set()
        for i in range(L):
            j_best = int(np.argmin(cost[i]))
            if cost[i, j_best] >= 1e5 or j_best in used_vals:
                continue
            pairs.append((i, j_best))
            used_vals.add(j_best)
        return pairs

