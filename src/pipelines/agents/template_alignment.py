"""
Template Alignment Agent - aligns scanned forms to reference templates.

PURPOSE: Warps scanned form image to canonical template space so zone-based
OCR can match schema regions. For CMS-1500, delegates to cms1500_register.
For other forms, uses ORB/feature matching or optional YOLO boundary detection.

USE CASE: Required for Lane C (scanned forms). Corrects rotation, skew, and
scale before OCR. Called by MultiAgentPipeline when form is scanned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.pipelines.core import BaseAgent, AlignmentResult, FormType

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.config import Config


class TemplateAlignmentAgent(BaseAgent):
    """
    Aligns scanned forms to a reference template using feature matching.
    Implements Sensible-style alignment with fallback to ML detection.
    """
    
    def __init__(self):
        super().__init__("TemplateAlignmentAgent")
        self._templates: Dict[FormType, np.ndarray] = {}
    
    async def initialize(self):
        if self._initialized:
            return
        # Load reference templates
        template_dir = Path(__file__).parent.parent.parent.parent / "data" / "templates"
        if template_dir.exists():
            for template_file in template_dir.glob("*.png"):
                form_name = template_file.stem.replace("_template", "")
                try:
                    form_type = FormType(form_name)
                    self._templates[form_type] = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                except:
                    pass
        # Fallback: load reference images from sample_docs if template PNGs missing
        if FormType.CMS1500 not in self._templates:
            try:
                from src.processing.registration import load_and_process_reference
                ref_data = load_and_process_reference("cms-1500")
                if ref_data and ref_data.get("image") is not None:
                    self._templates[FormType.CMS1500] = ref_data["image"]
            except Exception:
                pass
        if FormType.UB04 not in self._templates:
            try:
                from src.processing.registration import load_and_process_reference
                ref_data = load_and_process_reference("ub-04")
                if ref_data and ref_data.get("image") is not None:
                    self._templates[FormType.UB04] = ref_data["image"]
            except Exception:
                    pass
        self._initialized = True
    
    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points as: top-left, top-right, bottom-right, bottom-left.
        pts: shape (4,2)
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _detect_form_quad(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect the outer form rectangle as a quadrilateral.
        Returns (quad_points, score) where score is 0..1.
        This is a stable alignment primitive that will not "tilt" a straight scan
        if the quad is detected correctly.
        """
        try:
            g = gray
            if g.dtype != np.uint8:
                g = g.astype(np.uint8)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            # Strong edges
            edges = cv2.Canny(g, 50, 150)
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, 0.0

            h, w = g.shape[:2]
            img_area = float(h * w)

            best = None
            best_score = 0.0
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < img_area * 0.15:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) != 4:
                    continue

                quad = approx.reshape(4, 2).astype(np.float32)
                quad = self._order_quad_points(quad)
                # Score: large area + rectangular-ish shape
                area_ratio = area / img_area

                # Angle score: corners should be near 90 deg
                def _angle(a, b, c) -> float:
                    ba = a - b
                    bc = c - b
                    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
                    return float(np.degrees(np.arccos(cosang)))

                angs = [
                    _angle(quad[3], quad[0], quad[1]),
                    _angle(quad[0], quad[1], quad[2]),
                    _angle(quad[1], quad[2], quad[3]),
                    _angle(quad[2], quad[3], quad[0]),
                ]
                ang_err = float(np.mean([abs(a - 90.0) for a in angs]))
                angle_score = float(max(0.0, 1.0 - (ang_err / 25.0)))  # 25deg avg error -> 0

                score = 0.7 * min(1.0, area_ratio / 0.60) + 0.3 * angle_score
                if score > best_score:
                    best_score = score
                    best = quad

            return best, float(best_score)
        except Exception:
            return None, 0.0
    
    def _try_yolo_alignment(self, image: np.ndarray, template_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], float]:
        """
        Try to align using the fine-tuned YOLO model for form boundary detection.
        The YOLO model can detect the CMS-1500 form region more reliably than contours
        on noisy/handwritten scans.
        Returns (homography_matrix, quality_score) or (None, 0.0).
        """
        try:
            from pathlib import Path
            from src.pipelines.yolo_layout import YOLOLayoutDetector

            # Find the best YOLO model
            model_paths = []
            if Config.YOLO_MODEL_PATH:
                mp = Path(Config.YOLO_MODEL_PATH)
                model_paths.append(mp)
                if not mp.is_absolute():
                    model_paths.append(Config.PROJECT_ROOT / mp)
            model_paths.extend([
                Config.PROJECT_ROOT / "runs" / "detect" / "cms1500_yolo_run13" / "weights" / "best.pt",
                Path("/app/runs/detect/cms1500_yolo_run13/weights/best.pt"),
                Config.PROJECT_ROOT / "models" / "cms1500_yolo_v1.pt",
                Path("/app/models/cms1500_yolo_v1.pt"),
                Config.PROJECT_ROOT / "models" / "yolo" / "cms1500_best.pt",
            ])

            yolo = None
            for mp in model_paths:
                if mp and mp.exists():
                    yolo = YOLOLayoutDetector(str(mp), conf=0.15, iou=0.5)
                    self.log(f"YOLO alignment: loaded {mp.name}")
                    break

            if yolo is None:
                return None, 0.0

            # Run YOLO detection to find form blocks
            blocks = yolo.predict(image, page_id=0)
            if not blocks:
                return None, 0.0

            # Find the largest block (the form boundary)
            h, w = image.shape[:2]
            img_area = float(h * w)
            best_block = None
            best_area = 0.0
            for b in blocks:
                bx0, by0, bx1, by1 = b.bbox
                area = float((bx1 - bx0) * (by1 - by0))
                if area > best_area and area > img_area * 0.15:
                    best_area = area
                    best_block = b

            if best_block is None:
                return None, 0.0

            # Use YOLO bbox as ROI, then try to recover a rotated quad inside the ROI.
            bx0, by0, bx1, by1 = best_block.bbox
            bx0, by0, bx1, by1 = int(bx0), int(by0), int(bx1), int(by1)
            pad_x = int(max(2, (bx1 - bx0) * 0.02))
            pad_y = int(max(2, (by1 - by0) * 0.02))
            rx0 = max(0, bx0 - pad_x)
            ry0 = max(0, by0 - pad_y)
            rx1 = min(w, bx1 + pad_x)
            ry1 = min(h, by1 + pad_y)
            roi = image[ry0:ry1, rx0:rx1]
            if roi.size == 0:
                return None, 0.0
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi

            src_quad = None
            quad_score = None
            quad, q_score = self._detect_form_quad(roi_gray)
            if quad is not None:
                quad = quad.astype(np.float32)
                quad[:, 0] += rx0
                quad[:, 1] += ry0
                src_quad = quad
                quad_score = float(q_score)
                self.log(f"YOLO alignment: refined quad in ROI (score={q_score:.2f})")
            else:
                # Fallback: min-area rectangle inside ROI
                try:
                    edges = cv2.Canny(roi_gray, 50, 150)
                    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        area = float(cv2.contourArea(cnt))
                        roi_area = float((rx1 - rx0) * (ry1 - ry0))
                        if area > roi_area * 0.15:
                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect)
                            quad = self._order_quad_points(box)
                            quad[:, 0] += rx0
                            quad[:, 1] += ry0
                            src_quad = quad.astype(np.float32)
                            quad_score = 0.5
                            self.log("YOLO alignment: used minAreaRect fallback in ROI")
                except Exception:
                    pass

            if src_quad is None:
                return None, 0.0

            th, tw = template_shape
            dst_quad = np.array([
                [0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(src_quad, dst_quad)
            if H is not None:
                # Validate the homography
                if self._validate_homography(H, (h, w)):
                    quality = float(best_block.confidence)
                    if quad_score is not None:
                        quality = 0.6 * quality + 0.4 * float(quad_score)
                    self.log(f"YOLO alignment: form bbox ({bx0:.0f},{by0:.0f})-({bx1:.0f},{by1:.0f}), conf={best_block.confidence:.2f}")
                    return H, max(0.6, quality)

        except Exception as e:
            self.log(f"YOLO alignment failed: {e}")

        return None, 0.0

    def _compute_homography(self, image: np.ndarray, template: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute alignment from input image -> template space.

        Priority order:
        1. YOLO-based form detection (most robust for noisy scans)
        2. Contour-based quad detection (good for clean scans)
        3. SIFT/ORB feature matching (fallback)
        NO ECC refinement (prevents tilt on handwritten forms).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        th, tw = template.shape[:2]

        # 0) YOLO-based alignment (best for handwritten/noisy scans)
        H_yolo, q_yolo = self._try_yolo_alignment(image, (th, tw))
        if H_yolo is not None:
            self.log(f"Using YOLO alignment (quality={q_yolo:.2f})")
            return H_yolo, q_yolo

        # 1) Contour-based quad detection (good for clean digital scans)
        quad, q_score = self._detect_form_quad(gray)
        if quad is not None and q_score >= 0.55:
            dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
            if H is not None:
                self.log(f"Using contour quad alignment (quality={q_score:.2f})")
                return H, float(q_score)

        # Build structure masks for feature matching fallback
        def _structure_mask(g: np.ndarray) -> np.ndarray:
            g = cv2.GaussianBlur(g, (3, 3), 0)
            bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 11)
            h, w = bw.shape[:2]
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, w // 18), 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, h // 18)))
            horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
            vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
            mask = cv2.bitwise_or(horiz, vert)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            return mask

        h, w = gray.shape[:2]

        # Template is already grayscale; build structure masks for both (feature fallback)
        template_mask = _structure_mask(template)
        image_mask = _structure_mask(gray)
        
        h, w = gray.shape[:2]
        
        # 1. Coarse Alignment using SIFT (Handles large rotation/scale)
        kp1, des1 = [], None
        kp2, des2 = [], None
        
        # Try SIFT first (robust)
        try:
            sift = cv2.SIFT_create(nfeatures=8000)
            kp1, des1 = sift.detectAndCompute(image_mask, None)
            kp2, des2 = sift.detectAndCompute(template_mask, None)
        except Exception as e:
            print(f"[Alignment] SIFT failed: {e}")
            
        # Fallback to ORB if SIFT fails or returns few keypoints
        if des1 is None or len(kp1) < 10:
            try:
                orb = cv2.ORB_create(nfeatures=8000)
                kp1, des1 = orb.detectAndCompute(image_mask, None)
                kp2, des2 = orb.detectAndCompute(template_mask, None)
            except Exception as e:
                print(f"[Alignment] ORB failed: {e}")
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, 0.0
            
        bf = cv2.BFMatcher()
        good_matches = []
        try:
            # Check descriptor type to choose norm
            norm = cv2.NORM_L2
            if des1.dtype == np.uint8:  # ORB uses uint8 descriptors
                norm = cv2.NORM_HAMMING
                bf = cv2.BFMatcher(norm, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except Exception:
            pass
        
        if len(good_matches) < 10:
            return None, 0.0
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find Homography (Global)
        H_coarse, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # If Homography fails or is invalid, try Affine (simpler, more stable)
        if H_coarse is None or not self._validate_homography(H_coarse, (h, w)):
            try:
                # estimateAffinePartial2D covers rotation, translation, scaling (no shear)
                affine_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if affine_matrix is not None:
                    # Convert 2x3 affine to 3x3 homography
                    H_coarse = np.vstack([affine_matrix, [0, 0, 1]])
                else:
                    return None, 0.0
            except Exception:
                return None, 0.0
            
        # No ECC refinement (prevents tilt)
        return H_coarse, 0.55
    
    def _validate_homography(self, H: np.ndarray, img_shape: Tuple[int, int]) -> bool:
        """
        Validate homography is sane and won't cause distortion.
        STRICT validation to prevent zoom/skew/distortion artifacts.
        """
        if H is None:
            return False
        try:
            h, w = img_shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            if not np.isfinite(warped).all():
                return False
            
            # STRICT: Reject if warped polygon area differs by more than 30%
            # (prevents zoom/scale distortion)
            area = float(cv2.contourArea(warped.astype(np.float32)))
            orig = float(w * h)
            if area < orig * 0.70 or area > orig * 1.30:
                print(f"[Alignment] Rejected: area ratio {area/orig:.2f} out of bounds [0.70, 1.30]")
                return False
            
            # Reject flips (polygon should be convex)
            if not cv2.isContourConvex(warped.astype(np.float32)):
                print("[Alignment] Rejected: warped corners not convex (flip detected)")
                return False
            
            # STRICT: Check for excessive skew/perspective
            # Compute edge lengths of warped quad
            edge_lengths = []
            for i in range(4):
                p1, p2 = warped[i], warped[(i + 1) % 4]
                edge_lengths.append(np.linalg.norm(p2 - p1))
            
            # Top/bottom edges should be similar, left/right should be similar
            top_bottom_ratio = max(edge_lengths[0], edge_lengths[2]) / max(1, min(edge_lengths[0], edge_lengths[2]))
            left_right_ratio = max(edge_lengths[1], edge_lengths[3]) / max(1, min(edge_lengths[1], edge_lengths[3]))
            
            if top_bottom_ratio > 1.25 or left_right_ratio > 1.25:
                print(f"[Alignment] Rejected: excessive perspective (TB={top_bottom_ratio:.2f}, LR={left_right_ratio:.2f})")
                return False
            
            # Check diagonal condition - should form near-rectangle
            diag1 = np.linalg.norm(warped[2] - warped[0])
            diag2 = np.linalg.norm(warped[3] - warped[1])
            diag_ratio = max(diag1, diag2) / max(1, min(diag1, diag2))
            if diag_ratio > 1.15:
                print(f"[Alignment] Rejected: diagonal ratio {diag_ratio:.2f} too high (distortion)")
                return False
            
            return True
        except Exception as e:
            print(f"[Alignment] Validation exception: {e}")
            return False
    
    async def process(self, image: np.ndarray, form_type: FormType) -> AlignmentResult:
        """Align image to template."""
        await self.initialize()

        # Deterministic CMS-1500 alignment path (classical CV only).
        if form_type == FormType.CMS1500:
            try:
                from src.pipelines.registration import get_cms1500_registrar

                registrar = get_cms1500_registrar()
                reg = registrar.register(image)
                if reg.success and reg.aligned_image is not None and reg.homography_input_to_template is not None:
                    return AlignmentResult(
                        success=True,
                        aligned_image=reg.aligned_image,
                        homography_matrix=reg.homography_input_to_template,
                        alignment_quality=float(reg.quality),
                        fallback_used=False,
                        metadata={
                            "alignment_method": reg.method,
                            "registrar_debug": reg.debug,
                        },
                    )
                self.log(
                    f"CMS registrar fallback to legacy aligner (method={getattr(reg, 'method', 'unknown')}, q={float(getattr(reg, 'quality', 0.0)):.2f})"
                )
            except Exception as e:
                self.log(f"CMS registrar exception: {e}")
        
        # Check if we have a template for this form type
        if form_type not in self._templates:
            # Last-chance: try dynamic load from reference assets
            try:
                from src.processing.registration import load_and_process_reference
                ref_data = load_and_process_reference(form_type.value if hasattr(form_type, "value") else str(form_type))
                if ref_data and ref_data.get("image") is not None:
                    self._templates[form_type] = ref_data["image"]
            except Exception:
                pass
        if form_type not in self._templates:
            return AlignmentResult(
                success=False,
                aligned_image=image,
                homography_matrix=None,
                alignment_quality=0.0,
                fallback_used=True,
                error_message="No template available",
                metadata={"alignment_method": "none"},
            )
        
        template = self._templates[form_type]
        th, tw = template.shape[:2]
        
        # Compute homography
        H, quality = self._compute_homography(image, template)
        
        # Validate homography
        if H is not None and self._validate_homography(H, image.shape):
            # IMPORTANT:
            # H is estimated in the *template pixel coordinate system* (dst points come from template).
            # Therefore the aligned output should be rendered in (tw, th), not the original (w, h).
            aligned = cv2.warpPerspective(image, H, (tw, th))

            # Optional fine translation refinement to reduce small shifts
            try:
                if len(aligned.shape) == 3:
                    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
                else:
                    aligned_gray = aligned
                tmpl_gray = template
                # Edge maps for stable correlation
                a_edges = cv2.Canny(aligned_gray, 50, 150)
                t_edges = cv2.Canny(tmpl_gray, 50, 150)
                (shift_x, shift_y), response = cv2.phaseCorrelate(
                    np.float32(t_edges), np.float32(a_edges)
                )
                # Apply only small translations to avoid new drift
                if abs(shift_x) <= 20 and abs(shift_y) <= 20 and response > 0.1:
                    M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
                    aligned = cv2.warpAffine(aligned, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # Edge correlation score for alignment quality (0..1)
                if a_edges.shape == t_edges.shape:
                    corr = cv2.matchTemplate(a_edges, t_edges, cv2.TM_CCOEFF_NORMED)[0][0]
                    edge_score = float((corr + 1.0) / 2.0)
                    if edge_score > quality:
                        quality = edge_score
            except Exception:
                pass
            
            return AlignmentResult(
                success=True,
                aligned_image=aligned,
                homography_matrix=H,
                alignment_quality=quality,
                fallback_used=False,
                metadata={"alignment_method": "legacy_homography"},
            )
        
        # Fallback: return original image
        return AlignmentResult(
            success=False,
            aligned_image=image,
            homography_matrix=None,
            alignment_quality=0.0,
            fallback_used=True,
            error_message="Homography validation failed",
            metadata={"alignment_method": "legacy_failed"},
        )
