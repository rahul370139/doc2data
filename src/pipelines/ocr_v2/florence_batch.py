"""
Batched Florence-2 inference.

Why this exists
---------------
The original ``_florence2_ocr_run`` runs one ``generate()`` call per crop.
For an 86-field CMS-1500, combined with the upscale retry + raw fallback +
consistency check paths, we see up to 200-350 Florence-2 forward passes per
page.  Each forward pass pays the per-call Python/transformers overhead,
so on CPU we spend 30-60 seconds in Florence-2 alone.

This module adds a true batched inference path — N crops go through a
single ``generate()`` call.  Two outcomes:

1. We keep the same model quality (Florence-2-large <OCR> task).
2. Per-crop latency drops ~3-6x on CPU, more on GPU.

The module is designed to be drop-in:

    flo = get_batched_florence2()
    texts = flo.run_batch([crop1, crop2, ...])  # returns list[str]

All output post-processing (placeholder filtering, hallucination checks)
stays in the OCR agent — this module is just the model wrapper.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger("ocr_v2.florence_batch")


# Florence-2 occasionally emits raw `<pad>`, `</s>`, `<s>` tokens when
# ``post_process_generation`` doesn't fully clean its output — this regex is
# a defensive last line of defence that we apply on every F2 output.
_SENTINEL_TOKEN_RE = re.compile(r"</?\s*(pad|s|eos|bos|unk)\s*/?>", re.IGNORECASE)


def _clean_florence_output(text: str) -> str:
    """Strip Florence-2 sentinel tokens + collapse trailing whitespace."""
    if not text:
        return ""
    cleaned = _SENTINEL_TOKEN_RE.sub("", text)
    cleaned = cleaned.replace("\x00", "").replace("\ufffd", "")
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# Module-level singleton state — we only ever want ONE Florence-2 loaded.
_model = None
_processor = None
_device = None
_dtype = None
_load_attempts = 0
_max_attempts = 2
_load_lock = threading.Lock()


def _load_florence2() -> bool:
    """Lazy-load Florence-2-large, preferring GPU float16 when available."""
    global _model, _processor, _device, _dtype, _load_attempts

    with _load_lock:
        if _model is not None and _processor is not None:
            return True
        if _load_attempts >= _max_attempts:
            return False
        _load_attempts += 1

        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as e:
            logger.warning("Florence-2 import failed: %s", e)
            return False

        has_cuda = torch.cuda.is_available()
        model_id = "florence-community/Florence-2-large"

        strategies: List[Tuple[bool, str]] = []
        if has_cuda:
            strategies.append((True, "cuda:0"))
            strategies.append((False, "cuda:0"))
        strategies.append((True, "cpu"))
        strategies.append((False, "cpu"))

        for local_only, device in strategies:
            dtype = torch.float16 if "cuda" in device else torch.float32
            try:
                t0 = time.time()
                proc = AutoProcessor.from_pretrained(
                    model_id, local_files_only=local_only,
                )
                mdl = Florence2ForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    local_files_only=local_only,
                ).eval().to(device)
                _processor = proc
                _model = mdl
                _device = device
                _dtype = dtype
                logger.info(
                    "Florence-2 loaded on %s (%s) in %.1fs",
                    next(mdl.parameters()).device,
                    dtype,
                    time.time() - t0,
                )
                return True
            except Exception as e:
                short_err = str(e).split("\n")[0][:200]
                logger.debug("Florence-2 load failed on %s local=%s: %s",
                             device, local_only, short_err)
                continue

        logger.error("Florence-2 unavailable — all load strategies failed.")
        return False


def _np_to_pil(image: np.ndarray):
    """Convert a numpy crop to PIL RGB."""
    from PIL import Image as PILImage
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return PILImage.fromarray(image).convert("RGB")
    if image.ndim == 3 and image.shape[2] == 4:
        return PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
    return PILImage.fromarray(image)


class BatchedFlorence2:
    """Batched Florence-2 <OCR> runner.

    Uses one ``processor(text=[...], images=[...])`` + ``generate()`` call
    per batch.  Falls back to serial per-crop mode if batching fails for
    any reason (e.g. wildly different crop shapes on CPU with tight RAM).
    """

    def __init__(self, batch_size: int = 8, task: str = "<OCR>",
                 max_new_tokens: int = 128):
        self.batch_size = max(1, int(batch_size))
        self.task = task
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        return _load_florence2()

    def run_batch(self, crops: List[np.ndarray]) -> List[str]:
        """Run Florence-2 <OCR> on a list of crops.

        Returns a list of strings (same length as input).  Empty/None crops
        produce empty strings without a forward pass.
        """
        if not crops:
            return []
        if not _load_florence2():
            return ["" for _ in crops]

        results: List[str] = [""] * len(crops)

        # Separate empty-or-tiny crops (no point running model)
        work_items: List[Tuple[int, np.ndarray]] = []
        for idx, c in enumerate(crops):
            if c is None or c.size == 0:
                continue
            h, w = c.shape[:2]
            if h < 4 or w < 4:
                continue
            work_items.append((idx, c))

        if not work_items:
            return results

        # Process in batches
        for start in range(0, len(work_items), self.batch_size):
            chunk = work_items[start:start + self.batch_size]
            indices = [i for i, _ in chunk]
            imgs = [c for _, c in chunk]
            texts = self._run_one_batch(imgs)
            for idx, text in zip(indices, texts):
                results[idx] = text

        return results

    def run_single(self, crop: np.ndarray) -> str:
        """Convenience: single crop through the batched path."""
        return self.run_batch([crop])[0] if crop is not None else ""

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_one_batch(self, crops: List[np.ndarray]) -> List[str]:
        """Run a single batched generate call.  Falls back to per-crop serial."""
        try:
            return self._run_batched_impl(crops)
        except Exception as e:
            logger.warning("Batched Florence-2 failed (%s) — falling back to serial", e)
            return [self._run_single_impl(c) for c in crops]

    def _run_batched_impl(self, crops: List[np.ndarray]) -> List[str]:
        import torch

        pil_imgs = [_np_to_pil(c) for c in crops]
        pil_imgs = [p for p in pil_imgs if p is not None]
        if not pil_imgs:
            return ["" for _ in crops]

        with self._lock:
            inputs = _processor(
                text=[self.task] * len(pil_imgs),
                images=pil_imgs,
                return_tensors="pt",
                padding=True,
            ).to(_device, _dtype)

            with torch.no_grad():
                gen = _model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )

            # We do NOT skip special tokens here because Florence-2's
            # post_process_generation relies on <s>...</s> markers to bracket
            # the answer region.  But we always strip `<pad>` etc. below.
            raw_texts = _processor.batch_decode(gen, skip_special_tokens=False)

        out: List[str] = []
        for raw, pil in zip(raw_texts, pil_imgs):
            value = ""
            try:
                parsed = _processor.post_process_generation(
                    raw, task=self.task, image_size=(pil.width, pil.height),
                )
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, str) and v.strip():
                            value = v
                            break
                elif isinstance(parsed, str) and parsed.strip():
                    value = parsed
            except Exception:
                value = ""
            # Defence-in-depth — post_process_generation occasionally leaves
            # trailing `<pad>` tokens (e.g. when generation hits max_new_tokens
            # before EOS).  Strip them here so callers never see sentinels.
            cleaned = _clean_florence_output(value)
            if not cleaned and raw:
                # Last-resort: tokenizer output only — tolerate stripping <s>.
                cleaned = _clean_florence_output(
                    raw.replace("<s>", " ").replace("</s>", " ")
                )
            out.append(cleaned)
        return out

    def _run_single_impl(self, crop: np.ndarray) -> str:
        """Per-crop fallback if batching blew up."""
        if crop is None or crop.size == 0:
            return ""
        pil = _np_to_pil(crop)
        if pil is None:
            return ""
        try:
            import torch
            with self._lock:
                inputs = _processor(
                    text=self.task, images=pil, return_tensors="pt",
                ).to(_device, _dtype)
                with torch.no_grad():
                    gen = _model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                raw = _processor.batch_decode(gen, skip_special_tokens=False)[0]
                parsed = _processor.post_process_generation(
                    raw, task=self.task, image_size=(pil.width, pil.height),
                )
            value = ""
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, str) and v.strip():
                        value = v
                        break
            elif isinstance(parsed, str):
                value = parsed
            return _clean_florence_output(value)
        except Exception as e:
            logger.debug("Florence-2 single call failed: %s", e)
        return ""


# ---------------------------------------------------------------------- #
# Module-level singleton accessor
# ---------------------------------------------------------------------- #

_instance: Optional[BatchedFlorence2] = None
_instance_lock = threading.Lock()


def get_batched_florence2(batch_size: Optional[int] = None) -> BatchedFlorence2:
    """Return the process-wide BatchedFlorence2 singleton."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = BatchedFlorence2(batch_size=batch_size or 8)
        elif batch_size is not None:
            _instance.batch_size = max(1, int(batch_size))
        return _instance
