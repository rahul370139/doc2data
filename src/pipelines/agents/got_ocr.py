"""
GOT-OCR 2.0 agent — General OCR Transformer as a rescue engine.

PURPOSE: Add a second-opinion OCR engine whose errors are *uncorrelated*
with Florence-2 and TrOCR so the rescue ladder benefits from model
diversity.  GOT-OCR 2.0 is a unified vision-language OCR model that
handles printed, handwritten, and mixed-content text at the line level
and returns character-level bounding boxes when asked.  On the IAM
handwriting benchmark it reaches ~3.4% CER (TrOCR-handwritten is 2.89%
and still wins on clean handwriting), but GOT-OCR shines on:

  - Short numeric fields (NPI, phone, zip, tax-ID) because its decoder
    is pre-trained on mixed printed/handwritten forms.
  - Crops with mixed printed labels + handwriting, where TrOCR's IAM
    prior over-confidently cleans up the printed parts.
  - Crops where Florence-2 hallucinates template text — the GOT decoder
    has a much stronger letter-level prior and rarely emits full
    template phrases.

USE CASE: The graph's rescue ladder picks this engine when TrOCR and
the VLM disagree or both fail validation.  Running it as a third,
diverse model lets ``_is_text_better`` pick the consensus winner.

Model: ``stepfun-ai/GOT-OCR-2.0-hf`` (Apache-2.0, 580M params, merged
into ``transformers>=4.49``).  Inference ~1–3s per field crop on an
A100/H100/GB10.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.pipelines.core import BaseAgent, PipelineConfig

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


_DEFAULT_MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"


class GOTOCRAgent(BaseAgent):
    """Third-opinion OCR engine for the rescue ladder.

    This agent is intentionally *thin*: it loads the HF model lazily,
    holds a lock around the non-thread-safe ``generate()`` call, and
    returns ``(text, confidence)``.  All ladder integration happens in
    ``rescue_strategies._strategy_got_ocr``.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__("GOTOCRAgent")
        self.config = config
        self._model = None
        self._processor = None
        # generate() is not thread-safe and the decoder holds KV-cache
        # across calls.  The rescue node can batch several fields in
        # parallel via asyncio.to_thread, so serialize on this lock.
        self._lock = threading.Lock()
        self._device: str = "cpu"
        self._load_attempts = 0
        self._max_load_attempts = 2
        self._load_failed = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self):
        """No-op at startup — we lazy-load on first rescue call.

        Pre-loading here would add ~8–15s to cold-start and the GOT-OCR
        engine only fires as a rescue strategy, so most pages never
        need it.  Loading on demand keeps the common path fast.
        """
        if self._initialized:
            return
        self._initialized = True

    async def process(self, *args, **kwargs):
        """BaseAgent abstract surface — delegate to ``recognize``."""
        return self.recognize(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load(self) -> bool:
        """Lazy-load GOT-OCR 2.0; returns True on success.

        Why bfloat16: the model is 580M params with a Qwen-0.5B decoder
        that happily runs in bf16 on any Ampere+ GPU.  Halves VRAM,
        no measurable accuracy loss on single-line crops.
        """
        if self._model is not None and self._processor is not None:
            return True
        if self._load_failed or self._load_attempts >= self._max_load_attempts:
            return False
        self._load_attempts += 1

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as e:
            self.log(f"required transformers classes missing: {e}")
            self._load_failed = True
            return False

        model_id = getattr(
            self.config, "got_ocr_model_id", _DEFAULT_MODEL_ID,
        ) or _DEFAULT_MODEL_ID

        has_cuda = torch.cuda.is_available()
        self._device = "cuda" if has_cuda else "cpu"
        dtype = torch.bfloat16 if has_cuda else torch.float32

        # Pick up the HF access token from the environment.  We accept
        # either ``HF_ACCESS_TOKEN`` (our .env convention) or the
        # canonical ``HUGGING_FACE_HUB_TOKEN`` / ``HF_TOKEN`` names that
        # ``huggingface_hub`` already reads automatically.  Explicitly
        # passing ``token=`` makes the call work even on images where
        # the auth isn't configured globally.
        hf_token = (
            _os.environ.get("HF_ACCESS_TOKEN")
            or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or _os.environ.get("HF_TOKEN")
            or None
        )
        if hf_token:
            hf_token = hf_token.strip() or None

        t0 = time.time()
        self.log(f"loading {model_id} on {self._device} (bf16={has_cuda}) …")

        # IMPORTANT: we intentionally do *not* pass ``device_map`` to
        # ``from_pretrained``.  ``device_map`` triggers
        # ``accelerate.init_empty_weights()`` which requires the
        # ``accelerate`` package.  On minimal DGX containers that pkg
        # might be missing, and historically has been — see the April
        # 2026 benchmark where every GOT-OCR call returned empty.
        # We do our own ``.to(device)`` which is equivalent for a
        # single-GPU deployment and has zero extra dependencies.
        try:
            from_kwargs = {"dtype": dtype}
            if hf_token:
                from_kwargs["token"] = hf_token

            self._processor = AutoProcessor.from_pretrained(
                model_id,
                use_fast=True,
                token=hf_token if hf_token else None,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **from_kwargs,
            )
            self._model = self._model.to(self._device)
            self._model.eval()
            self.log(f"loaded in {time.time() - t0:.1f}s")
            return True
        except TypeError as e:
            # Some older transformers releases called it ``torch_dtype``
            # instead of ``dtype``; retry once with the legacy name.
            if "dtype" in str(e).lower():
                self.log("retrying load with torch_dtype= (older transformers) …")
                try:
                    legacy_kwargs = {"torch_dtype": dtype}
                    if hf_token:
                        legacy_kwargs["token"] = hf_token
                    self._model = AutoModelForImageTextToText.from_pretrained(
                        model_id,
                        **legacy_kwargs,
                    )
                    self._model = self._model.to(self._device)
                    self._model.eval()
                    self.log(f"loaded (legacy kwargs) in {time.time() - t0:.1f}s")
                    return True
                except Exception as e2:
                    self.log(f"legacy-kwarg load failed: {e2}")
                    self._model = None
                    self._processor = None
                    self._load_failed = True
                    return False
            self.log(f"load failed (TypeError): {e}")
            self._model = None
            self._processor = None
            self._load_failed = True
            return False
        except Exception as e:
            # ``transformers<4.49`` has no GOT-OCR 2.0 classes; bail gracefully.
            # Accelerate-missing errors used to live here too — now avoided
            # by dropping ``device_map``.
            self.log(f"load failed: {e}")
            self._model = None
            self._processor = None
            self._load_failed = True
            return False

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing — GOT-OCR does its own resize internally.

        We only normalise the crop shape (3-channel RGB uint8) and
        up-scale tiny crops so the ViT encoder has >48px of height to
        work with — below that the stroke detail is destroyed regardless
        of the model.
        """
        if image is None or image.size == 0:
            return image
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        h, w = image.shape[:2]
        if h < 48:
            scale = max(2.0, 48.0 / h)
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        return image

    def recognize(
        self,
        image: np.ndarray,
        *,
        format_markdown: bool = False,
        max_new_tokens: int = 96,
    ) -> Tuple[str, float]:
        """Run GOT-OCR 2.0 on a single crop.

        Returns ``(text, confidence)``.  Confidence is derived from the
        average per-token log-prob (same approach we use for TrOCR) —
        GOT-OCR's decoder is calibrated enough that ``exp(mean_logp)``
        is a reasonable proxy for field-level reliability.

        ``format_markdown`` enables the LaTeX/markdown output mode.
        For CMS-1500 field crops (single-line, short) this is disabled
        by default; tables use it via the labeling agent.

        ``max_new_tokens`` defaults to 96 — CMS-1500 fields rarely
        exceed ~30 characters and capping the generation avoids runaway
        decoding when the model sees a blank crop.
        """
        if image is None or image.size == 0:
            return "", 0.0
        if not self._load():
            return "", 0.0

        try:
            import torch
            from PIL import Image as PILImage
        except Exception:
            return "", 0.0

        processed = self._preprocess(image)
        pil_img = PILImage.fromarray(processed)

        with self._lock:
            try:
                inputs = self._processor(
                    pil_img, return_tensors="pt", format=format_markdown,
                ).to(self._device)
            except TypeError:
                # Older transformers — ``format`` kwarg not supported.
                inputs = self._processor(
                    pil_img, return_tensors="pt",
                ).to(self._device)
            except Exception as e:
                self.log(f"processor failed: {e}")
                return "", 0.0

            try:
                with torch.no_grad():
                    gen_kwargs = dict(
                        do_sample=False,
                        tokenizer=self._processor.tokenizer,
                        stop_strings="<|im_end|>",
                        max_new_tokens=max_new_tokens,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    outputs = self._model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                self.log(f"generate failed: {e}")
                return "", 0.0

            try:
                input_len = int(inputs["input_ids"].shape[1])
                new_tokens = outputs.sequences[:, input_len:]
                text = self._processor.batch_decode(
                    new_tokens, skip_special_tokens=True,
                )[0]
            except Exception as e:
                self.log(f"decode failed: {e}")
                return "", 0.0

            # Average per-step max prob → confidence.  ``scores`` is a
            # tuple of (batch, vocab) logits per generated step.  We
            # soft-max once, take the max per step, and average.
            conf = 0.70
            try:
                if getattr(outputs, "scores", None):
                    step_confs = []
                    for step_logits in outputs.scores:
                        p = torch.softmax(step_logits, dim=-1)
                        step_confs.append(float(p.max(dim=-1).values[0].item()))
                    if step_confs:
                        conf = sum(step_confs) / len(step_confs)
            except Exception:
                pass

        text = (text or "").strip()
        # GOT-OCR occasionally echoes the stop phrase — trim it.
        for tail in ("<|im_end|>", "<|endoftext|>"):
            if text.endswith(tail):
                text = text[: -len(tail)].strip()

        return text, float(conf)

    # ------------------------------------------------------------------ #
    # Introspection — used by the benchmark harness
    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        """True iff the model has been (or can be) loaded."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        return self._load()
