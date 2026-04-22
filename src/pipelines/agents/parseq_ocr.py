"""
PARSeq agent — Permuted Autoregressive Sequence scene-text recogniser.

PURPOSE: Add a fast (23M params, ~30-50ms / crop on GPU), scene-text-
specialist OCR engine to the rescue ladder.  PARSeq complements the
heavier engines we already have:

  * ``Florence-2`` (~800M params, ~400-800ms/crop) — good on mixed
    handwriting + printed text.
  * ``VLM`` (MiniCPM-V ~8B, ~2-15s/crop) — wins on context-dependent
    digit disambiguation but 40-100× slower.
  * ``GOT-OCR 2.0`` (~580M, ~1-2s/crop) — kept in the benchmark but
    dropped from the default ladder; no post-cleaning accuracy gain.

PARSeq's sweet spot is CMS-1500's *printed* short fields — NPI, phone,
zip, tax-ID, state codes — where most data entry is typewritten onto
the form.  The published benchmarks (ECCV 2022) put PARSeq at 97-99%
accuracy on scene text datasets like IC13/SVT where the text style is
essentially "printed at funny angles", which is close enough to laser-
printed form fills to be useful.  It's *not* expected to win on
handwriting (IAM trained models would need fine-tuning for that), so
the handwriting ladder doesn't include it.

MODEL: ``baudm/parseq`` via ``torch.hub``.  Variants:
  * ``parseq`` (base) — 23.8M params, img 128×32, default.
  * ``parseq_tiny`` — smaller (d_model=192), ~8M params.  Faster but
    less accurate; use if GPU memory pressure warrants it.

DEPENDENCIES: requires ``pytorch_lightning`` (the upstream model class
extends ``pl.LightningModule``) and ``timm``.  We already have
``timm``; ``pytorch_lightning`` is added to requirements.  If either
is missing the agent logs a warning and returns empty strings — the
rescue ladder treats that as "strategy unavailable" and moves on.

INTEGRATION: the rescue ladder calls ``agent.recognize(crop)`` via
``_strategy_parseq`` in ``graph/rescue_strategies.py``.  No template
subtraction happens in this file — the caller feeds us an already-
subtracted crop (see ``_get_subtracted_crop``) because we want every
vision rescue to see the same clean pixels.
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


_DEFAULT_VARIANT = "parseq"


class ParseqAgent(BaseAgent):
    """Thin wrapper around ``baudm/parseq`` for the rescue ladder.

    Lazy-loads the model on first call.  Thread-safe via an internal
    lock because PyTorch Hub models share a single forward pass state
    that's not re-entrant when we batch multiple fields in parallel
    under ``asyncio.to_thread``.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__("ParseqAgent")
        self.config = config
        self._model = None
        self._img_transform = None
        self._lock = threading.Lock()
        self._device: str = "cpu"
        self._load_failed = False
        self._variant: str = getattr(
            config, "parseq_variant", _DEFAULT_VARIANT
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self):
        """No-op at startup; we lazy-load on first recognise call.

        Pre-warming here would add a torch.hub network fetch to cold
        start even on pages where PARSeq never fires (e.g. a fully-
        widgetised PDF).  Deferring keeps the common path snappy.
        """
        if self._initialized:
            return
        self._initialized = True

    async def process(self, *args, **kwargs):
        return self.recognize(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load(self) -> bool:
        """Attempt to load PARSeq via torch.hub.  Returns True on success.

        Why torch.hub (and not a minimal HuggingFace wrapper): the
        upstream implementation is tightly coupled to ``pytorch_lightning``
        and ``strhub.data.module.SceneTextDataModule``.  Reimplementing
        the preprocessing + tokenizer would be ~200 LoC and a
        maintenance burden; pinning to ``torch.hub`` keeps us on the
        same code path as the authors' published checkpoints.
        """
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        try:
            import torch
        except Exception as e:
            self.log(f"torch not importable: {e}")
            self._load_failed = True
            return False

        # ``pytorch_lightning`` import is what usually trips — check
        # early so the user gets a clear error rather than a cryptic
        # one buried in torch.hub's internals.
        try:
            import pytorch_lightning  # noqa: F401
        except Exception as e:
            self.log(
                f"pytorch_lightning missing — PARSeq disabled: {e}. "
                "Install with 'pip install pytorch_lightning timm' to "
                "enable."
            )
            self._load_failed = True
            return False
        try:
            import timm  # noqa: F401
        except Exception as e:
            self.log(f"timm missing — PARSeq disabled: {e}")
            self._load_failed = True
            return False

        try:
            has_cuda = torch.cuda.is_available()
            self._device = "cuda" if has_cuda else "cpu"
            t0 = time.time()
            self.log(
                f"loading parseq variant={self._variant} on {self._device} "
                "via torch.hub …"
            )
            # ``trust_repo=True`` silences the interactive prompt in
            # non-TTY envs (Docker, Jupyter).  ``skip_validation=False``
            # (default) still verifies we're pulling the expected repo.
            try:
                model = torch.hub.load(
                    "baudm/parseq",
                    self._variant,
                    pretrained=True,
                    trust_repo=True,
                )
            except TypeError:
                # Older torch without ``trust_repo`` — fall back.
                model = torch.hub.load(
                    "baudm/parseq", self._variant, pretrained=True,
                )

            # Cache the same image transform the authors ship so the
            # preprocessing exactly matches training.
            try:
                from strhub.data.module import SceneTextDataModule

                self._img_transform = SceneTextDataModule.get_transform(
                    model.hparams.img_size
                )
            except Exception as e:
                self.log(f"couldn't load SceneTextDataModule transform: {e}")
                # Minimal fallback — preserve operational inference even
                # when optional PARSeq training deps (e.g. lmdb) are not
                # installed in production images.
                try:
                    from torchvision import transforms
                    h, w = getattr(model.hparams, "img_size", (32, 128))
                    self._img_transform = transforms.Compose([
                        transforms.Resize((int(h), int(w))),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                        ),
                    ])
                except Exception as e2:
                    self.log(f"fallback transform failed: {e2}")
                    self._img_transform = None

            model = model.to(self._device).eval()
            self._model = model
            self.log(f"loaded parseq in {time.time() - t0:.1f}s")
            return True

        except Exception as e:
            self.log(f"PARSeq load failed: {e}")
            self._model = None
            self._img_transform = None
            self._load_failed = True
            return False

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def _to_pil(self, image: np.ndarray):
        """Convert a numpy crop (BGR/RGB/gray) to a PIL RGB image."""
        from PIL import Image as PILImage

        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        # OpenCV can hand us either BGR or RGB depending on the caller.
        # Most of our pipeline is RGB by now but the processing module
        # still produces BGR occasionally; a cheap heuristic — if the
        # first channel mean is consistently less red than the last,
        # assume BGR and swap.  This is a heuristic only; the caller
        # can disambiguate by passing an already-RGB np array.
        return PILImage.fromarray(arr)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """Run PARSeq on a single line-shaped crop.

        Returns ``(text, confidence)``.  Confidence is the mean of
        per-token softmax max probs, same convention as GOT-OCR.  We
        clip confidence to 0.99 so a lucky-looking PARSeq output
        doesn't unilaterally beat a validator-verified Florence-2
        candidate in ``_is_text_better``.

        Silent failures (model not loaded, transform missing, numpy
        decode error) all return ``("", 0.0)`` so the rescue ladder
        treats them as "skip this strategy".
        """
        if image is None or image.size == 0:
            return "", 0.0
        if not self._load():
            return "", 0.0
        if self._img_transform is None or self._model is None:
            return "", 0.0

        try:
            import torch
        except Exception:
            return "", 0.0

        pil = self._to_pil(image)

        with self._lock:
            try:
                tensor = self._img_transform(pil).unsqueeze(0).to(self._device)
            except Exception as e:
                self.log(f"transform failed: {e}")
                return "", 0.0

            try:
                with torch.no_grad():
                    logits = self._model(tensor)
            except Exception as e:
                self.log(f"forward failed: {e}")
                return "", 0.0

            try:
                pred = logits.softmax(-1)
                label, confidence = self._model.tokenizer.decode(pred)
                text = label[0] if isinstance(label, (list, tuple)) else str(label)
                if isinstance(confidence, (list, tuple)):
                    # confidence is a list of per-sample conf vectors;
                    # take the mean of the first sample.
                    c = confidence[0]
                    try:
                        conf = float(c.mean().item()) if hasattr(c, "mean") \
                            else float(sum(c) / max(1, len(c)))
                    except Exception:
                        conf = 0.70
                else:
                    try:
                        conf = float(confidence)
                    except Exception:
                        conf = 0.70
            except Exception as e:
                self.log(f"decode failed: {e}")
                return "", 0.0

        text = (text or "").strip()
        # Cap at 0.99 — PARSeq's softmax can be over-confident on short
        # strings, and we don't want it to dominate consensus voting.
        conf = min(float(conf), 0.99)
        return text, conf

    # ------------------------------------------------------------------ #
    # Introspection — used by the benchmark harness
    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        return self._load()
