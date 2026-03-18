"""
src/vision/image_processor.py
──────────────────────────────
PIL-based image loading, preprocessing, and base64 encoding utilities.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image
    import numpy as np
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class ImageProcessor:
    """
    Handles image loading, resizing, and encoding for vision model input.

    All methods degrade gracefully when PIL is not installed —
    stub methods return empty data so the pipeline can still run.
    """

    DEFAULT_SIZE: Tuple[int, int] = (336, 336)  # LLaVA-1.5 default

    def __init__(self, target_size: Tuple[int, int] = DEFAULT_SIZE) -> None:
        self.target_size = target_size

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self, image_path: str) -> "Image.Image":
        """Load an image from disk and ensure it is RGB."""
        if not _PIL_AVAILABLE:
            raise ImportError("pip install Pillow")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(path).convert("RGB")
        return img

    def load_and_resize(self, image_path: str) -> "Image.Image":
        """Load and resize to target_size."""
        img = self.load(image_path)
        return img.resize(self.target_size, Image.LANCZOS)

    # ── Encoding ───────────────────────────────────────────────────────────────

    def to_base64(self, image_path: str) -> str:
        """Return base64-encoded JPEG string suitable for API payloads."""
        if not _PIL_AVAILABLE:
            return ""
        img = self.load_and_resize(image_path)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def to_numpy(self, image_path: str) -> "np.ndarray":
        """Return image as a float32 numpy array normalised to [0, 1]."""
        if not _PIL_AVAILABLE:
            raise ImportError("pip install Pillow numpy")
        img = self.load_and_resize(image_path)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    # ── Metadata ───────────────────────────────────────────────────────────────

    def get_metadata(self, image_path: str) -> dict:
        """Return basic image metadata."""
        if not _PIL_AVAILABLE:
            return {"path": image_path, "available": False}
        img = self.load(image_path)
        return {
            "path": str(image_path),
            "size": img.size,
            "mode": img.mode,
            "format": img.format,
        }

    # ── Stub fallback ──────────────────────────────────────────────────────────

    @staticmethod
    def placeholder_description(image_path: str) -> str:
        """Return a generic description when PIL is unavailable."""
        name = Path(image_path).stem
        return (
            f"Image '{name}': A high-resolution photograph containing visual structures "
            "with distinct regions of interest including objects, textures, and spatial "
            "relationships that are relevant for analysis."
        )
