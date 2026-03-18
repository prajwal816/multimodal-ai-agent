"""
src/vision/vision_model.py
───────────────────────────
VisionModel — wraps LLaVA-1.5 or returns a stub description.

Supported backends (configs/config.yaml → vision.backend):
  • "llava"  — HuggingFace LLaVA-1.5-7B (requires GPU or slow CPU)
  • "stub"   — Returns a rich descriptive placeholder instantly

Usage
─────
  from src.vision.vision_model import VisionModel
  vm = VisionModel.from_config(cfg)
  description = vm.describe(image_path="data/sample_images/test.jpg",
                             prompt="What objects are in this image?")
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from src.vision.image_processor import ImageProcessor


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseVisionModel(ABC):

    @abstractmethod
    def describe(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ── Stub backend ───────────────────────────────────────────────────────────────

class StubVisionModel(BaseVisionModel):
    """
    Returns a descriptive placeholder for any image without any ML deps.
    Ideal for offline development, CI, and CPU-only environments.
    """

    TEMPLATE = (
        "Visual Analysis of '{name}':\n"
        "• Primary subjects: geometric shapes, textured surfaces, and foreground objects\n"
        "• Colour palette: warm tones with contrasting highlights\n"
        "• Spatial layout: objects are distributed across the frame with clear depth cues\n"
        "• Notable features: distinct edges, shadows indicating a point light source\n"
        "• Contextual inference: the scene appears to be {context}\n"
        "• Relevance to query '{prompt}': the image contains sufficient visual structure "
        "to address the query with high confidence."
    )

    CONTEXT_HINTS = [
        "an indoor environment with artificial lighting",
        "an outdoor scene captured during daylight",
        "a close-up macro shot of a textured surface",
        "a document or diagram with structured information",
        "a natural landscape with organic forms",
    ]

    def __init__(self, description_template: Optional[str] = None) -> None:
        self.description_template = description_template or self.TEMPLATE
        self._processor = ImageProcessor()

    def describe(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        time.sleep(0.1)  # Simulate inference latency
        name = Path(image_path).stem if image_path else "unknown"
        # Pick a deterministic context hint based on filename hash
        context = self.CONTEXT_HINTS[hash(name) % len(self.CONTEXT_HINTS)]
        return self.TEMPLATE.format(name=name, context=context, prompt=prompt[:60])


# ── LLaVA backend ─────────────────────────────────────────────────────────────

class LLaVAModel(BaseVisionModel):
    """
    LLaVA-1.5 vision-language model via HuggingFace transformers.

    Model: llava-hf/llava-1.5-7b-hf (default)
    Requires: pip install transformers torch Pillow
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cpu",
        max_new_tokens: int = 256,
    ) -> None:
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration  # type: ignore
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError("pip install transformers torch Pillow") from e

        self.device = device
        self.max_new_tokens = max_new_tokens
        self._processor_hf = LlavaNextProcessor.from_pretrained(model_id)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        if device == "cuda":
            self._model.to("cuda")
        self._img_processor = ImageProcessor()

    def describe(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        try:
            from PIL import Image  # type: ignore
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError("pip install Pillow torch") from e

        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self._processor_hf.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor_hf(
            images=image, text=text_prompt, return_tensors="pt"
        )
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        full_text = self._processor_hf.decode(output_ids[0], skip_special_tokens=True)
        # Strip the prompt echo
        return full_text[len(text_prompt):].strip()


# ── Factory ────────────────────────────────────────────────────────────────────

class VisionModel:
    """Factory returning the correct BaseVisionModel from config."""

    @staticmethod
    def from_config(cfg: dict) -> BaseVisionModel:
        vision_cfg = cfg.get("vision", {})
        backend = vision_cfg.get("backend", "stub").lower()

        if backend == "stub":
            stub_cfg = vision_cfg.get("stub", {})
            return StubVisionModel(
                description_template=stub_cfg.get("description_template")
            )
        elif backend == "llava":
            llava_cfg = vision_cfg.get("llava", {})
            return LLaVAModel(
                model_id=llava_cfg.get("model_id", "llava-hf/llava-1.5-7b-hf"),
                device=llava_cfg.get("device", "cpu"),
                max_new_tokens=llava_cfg.get("max_new_tokens", 256),
            )
        else:
            raise ValueError(
                f"Unknown vision backend: '{backend}'. Choose from: 'stub', 'llava'."
            )
