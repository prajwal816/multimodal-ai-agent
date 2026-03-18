"""
src/memory/embedder.py
───────────────────────
Text embedder using sentence-transformers (or stub random vectors).

Supported backends (configs/config.yaml → embeddings.backend):
  • "sentence_transformers" — real embeddings via all-MiniLM-L6-v2
  • "stub"                 — deterministic random vectors (no ML deps)
"""

from __future__ import annotations

import hashlib
from typing import List, Union

import numpy as np


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseEmbedder:

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


# ── Stub embedder ──────────────────────────────────────────────────────────────

class StubEmbedder(BaseEmbedder):
    """
    Returns deterministic pseudo-random unit vectors derived from text hashes.
    Semantically similar texts will NOT have high cosine similarity —
    use only for pipeline testing and CI.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _text_to_vector(self, text: str) -> np.ndarray:
        # Seed RNG with MD5 hash of text for determinism
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)  # L2 normalise

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([self._text_to_vector(t) for t in texts])


# ── SentenceTransformers embedder ─────────────────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Real semantic embeddings via sentence-transformers.
    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError("pip install sentence-transformers") from e

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


# ── Factory ────────────────────────────────────────────────────────────────────

class Embedder:
    """Factory that returns the correct BaseEmbedder from config."""

    @staticmethod
    def from_config(cfg: dict) -> BaseEmbedder:
        emb_cfg = cfg.get("embeddings", {})
        backend = emb_cfg.get("backend", "stub").lower()
        dim = emb_cfg.get("dimension", 384)

        if backend == "stub":
            return StubEmbedder(dim=dim)
        elif backend == "sentence_transformers":
            model = emb_cfg.get("model", "all-MiniLM-L6-v2")
            try:
                return SentenceTransformerEmbedder(model_name=model)
            except ImportError:
                # Graceful degradation
                return StubEmbedder(dim=dim)
        else:
            raise ValueError(
                f"Unknown embeddings backend: '{backend}'. "
                "Choose from: 'stub', 'sentence_transformers'."
            )
