"""
src/memory/faiss_memory.py
───────────────────────────
FAISSMemory — production-grade semantic vector store.

Features
────────
• Add texts → embed → FAISS IndexFlatIP (or IVF for scale)
• Semantic search with cosine similarity scores
• Persistent save/load (index + metadata pickle)
• simulate_large_index(n) — populate 100 K+ synthetic entries for benchmarking
• Thread-safe via threading.Lock

Usage
─────
  from src.memory.faiss_memory import FAISSMemory
  from src.memory.embedder import Embedder

  embedder = Embedder.from_config(cfg)
  mem = FAISSMemory(embedder=embedder, cfg=cfg)
  mem.add(["FAISS is a library for similarity search.", "RAG improves LLM accuracy."])
  results = mem.search("vector similarity", k=3)
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from src.memory.embedder import BaseEmbedder

logger = logging.getLogger("multimodal_agent.memory")


class FAISSMemory:
    """
    FAISS-backed vector memory store with metadata management.

    Notes
    ─────
    Uses L2-normalised embeddings + IndexFlatIP ≡ cosine similarity.
    For indices > 50 K entries, automatically switches to IndexIVFFlat.
    """

    _IVF_THRESHOLD = 10_000  # switch to IVF above this count

    def __init__(
        self,
        embedder: BaseEmbedder,
        cfg: Optional[dict] = None,
    ) -> None:
        self._embedder = embedder
        self._dim = embedder.dimension
        self._lock = threading.Lock()

        cfg = cfg or {}
        mem_cfg = cfg.get("memory", {})
        self._index_path = mem_cfg.get("index_path", "vector_store/faiss.index")
        self._meta_path = mem_cfg.get("metadata_path", "vector_store/metadata.pkl")
        self._default_k = mem_cfg.get("top_k", 5)

        # Internal state
        self._index: Optional[Any] = None          # faiss index
        self._texts: List[str] = []                # parallel list of stored texts
        self._metadata: List[Dict[str, Any]] = []  # per-entry metadata

        self._init_index()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_index(self) -> None:
        if not _FAISS_AVAILABLE:
            logger.warning(
                "faiss-cpu not installed. Memory will operate in in-memory numpy mode."
            )
            self._index = None
            return
        self._index = faiss.IndexFlatIP(self._dim)

    def _maybe_upgrade_to_ivf(self) -> None:
        """Upgrade a flat index to IVF when corpus size warrants it."""
        if not _FAISS_AVAILABLE or self._index is None:
            return
        n = self._index.ntotal
        if n >= self._IVF_THRESHOLD and isinstance(self._index, faiss.IndexFlatIP):
            logger.info(f"Upgrading FAISS index to IVFFlat (n={n:,})")
            quantiser = faiss.IndexFlatIP(self._dim)
            nlist = min(256, n // 39)
            ivf = faiss.IndexIVFFlat(quantiser, self._dim, nlist, faiss.METRIC_INNER_PRODUCT)
            # Train on existing vectors
            vecs = faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self._dim
            ).reshape(n, self._dim).copy()
            ivf.train(vecs)
            ivf.add(vecs)
            ivf.nprobe = 32
            self._index = ivf

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def add(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Embed and add a batch of texts to the index."""
        if not texts:
            return

        with self._lock:
            embeddings = self._embedder.embed(texts)  # (N, D) float32

            if _FAISS_AVAILABLE and self._index is not None:
                faiss.normalize_L2(embeddings)
                self._index.add(embeddings)
            else:
                # Fallback: store embeddings in a numpy array
                if not hasattr(self, "_np_store"):
                    self._np_store = embeddings
                else:
                    self._np_store = np.vstack([self._np_store, embeddings])

            self._texts.extend(texts)
            meta = metadata or [{} for _ in texts]
            self._metadata.extend(meta)

        self._maybe_upgrade_to_ivf()
        logger.debug(f"Added {len(texts)} entries. Total: {self.size}")

    def search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most semantically similar entries.

        Returns list of dicts: {text, score, metadata, rank}
        """
        k = k or self._default_k
        if self.size == 0:
            return []

        q_vec = self._embedder.embed(query)  # (1, D)

        with self._lock:
            if _FAISS_AVAILABLE and self._index is not None:
                faiss.normalize_L2(q_vec)
                eff_k = min(k, self._index.ntotal)
                scores, indices = self._index.search(q_vec, eff_k)
                scores = scores[0]
                indices = indices[0]
            else:
                # Numpy cosine similarity fallback
                store = self._np_store  # type: ignore[attr-defined]
                q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
                sims = (store @ q_norm.T).flatten()
                eff_k = min(k, len(sims))
                top_idx = np.argsort(sims)[::-1][:eff_k]
                indices = top_idx
                scores = sims[top_idx]

            results = []
            for rank, (idx, score) in enumerate(zip(indices, scores)):
                if idx < 0 or idx >= len(self._texts):
                    continue
                results.append({
                    "text": self._texts[idx],
                    "score": float(score),
                    "metadata": self._metadata[idx],
                    "rank": rank + 1,
                })

        return results

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate_large_index(self, n: int = 100_000, batch_size: int = 5_000) -> None:
        """
        Populate the index with n synthetic documents for benchmarking.
        Uses batched addition to keep memory usage manageable.
        """
        logger.info(f"Simulating large index with {n:,} entries …")
        t0 = time.time()
        templates = [
            "The concept of {topic} is fundamental to modern AI systems.",
            "Research in {topic} has accelerated significantly in recent years.",
            "{topic} enables breakthrough applications in computer vision and NLP.",
            "Efficient implementation of {topic} requires careful engineering.",
            "Practitioners use {topic} to improve model performance and reliability.",
        ]
        topics = [
            "neural networks", "transformers", "FAISS indexing", "vector search",
            "retrieval-augmented generation", "multi-head attention", "contrastive learning",
            "knowledge distillation", "prompt engineering", "zero-shot learning",
            "federated learning", "quantisation", "pruning", "speculative decoding",
            "mixture of experts", "reinforcement learning from human feedback",
        ]
        added = 0
        while added < n:
            batch_n = min(batch_size, n - added)
            texts = [
                templates[i % len(templates)].format(topics[i % len(topics)])
                + f" [entry {added + i}]"
                for i in range(batch_n)
            ]
            self.add(texts)
            added += batch_n
            logger.debug(f"  … {added:,}/{n:,}")

        elapsed = time.time() - t0
        logger.info(
            f"Large index built: {self.size:,} entries in {elapsed:.2f}s "
            f"({self.size / elapsed:,.0f} entries/s)"
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save the FAISS index and metadata to disk."""
        idx_path = Path(self._index_path)
        meta_path = Path(self._meta_path)
        idx_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if _FAISS_AVAILABLE and self._index is not None:
                faiss.write_index(self._index, str(idx_path))
            with open(meta_path, "wb") as f:
                pickle.dump({"texts": self._texts, "metadata": self._metadata}, f)

        logger.info(f"Memory saved → {idx_path} ({self.size:,} entries)")

    def load(self) -> bool:
        """Load the FAISS index and metadata from disk. Returns True on success."""
        idx_path = Path(self._index_path)
        meta_path = Path(self._meta_path)

        if not idx_path.exists() or not meta_path.exists():
            logger.warning("No saved index found. Starting fresh.")
            return False

        with self._lock:
            if _FAISS_AVAILABLE:
                self._index = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self._texts = data.get("texts", [])
                self._metadata = data.get("metadata", [])

        logger.info(f"Memory loaded ← {idx_path} ({self.size:,} entries)")
        return True

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        if _FAISS_AVAILABLE and self._index is not None:
            return self._index.ntotal
        return len(self._texts)

    def __repr__(self) -> str:
        return f"<FAISSMemory size={self.size:,} dim={self._dim}>"
