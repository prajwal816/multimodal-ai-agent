"""
src/rag/rag_pipeline.py
────────────────────────
RAGPipeline — query → retrieve → augment → generate.

Flow
────
  1. Embed the query.
  2. Retrieve top-k semantically similar chunks from FAISSMemory.
  3. Build an augmented prompt (context + query).
  4. Call LLMBackend.generate().
  5. Return structured result: answer, sources, retrieval metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.llm.llm_backend import BaseLLM
from src.llm.prompt_templates import build_rag_prompt
from src.memory.faiss_memory import FAISSMemory
from src.memory.embedder import BaseEmbedder
from src.rag.document_loader import Document, DocumentLoader

logger = logging.getLogger("multimodal_agent.rag")


class RAGResult:
    """Structured result returned by RAGPipeline.query()."""

    def __init__(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        query: str,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
    ) -> None:
        self.answer = answer
        self.sources = sources
        self.query = query
        self.retrieval_latency_ms = retrieval_latency_ms
        self.generation_latency_ms = generation_latency_ms
        self.total_latency_ms = retrieval_latency_ms + generation_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"RAGResult(\n"
            f"  answer={self.answer[:80]!r}…\n"
            f"  sources={len(self.sources)} chunks\n"
            f"  latency={self.total_latency_ms:.1f} ms\n"
            f")"
        )


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Parameters
    ──────────
    memory      : FAISSMemory instance (shared with the agent)
    llm         : BaseLLM instance
    top_k       : number of chunks to retrieve
    """

    def __init__(
        self,
        memory: FAISSMemory,
        llm: BaseLLM,
        top_k: int = 5,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._top_k = top_k
        self._loader = DocumentLoader()

    # ── Corpus ingestion ───────────────────────────────────────────────────────

    def ingest_file(self, path: str) -> int:
        """Ingest a document file into the memory store. Returns chunk count."""
        docs = self._loader.load(path)
        self._ingest_docs(docs)
        logger.info(f"Ingested {len(docs)} chunks from {path!r}")
        return len(docs)

    def ingest_text(self, text: str, source: str = "inline") -> int:
        """Ingest a raw string into memory. Returns chunk count."""
        docs = self._loader.load_text(text, source=source)
        self._ingest_docs(docs)
        logger.info(f"Ingested {len(docs)} inline chunks")
        return len(docs)

    def _ingest_docs(self, docs: List[Document]) -> None:
        texts = [d.content for d in docs]
        meta = [{"source": d.source, "chunk": d.chunk_index} for d in docs]
        self._memory.add(texts, meta)

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, query: str, k: Optional[int] = None) -> RAGResult:
        """
        Execute the full RAG pipeline for a given query.

        Returns a RAGResult with the generated answer and source snippets.
        """
        k = k or self._top_k

        # Step 1 — Retrieve
        t0 = time.perf_counter()
        retrieved = self._memory.search(query, k=k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            f"Retrieved {len(retrieved)} chunks for query {query[:60]!r} "
            f"in {retrieval_ms:.1f} ms"
        )

        # Step 2 — Augment
        context_chunks = [r["text"] for r in retrieved]
        if not context_chunks:
            context_chunks = ["No relevant context found in the knowledge base."]
        prompt = build_rag_prompt(query=query, context_chunks=context_chunks)

        # Step 3 — Generate
        t1 = time.perf_counter()
        answer = self._llm.generate(prompt)
        generation_ms = (time.perf_counter() - t1) * 1000

        logger.debug(f"Generated answer in {generation_ms:.1f} ms")

        return RAGResult(
            answer=answer,
            sources=retrieved,
            query=query,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )

    # ── Convenience ────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        memory: FAISSMemory,
        llm: BaseLLM,
    ) -> "RAGPipeline":
        rag_cfg = cfg.get("rag", {})
        return cls(
            memory=memory,
            llm=llm,
            top_k=rag_cfg.get("top_k", 5),
        )
