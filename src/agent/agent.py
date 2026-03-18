"""
src/agent/agent.py
───────────────────
MultimodalAgent — top-level orchestrator.

Wires together: Config → LLM → Vision → Memory → RAG → Tools → Planner → Executor.
Exposes a single public method: run(task, image_path) → structured result dict.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.agent.executor import TaskExecutor
from src.agent.planner import TaskPlanner
from src.llm.llm_backend import LLMBackend
from src.llm.prompt_templates import build_summary_prompt
from src.memory.embedder import Embedder
from src.memory.faiss_memory import FAISSMemory
from src.rag.rag_pipeline import RAGPipeline
from src.tools.memory_tool import MemoryRetrievalTool
from src.tools.search_tool import SearchTool
from src.tools.vision_tool import VisionAnalysisTool
from src.utils.logger import get_logger_from_config
from src.utils.metrics import MetricsTracker
from src.vision.vision_model import VisionModel

logger = logging.getLogger("multimodal_agent")


class MultimodalAgent:
    """
    Production-grade multimodal AI agent.

    Parameters
    ──────────
    config_path : path to configs/config.yaml (default)

    Quick start
    ───────────
    >>> agent = MultimodalAgent()
    >>> result = agent.run("Analyse the image and summarise key insights.",
    ...                    image_path="data/sample_images/test.jpg")
    >>> print(result["answer"])
    """

    DEFAULT_CONFIG = "configs/config.yaml"

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._cfg = self._load_config(config_path or self.DEFAULT_CONFIG)
        self._log = get_logger_from_config(self._cfg)
        self._metrics = MetricsTracker(
            output_path=self._cfg.get("metrics", {}).get("output_path", "logs/metrics.json")
        )
        self._log.info("Initialising MultimodalAgent …")
        self._setup_subsystems()
        self._log.info("MultimodalAgent ready ✅")

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _setup_subsystems(self) -> None:
        cfg = self._cfg

        # LLM
        self._llm = LLMBackend.from_config(cfg)
        self._log.info(f"LLM backend: {self._llm}")

        # Vision
        self._vision_model = VisionModel.from_config(cfg)
        self._log.info(f"Vision model: {self._vision_model}")

        # Memory
        self._embedder = Embedder.from_config(cfg)
        self._memory = FAISSMemory(embedder=self._embedder, cfg=cfg)

        # RAG (must be instatiated before ingestion)
        self._rag = RAGPipeline.from_config(cfg, memory=self._memory, llm=self._llm)

        # Try to load persisted index; otherwise ingest corpus
        if not self._memory.load():
            self._ingest_corpus()

        self._log.info(f"Memory store: {self._memory}")

        # LangChain Tools
        self._vision_tool = VisionAnalysisTool(vision_model=self._vision_model)
        self._memory_tool = MemoryRetrievalTool(memory=self._memory)
        self._search_tool = SearchTool.from_config(cfg)

        # Planner & Executor
        self._planner = TaskPlanner(llm=self._llm, max_steps=8)
        self._executor = TaskExecutor(
            llm=self._llm,
            vision_tool=self._vision_tool,
            memory_tool=self._memory_tool,
            search_tool=self._search_tool,
        )

    def _ingest_corpus(self) -> None:
        corpus_path = self._cfg.get("rag", {}).get("corpus_path", "data/sample_documents.txt")
        if Path(corpus_path).exists():
            n = self._rag.ingest_file(corpus_path)
            self._log.info(f"Ingested corpus: {corpus_path} ({n} chunks)")
        else:
            self._log.warning(f"Corpus not found: {corpus_path}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a multimodal task.

        Parameters
        ──────────
        task       : natural-language task description
        image_path : optional path to an image file

        Returns
        ───────
        dict with keys: task, answer, sources, plan_steps, step_results,
                        metrics, image_path
        """
        self._log.info(f"▶ Task: {task}")
        m = self._metrics.start_run(task=task, image_path=image_path)

        # Step 1 — RAG retrieval for background context
        rag_result = self._rag.query(task)
        m.record_retrieval(
            query=task,
            retrieved_k=len(rag_result.sources),
            top_score=rag_result.sources[0]["score"] if rag_result.sources else 0.0,
            latency_ms=rag_result.retrieval_latency_ms,
        )

        # Step 2 — Plan
        steps = self._planner.plan(task)

        # Step 3 — Execute
        step_results = self._executor.execute(
            steps=steps,
            overall_task=task,
            image_path=image_path,
            metrics=m,
        )

        # Step 4 — Synthesise final answer
        combined = (
            f"Task: {task}\n\n"
            f"RAG Answer: {rag_result.answer}\n\n"
            + "\n".join(
                f"Step {k}: {v[:300]}" for k, v in step_results.items()
            )
        )
        final_answer = self._llm.generate(build_summary_prompt(combined))

        # Finalise metrics
        m.finalise(goal_completed=True)
        self._log.info(f"\n{m.summary_str()}")

        return {
            "task": task,
            "answer": final_answer,
            "rag_answer": rag_result.answer,
            "sources": rag_result.sources,
            "plan_steps": [
                {"index": s.index, "tool": s.tool, "description": s.description}
                for s in steps
            ],
            "step_results": step_results,
            "metrics": m.to_dict(),
            "image_path": image_path,
        }

    def query_rag(self, query: str) -> Dict[str, Any]:
        """Run RAG-only (no planning) for a simple knowledge query."""
        result = self._rag.query(query)
        return result.to_dict()

    def benchmark_memory(self, n: int = 100_000) -> Dict[str, Any]:
        """Populate the index with n synthetic entries and report performance."""
        t0 = time.perf_counter()
        self._memory.simulate_large_index(n=n)
        elapsed = time.perf_counter() - t0

        # Warm search
        results = self._memory.search("neural network training efficiency", k=5)
        return {
            "index_size": self._memory.size,
            "build_time_s": round(elapsed, 2),
            "throughput_entries_per_s": round(n / elapsed),
            "sample_results": results[:3],
        }

    def save_memory(self) -> None:
        """Persist the FAISS index to disk."""
        self._memory.save()

    @classmethod
    def from_config(cls, config_path: str) -> "MultimodalAgent":
        return cls(config_path=config_path)
