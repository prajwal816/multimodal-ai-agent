"""
src/agent/executor.py
──────────────────────
TaskExecutor — iterates planner Steps and dispatches to the right tool.

For each Step it:
  1. Selects the appropriate tool (vision / memory / search / llm).
  2. Calls the tool.
  3. Records the result on the Step object.
  4. Accumulates a running context summary.
  5. Logs latency and success to MetricsTracker.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from src.agent.planner import Step
from src.llm.llm_backend import BaseLLM
from src.llm.prompt_templates import build_executor_prompt
from src.memory.faiss_memory import FAISSMemory
from src.tools.memory_tool import MemoryRetrievalTool
from src.tools.search_tool import SearchTool
from src.tools.vision_tool import VisionAnalysisTool
from src.utils.metrics import AgentRunMetrics

logger = logging.getLogger("multimodal_agent.executor")


class TaskExecutor:

    def __init__(
        self,
        llm: BaseLLM,
        vision_tool: VisionAnalysisTool,
        memory_tool: MemoryRetrievalTool,
        search_tool: SearchTool,
    ) -> None:
        self._llm = llm
        self._vision_tool = vision_tool
        self._memory_tool = memory_tool
        self._search_tool = search_tool

    def execute(
        self,
        steps: List[Step],
        overall_task: str,
        image_path: Optional[str] = None,
        metrics: Optional[AgentRunMetrics] = None,
    ) -> Dict[str, str]:
        """
        Execute all Steps sequentially.

        Returns a dict mapping step index → step result.
        """
        results: Dict[str, str] = {}
        context_summary = ""

        for step in steps:
            logger.info(f"Executing {step}")
            t0 = time.perf_counter()

            try:
                result = self._dispatch(step, overall_task, context_summary, image_path)
                step.result = result
                step.completed = True
                success = True
            except Exception as exc:
                result = f"[Error in step {step.index}]: {exc}"
                step.result = result
                step.completed = False
                success = False
                logger.error(f"Step {step.index} failed: {exc}", exc_info=True)

            latency_ms = (time.perf_counter() - t0) * 1000
            results[str(step.index)] = result

            # Update rolling summary (last 400 chars of each result)
            context_summary += f"\nStep {step.index} [{step.tool}]: {result[:400]}"

            if metrics:
                metrics.record_step(
                    step_index=step.index,
                    tool_name=step.tool,
                    input_summary=step.description[:100],
                    output_summary=result[:200],
                    latency_ms=latency_ms,
                    success=success,
                )

            logger.info(f"  → Done in {latency_ms:.0f} ms (success={success})")

        return results

    def _dispatch(
        self,
        step: Step,
        overall_task: str,
        context_summary: str,
        image_path: Optional[str],
    ) -> str:
        tool = step.tool.upper()

        if tool == "VISION":
            path = image_path or "data/sample_images/test.jpg"
            return self._vision_tool._run(
                image_path=path,
                prompt=step.description,
            )

        elif tool == "MEMORY":
            return self._memory_tool._run(query=step.description)

        elif tool == "SEARCH":
            return self._search_tool._run(query=step.description)

        elif tool in ("LLM", "NONE"):
            prompt = build_executor_prompt(
                overall_task=overall_task,
                step=step.description,
                previous_summary=context_summary[-600:] if context_summary else "None",
                context="No additional external context.",
            )
            return self._llm.generate(prompt)

        else:
            return f"[Unknown tool: {tool}]"
