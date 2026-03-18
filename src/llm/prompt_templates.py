"""
src/llm/prompt_templates.py
────────────────────────────
Reusable prompt templates for RAG, vision-language, planning, and summarisation.
All templates are pure Python f-strings — no external dependency required.
"""

from __future__ import annotations

from typing import List


# ── RAG Augmentation Template ─────────────────────────────────────────────────

RAG_TEMPLATE = """\
You are an expert AI assistant with access to a curated knowledge base.

## Retrieved Context
{context}

## User Question
{query}

## Instructions
Answer the question using ONLY the retrieved context above.
If the context does not contain enough information, say so clearly.
Cite the relevant passage numbers when applicable.

## Answer
"""


def build_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """Render the RAG prompt with numbered context chunks."""
    numbered = "\n\n".join(
        f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(context_chunks)
    )
    return RAG_TEMPLATE.format(context=numbered, query=query)


# ── Vision-Language Template ───────────────────────────────────────────────────

VISION_LANGUAGE_TEMPLATE = """\
You are a multimodal AI assistant capable of analysing images.

## Image Description
{image_description}

## User Prompt
{user_prompt}

## Instructions
Provide a detailed, structured response that:
1. Directly addresses the user's prompt.
2. References specific visual elements from the image description.
3. Draws reasoned conclusions or insights.

## Response
"""


def build_vision_prompt(image_description: str, user_prompt: str) -> str:
    return VISION_LANGUAGE_TEMPLATE.format(
        image_description=image_description,
        user_prompt=user_prompt,
    )


# ── Task Planner Template ──────────────────────────────────────────────────────

PLANNER_TEMPLATE = """\
You are a strategic AI task planner. Break the following high-level task into
a numbered list of concrete, executable sub-steps.

Rules:
- Maximum {max_steps} steps.
- Each step must be actionable (start with a verb).
- Steps should be ordered logically.
- Indicate which tool each step uses: [VISION | MEMORY | SEARCH | LLM | NONE]

## Task
{task}

## Available Tools
- VISION  : Analyse an image and extract information
- MEMORY  : Retrieve semantically relevant knowledge from the vector store
- SEARCH  : Search the web for up-to-date information
- LLM     : Generate text, reason, or summarise

## Sub-Steps
"""


def build_planner_prompt(task: str, max_steps: int = 8) -> str:
    return PLANNER_TEMPLATE.format(task=task, max_steps=max_steps)


# ── Step Executor Template ─────────────────────────────────────────────────────

EXECUTOR_TEMPLATE = """\
You are an AI executor completing one step of a multi-step task.

## Overall Task
{overall_task}

## Current Step
{step}

## Previous Steps Summary
{previous_summary}

## Available Context
{context}

## Instructions
Complete the current step using the available context.
Be concise and precise. Provide a clear output for this step only.

## Step Output
"""


def build_executor_prompt(
    overall_task: str,
    step: str,
    previous_summary: str,
    context: str,
) -> str:
    return EXECUTOR_TEMPLATE.format(
        overall_task=overall_task,
        step=step,
        previous_summary=previous_summary,
        context=context,
    )


# ── Summarisation Template ─────────────────────────────────────────────────────

SUMMARISE_TEMPLATE = """\
Summarise the following agent execution output into a concise, user-friendly
response. Highlight key findings, insights, and any action items.

## Raw Agent Output
{raw_output}

## Concise Summary
"""


def build_summary_prompt(raw_output: str) -> str:
    return SUMMARISE_TEMPLATE.format(raw_output=raw_output)
