"""
src/llm/llm_backend.py
───────────────────────
Configurable LLM backend factory.

Supported backends (set via configs/config.yaml → llm.backend):
  • "openai"        — OpenAI Chat Completions API
  • "huggingface"   — Local HuggingFace text-generation pipeline
  • "stub"          — Deterministic stub for testing / CI

Usage
─────
  from src.llm.llm_backend import LLMBackend
  llm = LLMBackend.from_config(cfg)
  response = llm.generate("Tell me about RAG.")
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ── Stub backend ───────────────────────────────────────────────────────────────

class StubLLM(BaseLLM):
    """
    Zero-dependency stub that returns deterministic canned responses.
    Perfect for offline development and CI testing.
    """

    def __init__(self, response_prefix: str = "[STUB LLM]") -> None:
        self.response_prefix = response_prefix

    def generate(self, prompt: str, **kwargs: Any) -> str:
        # Emulate a short processing delay for realism
        time.sleep(0.05)
        short_prompt = prompt[:80].replace("\n", " ").strip()
        return (
            f"{self.response_prefix} This is a simulated response to the prompt: "
            f"'{short_prompt}...'. "
            "In a production deployment, a real LLM (OpenAI GPT-4o, Mistral, LLaVA, etc.) "
            "would process this prompt and return a semantically meaningful answer. "
            "The system is correctly wired: retrieval, augmentation, and generation "
            "pipelines are all functional."
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return self.generate(last_user, **kwargs)


# ── OpenAI backend ─────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    """
    OpenAI Chat Completions backend.
    Requires: pip install openai
    Set OPENAI_API_KEY in environment or .env file.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("pip install openai") from e

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or "
                "switch llm.backend to 'stub' in configs/config.yaml."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content or ""


# ── HuggingFace backend ────────────────────────────────────────────────────────

class HuggingFaceLLM(BaseLLM):
    """
    Local HuggingFace text-generation pipeline.
    Supports CPU and CUDA inference.
    Requires: pip install transformers torch
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as e:
            raise ImportError("pip install transformers torch") from e

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._pipe = pipeline(
            "text-generation",
            model=model_id,
            device=0 if device == "cuda" else -1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        result = self._pipe(prompt, **kwargs)
        return result[0]["generated_text"][len(prompt):]  # strip echoed prompt

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        # Build a simple instruction-following prompt
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ) + "\nASSISTANT:"
        return self.generate(prompt, **kwargs)


# ── Factory ────────────────────────────────────────────────────────────────────

class LLMBackend:
    """
    Static factory that returns the correct BaseLLM subclass
    from the project config dict.
    """

    @staticmethod
    def from_config(cfg: dict) -> BaseLLM:
        llm_cfg = cfg.get("llm", {})
        backend = llm_cfg.get("backend", "stub").lower()

        if backend == "stub":
            stub_cfg = llm_cfg.get("stub", {})
            return StubLLM(
                response_prefix=stub_cfg.get("response_prefix", "[STUB LLM]")
            )

        elif backend == "openai":
            oa_cfg = llm_cfg.get("openai", {})
            api_key_env = oa_cfg.get("api_key_env", "OPENAI_API_KEY")
            return OpenAILLM(
                model=oa_cfg.get("model", "gpt-4o"),
                temperature=oa_cfg.get("temperature", 0.2),
                max_tokens=oa_cfg.get("max_tokens", 1024),
                api_key=os.environ.get(api_key_env, ""),
            )

        elif backend == "huggingface":
            hf_cfg = llm_cfg.get("huggingface", {})
            return HuggingFaceLLM(
                model_id=hf_cfg.get("model_id", "mistralai/Mistral-7B-Instruct-v0.2"),
                device=hf_cfg.get("device", "cpu"),
                max_new_tokens=hf_cfg.get("max_new_tokens", 512),
                temperature=hf_cfg.get("temperature", 0.3),
            )

        else:
            raise ValueError(
                f"Unknown LLM backend: '{backend}'. "
                "Choose from: 'stub', 'openai', 'huggingface'."
            )
