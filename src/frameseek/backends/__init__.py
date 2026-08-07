from __future__ import annotations

from .base import ResearchBackend


def create_backend(
    kind: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> ResearchBackend:
    normalized = kind.casefold()
    if normalized in {"openai", "openai-compatible"}:
        from .openai_compatible import OpenAICompatibleBackend

        if not model:
            raise ValueError("--model is required for the OpenAI-compatible backend")
        return OpenAICompatibleBackend(
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
            api_key_env=api_key_env,
        )
    if normalized in {"smolvlm2", "smolvlm"}:
        from .smolvlm2 import SmolVLM2Backend

        return SmolVLM2Backend(
            model=model or "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        )
    raise ValueError(f"unknown backend: {kind}")


__all__ = ["ResearchBackend", "create_backend"]
