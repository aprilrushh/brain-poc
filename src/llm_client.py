"""
LLM client with multi-provider adapter (together / anthropic).
Supports OpenAI-compatible (Together AI, OpenRouter) and Anthropic native.
"""

from __future__ import annotations
import os
from typing import Any


PLACEHOLDER_VALUES = {
    "",
    "your_together_api_key_here",
    "your_openrouter_api_key_here",
    "your_anthropic_api_key_here",
    "your_api_key_here",
    "tgp_v1_xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
}


def _clean(val: str | None) -> str:
    """Return val if it looks like a real key, else empty string."""
    if val is None:
        return ""
    val = val.strip()
    if val in PLACEHOLDER_VALUES:
        return ""
    if val.startswith("your_") and val.endswith("_here"):
        return ""
    return val


# ============================================================
# Adapter pattern — unified chat_complete interface
# ============================================================

class LLMAdapter:
    """Base adapter. Subclasses must implement chat_complete()."""
    def chat_complete(self, messages: list[dict], **kwargs) -> dict:
        raise NotImplementedError


class OpenAIAdapter(LLMAdapter):
    """For Together AI, OpenRouter, vLLM local — anything OpenAI-compatible."""
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def chat_complete(self, messages: list[dict], **kwargs) -> dict:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        Returns: {"text": str, "input_tokens": int, "output_tokens": int,
                  "stop_reason": str, "model": str}
        """
        max_tokens = kwargs.pop("max_tokens", 2048)
        temperature = kwargs.pop("temperature", 0.3)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return {
            "text": resp.choices[0].message.content or "",
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "stop_reason": resp.choices[0].finish_reason,
            "model": resp.model,
        }


class AnthropicAdapter(LLMAdapter):
    """For Anthropic Claude (Opus, Sonnet, Haiku)."""
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def chat_complete(self, messages: list[dict], **kwargs) -> dict:
        """
        Same input/output format as OpenAIAdapter.
        Anthropic API differences handled internally:
          - 'system' messages extracted as separate parameter
          - max_tokens is required
        """
        max_tokens = kwargs.pop("max_tokens", 2048)
        temperature = kwargs.pop("temperature", None)

        # Anthropic: system messages must be separate from messages list
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        system_text = "\n\n".join(system_parts) if system_parts else None

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": non_system,
        }
        # Opus 4.7+ deprecated temperature; only include if explicitly given
        # AND not using a model that rejects it
        DEPRECATED_TEMP_MODELS = ("claude-opus-4-7",)
        if temperature is not None and not any(
            m in self.model for m in DEPRECATED_TEMP_MODELS
        ):
            create_kwargs["temperature"] = temperature
        if system_text:
            create_kwargs["system"] = system_text

        resp = self.client.messages.create(**create_kwargs, **kwargs)

        # Extract text from content blocks
        text_parts = [b.text for b in resp.content if hasattr(b, "text")]
        text = "".join(text_parts)

        return {
            "text": text,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "stop_reason": resp.stop_reason,
            "model": resp.model,
        }


# ============================================================
# Factory — provider-aware client construction
# ============================================================

def get_llm_provider() -> str:
    """Returns 'together' (default) or 'anthropic'."""
    return os.environ.get("LLM_PROVIDER", "together").lower()


def get_llm_client() -> LLMAdapter:
    """
    Returns LLMAdapter based on LLM_PROVIDER env var.
    Default: 'together' (OpenAI-compatible, backward compat).
    """
    provider = get_llm_provider()
    model = get_llm_model()

    if provider == "anthropic":
        from anthropic import Anthropic
        api_key = _clean(os.environ.get("ANTHROPIC_API_KEY"))
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set or placeholder. "
                "Set in .env to use LLM_PROVIDER=anthropic."
            )
        client = Anthropic(api_key=api_key)
        return AnthropicAdapter(client, model)

    # Default: OpenAI-compatible (together / openrouter / local vLLM)
    from openai import OpenAI

    mode = os.environ.get("LLM_MODE", "api").lower()

    if mode == "local":
        client = OpenAI(
            base_url=os.environ.get("LLM_API_BASE", "http://localhost:8000/v1"),
            api_key="local-no-auth",
        )
        return OpenAIAdapter(client, model)

    api_key = (
        _clean(os.environ.get("LLM_API_KEY"))
        or _clean(os.environ.get("TOGETHER_API_KEY"))
        or _clean(os.environ.get("OPENROUTER_API_KEY"))
    )
    if not api_key:
        raise RuntimeError(
            "No valid API key found. Set TOGETHER_API_KEY (or LLM_API_KEY) "
            "in .env. Placeholder values are ignored."
        )

    base_url = os.environ.get("LLM_API_BASE", "https://api.together.xyz/v1")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return OpenAIAdapter(client, model)


def get_llm_model() -> str:
    """Resolve model name from env or default based on provider."""
    explicit = os.environ.get("LLM_MODEL")
    if explicit:
        return explicit
    # Provider-aware default
    if get_llm_provider() == "anthropic":
        return "claude-opus-4-7"
    return "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"


def is_openrouter() -> bool:
    return "openrouter" in os.environ.get("LLM_API_BASE", "").lower()
