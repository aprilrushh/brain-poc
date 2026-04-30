"""
LLM client with LLM_MODE swap (api / local).
"""

from __future__ import annotations
import os
from openai import OpenAI


PLACEHOLDER_VALUES = {
    "",
    "your_together_api_key_here",
    "your_openrouter_api_key_here",
    "your_api_key_here",
    "tgp_v1_xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
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


def get_llm_client() -> OpenAI:
    """
    Returns OpenAI-compatible client based on env vars.
    Mode 'api' uses LLM_API_BASE + LLM_API_KEY (or TOGETHER_API_KEY fallback).
    Mode 'local' assumes vLLM serving on localhost:8000.
    """
    mode = os.environ.get("LLM_MODE", "api").lower()

    if mode == "local":
        return OpenAI(
            base_url=os.environ.get("LLM_API_BASE", "http://localhost:8000/v1"),
            api_key="local-no-auth",
        )

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

    return OpenAI(base_url=base_url, api_key=api_key)


def get_llm_model() -> str:
    """Resolve model name from env or default."""
    return os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-397B-A17B")


def is_openrouter() -> bool:
    return "openrouter" in os.environ.get("LLM_API_BASE", "").lower()
