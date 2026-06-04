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
    """For Together AI, OpenRouter, vLLM local, OpenAI — anything OpenAI-compatible."""
    def _is_gpt(self):
        return "gpt" in (self.model or "").lower()
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        # Qwen3.5/3.6/3.7 are reasoning models; disable thinking for grounded
        # 0%-hallucination task. Qwen3-235B does not match -> unchanged.
        if any(t in self.model for t in ("Qwen3.5", "Qwen3.6", "Qwen3.7")):
            self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            self._extra_body = {}

    def chat_complete(self, messages: list[dict], **kwargs) -> dict:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        Returns: {"text": str, "input_tokens": int, "output_tokens": int,
                  "stop_reason": str, "model": str}
        """
        max_tokens = kwargs.pop("max_tokens", 2048)
        temperature = kwargs.pop("temperature", 0.3)
        if self._extra_body and "extra_body" not in kwargs:
            kwargs["extra_body"] = self._extra_body

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

    def chat_complete_stream(self, messages: list[dict], **kwargs):
        """Streaming version — yields text chunks as they arrive, then a final
        dict with usage/stop_reason. Matches OpenAI streaming protocol.

        Yields:
            ('chunk', str)   — delta text as it arrives
            ('done',  dict)  — final {text, input_tokens, output_tokens,
                               stop_reason, model} (text = full concatenated)

        Usage tokens may be None for some providers when stream=True; caller
        handles None gracefully (cost log records what's available).
        """
        max_tokens = kwargs.pop("max_tokens", 2048)
        temperature = kwargs.pop("temperature", 0.3)
        if self._extra_body and "extra_body" not in kwargs:
            kwargs["extra_body"] = self._extra_body

        if self._is_gpt():
            # GPT-5.x reasoning model: max_completion_tokens (not max_tokens),
            # no custom temperature, reasoning_effort (default medium, env overrides).
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                reasoning_effort=os.environ.get("GENERAL_REASONING_EFFORT", "medium"),
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
        else:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )

        full_text = []
        usage = None
        stop_reason = None
        model_used = self.model

        for ev in stream:
            # usage event (final, after [DONE])
            if getattr(ev, "usage", None) is not None:
                usage = ev.usage
            if not ev.choices:
                continue
            choice = ev.choices[0]
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta else None
            if content:
                full_text.append(content)
                yield ("chunk", content)
            if getattr(choice, "finish_reason", None):
                stop_reason = choice.finish_reason
            if getattr(ev, "model", None):
                model_used = ev.model

        yield ("done", {
            "text": "".join(full_text),
            "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "stop_reason": stop_reason,
            "model": model_used,
        })


    def responses_stream(self, messages: list[dict], max_tokens: int = 16000,
                         reasoning_effort: str = "high", reasoning_summary: str = "detailed",
                         verbosity: str = None):
        """GPT reasoning 모델 전용 (Responses API). reasoning summary + 본문을 한 stream 에서 받음.
        yields: ("reasoning", text) | ("chunk", text) | ("done", {text, model, ...})
        messages([{system},{user/assistant}...]) -> Responses API instructions + input 변환."""
        # system -> instructions, 나머지 -> input 메시지 리스트
        instructions = ""
        input_items = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                instructions = (instructions + "\n\n" + content) if instructions else content
            else:
                input_items.append({"role": role, "content": content})
        full_text = []
        reasoning_text = []
        # effort=none 은 reasoning 출력이 없으므로 summary 필드 제거 (API 400 회피)
        if reasoning_effort == "none":
            _reasoning = {"effort": "none"}
        else:
            _reasoning = {"effort": reasoning_effort, "summary": reasoning_summary}
        _create_kw = dict(
            model=self.model,
            instructions=instructions or None,
            input=input_items,
            max_output_tokens=max_tokens,
            reasoning=_reasoning,
            stream=True,
        )
        if verbosity:
            _create_kw["text"] = {"verbosity": verbosity}
        stream = self.client.responses.create(**_create_kw)
        for ev in stream:
            t = getattr(ev, "type", "")
            if t == "response.reasoning_summary_text.delta":
                d = getattr(ev, "delta", None)
                if d:
                    reasoning_text.append(d)
                    yield ("reasoning", d)
            elif t == "response.output_text.delta":
                d = getattr(ev, "delta", None)
                if d:
                    full_text.append(d)
                    yield ("chunk", d)
            elif t == "response.completed":
                resp = getattr(ev, "response", None)
                usage = getattr(resp, "usage", None) if resp else None
                yield ("done", {
                    "text": "".join(full_text),
                    "reasoning": "".join(reasoning_text),
                    "model": self.model,
                    "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                    "stop_reason": "stop",
                })


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


def get_general_llm_client():
    """General(Explore) 경로 전용 LLM client. GENERAL_LLM_PROVIDER 설정 시에만.
    미설정이면 None -> 호출부가 기존 adapter 로 fallback (하위호환)."""
    prov = _clean(os.environ.get("GENERAL_LLM_PROVIDER", "")).lower()
    if prov != "openai":
        return None
    from openai import OpenAI
    api_key = _clean(os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        return None
    model = _clean(os.environ.get("GENERAL_LLM_MODEL")) or "gpt-5.5"
    client = OpenAI(api_key=api_key)  # base_url 기본 = OpenAI
    return OpenAIAdapter(client, model)


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
