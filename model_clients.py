"""
Model provider abstraction for Turing-Bench.

Subclass ``ModelClient`` and implement ``_generate`` to add a new provider.
Prompts come from ``config`` unchanged; normalization accepts JSON verdicts or plain A/B.

Supported hosted APIs:
- OpenAI Responses: https://platform.openai.com/docs/api-reference/responses/create
- Anthropic Messages: https://docs.claude.com/en/api/messages
- AWS Bedrock (Anthropic SDK): https://github.com/anthropics/anthropic-sdk-python

To support a new model id string, add a substring entry to ``MODEL_SUBSTRING_TO_PROVIDER``.

Reasoning vs thinking: OpenAI uses ``reasoning={"effort": ...}`` on reasoning models;
Anthropic/Bedrock use ``thinking`` with ``type: "enabled"`` and ``budget_tokens`` (>= 1024).
"""
from __future__ import annotations

import abc
import json
import os
import time
from typing import Any

from config import BASE_DELAY, MAX_RETRIES, SYSTEM_PROMPT, USER_TEMPLATE

# Order matters: first match wins (list longer / more specific substrings first).
MODEL_SUBSTRING_TO_PROVIDER: list[tuple[str, str]] = [
    ("us.anthropic.", "bedrock"),
    ("eu.anthropic.", "bedrock"),
    ("apac.anthropic.", "bedrock"),
    ("anthropic.", "bedrock"),
    ("chatgpt", "openai"),
    ("claude", "anthropic"),
    ("sonnet", "anthropic"),
    ("opus", "anthropic"),
    ("haiku", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("o5", "openai"),
]


def infer_provider_from_model(model: str) -> str:
    """Return ``openai``, ``anthropic``, or ``bedrock`` from model id substrings."""
    m = model.strip().lower()
    if not m:
        raise ValueError("Model id must be non-empty.")
    for token, provider in MODEL_SUBSTRING_TO_PROVIDER:
        if token in m:
            return provider
    raise ValueError(
        f"Could not infer provider from model {model!r}. "
        "Add a matching substring to MODEL_SUBSTRING_TO_PROVIDER in model_clients.py, "
        "pass --provider openai|anthropic|bedrock explicitly, "
        "or use --provider custom with predict.py."
    )


def normalize_prediction(raw: Any) -> str | None:
    """Return 'A', 'B', or None."""
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip().upper()
        if s in ("A", "B"):
            return s
        return _verdict_from_json_string(raw)

    if isinstance(raw, dict):
        result = raw.get("result")
        verdict = None
        if isinstance(result, dict):
            verdict = result.get("verdict")
        if verdict is None:
            verdict = raw.get("verdict")
        if isinstance(verdict, str):
            s = verdict.strip().upper()
            if s in ("A", "B"):
                return s
    return None


def _verdict_from_json_string(text: str) -> str | None:
    t = text.strip()
    try:
        if t.startswith("{") and "}" in t:
            parsed = json.loads(t)
            return normalize_prediction(parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    for ch in t.upper():
        if ch in ("A", "B"):
            return ch
    return None


class ModelClient(abc.ABC):
    """
    Stateless per-row callers: builds system+user from config and parses model output.
    reasoning_effort: mapped for OpenAI Responses on reasoning models.
    thinking_budget: Anthropic/Bedrock extended thinking (>0 enables; API requires >= 1024).
    """

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str = "low",
        thinking_budget: int = 0,
        max_tokens: int = 3072,
        temperature: float = 1.0,
        max_retries: int = MAX_RETRIES,
        base_delay: float = BASE_DELAY,
    ) -> None:
        self.model = model
        self.reasoning_effort = (reasoning_effort or "low").strip().lower()
        self.thinking_budget = int(thinking_budget)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay

    def build_messages(self, dialogue_a: str, dialogue_b: str) -> list[dict[str, str]]:
        user_text = USER_TEMPLATE.format(dialogueA=dialogue_a, dialogueB=dialogue_b)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def predict(self, dialogue_a: str, dialogue_b: str) -> str:
        messages = self.build_messages(dialogue_a, dialogue_b)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw_text = self._generate(messages)
                label = normalize_prediction(raw_text.strip() if isinstance(raw_text, str) else raw_text)
                if label in ("A", "B"):
                    return label
                raise ValueError(
                    f"No valid verdict in model output: {raw_text!r}"
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = self.base_delay * (2**attempt)
                    time.sleep(wait)
                else:
                    break
        assert last_exc is not None
        raise last_exc

    @abc.abstractmethod
    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Execute one model call and return raw string output (possibly JSON text)."""


# --- OpenAI (Responses API) ---------------------------------------------------


class OpenAIModelClient(ModelClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self._api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import getpass
            from openai import OpenAI

            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                key = getpass.getpass("Enter OpenAI API key: ")
                os.environ["OPENAI_API_KEY"] = key
            self._client = OpenAI(api_key=key)
        return self._client

    def _to_response_input(self, messages: list[dict[str, str]]) -> list[dict]:
        out = []
        for m in messages:
            out.append(
                {
                    "role": m["role"],
                    "content": [{"type": "input_text", "text": m["content"]}],
                }
            )
        return out

    def _openai_uses_reasoning_parameter(self) -> bool:
        """``reasoning`` is for GPT-5 family and o-series per OpenAI Responses docs."""
        n = self.model.lower()
        if "gpt-5" in n:
            return True
        for prefix in ("o1", "o3", "o4", "o5"):
            if n == prefix or n.startswith(f"{prefix}-") or n.startswith(f"{prefix}."):
                return True
        return False

    def _generate(self, messages: list[dict[str, str]]) -> str:
        client = self._ensure_client()
        inp = self._to_response_input(messages)
        text_fmt = {"format": {"type": "json_object"}}

        if self._openai_uses_reasoning_parameter():
            response = client.responses.create(
                model=self.model,
                input=inp,
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=self.max_tokens,
                text={**text_fmt, "verbosity": "low"},
            )
        else:
            response = client.responses.create(
                model=self.model,
                input=inp,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=1,
                text=text_fmt,
            )
        return (getattr(response, "output_text", None) or "").strip()


# --- Anthropic ----------------------------------------------------------------


class AnthropicModelClient(ModelClient):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        import getpass

        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            key = getpass.getpass("Enter Anthropic API key: ")
            os.environ["ANTHROPIC_API_KEY"] = key
        self._client = anthropic.Anthropic()
        return self._client

    def _generate(self, messages: list[dict[str, str]]) -> str:
        client = self._ensure_client()
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        anth_msgs = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = m["role"] if m["role"] in ("user", "assistant") else "user"
            anth_msgs.append({"role": role, "content": m["content"]})

        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": anth_msgs,
            "temperature": self.temperature,
        }

        if self.thinking_budget > 0:
            if self.thinking_budget < 1024:
                raise ValueError(
                    "Anthropic extended thinking requires thinking_budget >= 1024 "
                    "(see Claude Messages API / extended thinking docs)."
                )
            if self.max_tokens <= self.thinking_budget:
                raise ValueError(
                    "When extended thinking is on, max_tokens must be greater than "
                    "thinking_budget (API requires budget_tokens < max_tokens)."
                )
            kw.pop("temperature", None)
            kw["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        msg = client.messages.create(**kw)
        text_parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_parts.append(block.text)
        out = "".join(text_parts).strip()
        return out if out else str(msg.content[0])


# --- Bedrock ------------------------------------------------------------------


class BedrockAnthropicClient(ModelClient):
    """Anthropic-compatible models on AWS Bedrock.

    Set ``AWS_REGION`` (or ``--aws-region``) and authenticate with a Bedrock API key in
    ``AWS_BEARER_TOKEN_BEDROCK`` (read automatically by boto3) or another boto3 credential
    source you already use.
    """

    def __init__(self, model: str, *, aws_region: str | None = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._aws_region = aws_region or os.environ.get("AWS_REGION")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from anthropic import AnthropicBedrock

        if self._aws_region:
            self._client = AnthropicBedrock(aws_region=self._aws_region)
        else:
            self._client = AnthropicBedrock()
        return self._client

    def _generate(self, messages: list[dict[str, str]]) -> str:
        client = self._ensure_client()
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        anth_msgs = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = m["role"] if m["role"] in ("user", "assistant") else "user"
            anth_msgs.append({"role": role, "content": m["content"]})

        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": anth_msgs,
        }

        if self.thinking_budget >= 1024:
            if self.max_tokens <= self.thinking_budget:
                raise ValueError(
                    "When Bedrock extended thinking is on, max_tokens must be greater "
                    "than thinking_budget."
                )
            kw["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        else:
            kw["temperature"] = self.temperature

        msg = client.messages.create(**kw)
        text_parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_parts.append(block.text)
        out = "".join(text_parts).strip()
        return out if out else str(msg.content[0])


# --- Delegates to user predict.py -----------------------------------------------


class CustomPythonModelClient(ModelClient):
    """Calls ``predict.predict(dialogue_a, dialogue_b)`` from ``predict.py``."""

    def __init__(self) -> None:
        super().__init__("custom")

    def _generate(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("_generate unused for CustomPythonModelClient")

    def predict(self, dialogue_a: str, dialogue_b: str) -> str:
        import predict as user_predict_module

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = user_predict_module.predict(dialogue_a, dialogue_b)
                label = normalize_prediction(raw)
                if label in ("A", "B"):
                    return label
                raise ValueError(f"predict() returned invalid verdict: {raw!r}")
            except NotImplementedError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (2**attempt))
        assert last_exc is not None
        raise last_exc


def create_model_client(
    provider: str,
    model: str | None,
    *,
    aws_region: str | None = None,
) -> ModelClient:
    """
    Build a ``ModelClient``.

    ``provider`` is ``custom``, ``openai``, ``anthropic``, ``bedrock``, or ``auto``.
    For ``auto``, the provider is inferred from ``model`` via ``infer_provider_from_model``.
    """
    prov = provider.strip().lower()
    if prov == "custom":
        return CustomPythonModelClient()

    if not model or not str(model).strip():
        raise ValueError("--model / -m is required unless --provider custom")

    model = str(model).strip()
    if prov == "auto":
        prov = infer_provider_from_model(model)
    elif prov not in ("openai", "anthropic", "bedrock"):
        raise ValueError(
            f"Unknown provider {provider!r}. "
            "Use: auto, openai, anthropic, bedrock, custom."
        )

    if prov == "openai":
        return OpenAIModelClient(model)
    if prov == "anthropic":
        return AnthropicModelClient(model)
    if prov == "bedrock":
        return BedrockAnthropicClient(model, aws_region=aws_region)
    raise ValueError(f"Unknown provider {provider!r}.")
