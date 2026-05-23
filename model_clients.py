"""
Model provider abstraction for Turing-Bench.

Subclass ``ModelClient`` and implement ``_generate`` to add a new provider.
Prompts come from ``config`` unchanged; normalization accepts JSON verdicts or plain A/B.

Supported hosted APIs:
- OpenAI Responses: https://platform.openai.com/docs/api-reference/responses/create
- OpenAI Chat Completions: https://developers.openai.com/api/reference/chat-completions/overview
- Anthropic Messages: https://docs.claude.com/en/api/messages
- AWS Bedrock (Anthropic SDK): https://github.com/anthropics/anthropic-sdk-python
- Moonshot/Kimi Chat Completions: https://platform.kimi.ai/docs/api/chat

To support a new model id string, add a substring entry to ``MODEL_SUBSTRING_TO_PROVIDER``.

Reasoning vs thinking: OpenAI uses ``reasoning={"effort": ...}`` on reasoning
models; newer Claude models use adaptive thinking via ``thinking.type`` and
``output_config.effort``. Claude 4.5 uses on/off thinking only in this runner.
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
    ("us.anthropic.",    "bedrock"),
    ("eu.anthropic.",    "bedrock"),
    ("apac.anthropic.",  "bedrock"),
    ("anthropic.",       "bedrock"),
    ("claude",           "anthropic"),
    ("sonnet",           "anthropic"),
    ("opus",             "anthropic"),
    ("haiku",            "anthropic"),
    ("chatgpt",          "openai_responses"),
    ("gpt-5",            "openai_responses"),
    ("o1",               "openai_responses"),
    ("o3",               "openai_responses"),
    ("o4",               "openai_responses"),
    ("o5",               "openai_responses"),
    ("gpt-4o",           "openai_legacy"),
    ("gpt-4.5",          "openai_legacy"),
    ("gpt-4.1",          "openai_legacy"),
    ("gpt-4-turbo",      "openai_legacy"),
    ("gpt-4",            "openai_legacy"),
    ("gpt-3.5",          "openai_legacy"),
    ("gpt",              "openai_legacy"),
    ("kimi",             "moonshot"),
]

OPENAI_REASONING_EFFORTS = ("low", "medium", "high")
CLAUDE_REASONING_EFFORTS = ("on", "off", "low", "medium", "high", "max", "xhigh")
CLAUDE_ON_OFF_EFFORTS = ("on", "off")
CLAUDE_ON_THINKING_BUDGET = 1024
KIMI_REASONING_EFFORTS = ("enabled", "disabled")



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
        print("normalize_prediction | No model response recieved; returning None.")
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
    print("normalize_prediction | No model response recieved; returning None.")
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
    reasoning_effort: mapped to OpenAI reasoning effort and Claude thinking effort.
    """

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str | None = None,
        max_tokens: int = 3072,
        temperature: float = 1.0,
        max_retries: int = MAX_RETRIES,
        base_delay: float = BASE_DELAY,
    ) -> None:
        self.model = model
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _normalize_reasoning_effort(self, reasoning_effort: str | None) -> str | None:
        if reasoning_effort is None:
            return None
        effort = reasoning_effort.strip().lower()
        if not effort:
            return None
        return effort

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
            if self.reasoning_effort and self.reasoning_effort not in OPENAI_REASONING_EFFORTS:
                raise ValueError(
                    f"OpenAI reasoning models accept --reasoning-effort values: "
                    f"{', '.join(OPENAI_REASONING_EFFORTS)}. "
                    f"Got {self.reasoning_effort!r}."
                )
            reasoning = (
                {"effort": self.reasoning_effort}
                if self.reasoning_effort
                else {"effort": "low"}
            )
            response = client.responses.create(
                model=self.model,
                input=inp,
                reasoning=reasoning,
                max_output_tokens=self.max_tokens,
                text={**text_fmt, "verbosity": "low"},
            )
        else:
            if self.reasoning_effort:
                raise ValueError(
                    f"OpenAI model {self.model!r} does not support --reasoning-effort. "
                    "Only GPT-5 and o-series models use OpenAI reasoning.effort. "
                    "Remove --reasoning-effort or choose a reasoning-capable model."
                )
            response = client.responses.create(
                model=self.model,
                input=inp,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=1,
                text=text_fmt,
            )
        return (getattr(response, "output_text", None) or "").strip()

# --- Legacy OpenAI ------------------------------------------------------------

JSON_MODE_MIN_DATE = 1106  # inclusive lower bound (MMDD integer)

# Model name fragments that always support JSON mode regardless of date stamp.
_JSON_MODE_ALWAYS = ("gpt-4-turbo", "gpt-4o", "gpt-4.1", "gpt-4.5")


class LegacyOpenAIModelClient(ModelClient):

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

    def _supports_json_mode(self) -> bool:
        n = self.model.lower()
        for fragment in _JSON_MODE_ALWAYS:
            if fragment in n:
                return True
        import re
        m = re.search(r"-(\d{4})(?:-|$)", n)
        if m:
            return int(m.group(1)) >= JSON_MODE_MIN_DATE
        return False

    def _generate(self, messages: list[dict[str, str]]) -> str:
        if self.reasoning_effort is not None:
            raise ValueError(
                f"LegacyOpenAIModelClient does not support --reasoning-effort. "
                "The Chat Completions API has no reasoning parameter. "
                "Remove --reasoning-effort or switch to a Responses-API model."
            )

        client = self._ensure_client()

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,          # role/content dicts pass through as-is
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1,
        )
        if self._supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()
    

# --- Anthropic ----------------------------------------------------------------


class AnthropicModelClient(ModelClient):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._client = None

    def _claude_uses_on_off_thinking(self) -> bool:
        return "4-5" in self.model.lower()

    def _apply_claude_thinking(self, kw: dict[str, Any]) -> None:
        if not self.reasoning_effort:
            return

        if self.reasoning_effort not in CLAUDE_REASONING_EFFORTS:
            raise ValueError(
                "Claude accepts --reasoning-effort values: "
                f"{', '.join(CLAUDE_REASONING_EFFORTS)}. "
                f"Got {self.reasoning_effort!r}. "
                "These are passed through using Anthropic's adaptive thinking API; "
                "choose a value supported by your Claude model."
            )

        kw.pop("temperature", None)

        if self._claude_uses_on_off_thinking():
            if self.reasoning_effort not in CLAUDE_ON_OFF_EFFORTS:
                raise ValueError(
                    f"Claude 4.5 models accept --reasoning-effort values: "
                    f"{', '.join(CLAUDE_ON_OFF_EFFORTS)}. "
                    f"Got {self.reasoning_effort!r}. "
                    "Use Claude 4.6+ / Opus 4.7-style models for adaptive "
                    "effort values such as low, medium, high, max, or xhigh."
                )
            if self.reasoning_effort == "off":
                kw["thinking"] = {"type": "disabled"}
                return
            kw["max_tokens"] = max(
                int(kw["max_tokens"]),
                CLAUDE_ON_THINKING_BUDGET + 1024,
            )
            kw["thinking"] = {
                "type": "enabled",
                "budget_tokens": CLAUDE_ON_THINKING_BUDGET,
            }
            return

        if self.reasoning_effort == "off":
            kw["thinking"] = {"type": "disabled"}
            return

        kw["thinking"] = {"type": "adaptive"}
        if self.reasoning_effort != "on":
            kw["output_config"] = {"effort": self.reasoning_effort}

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

        if self.reasoning_effort:
            self._apply_claude_thinking(kw)
        else:
            kw["thinking"] = {"type": "disabled"}

        msg = client.messages.create(**kw)
        text_parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_parts.append(block.text)
        out = "".join(text_parts).strip()
        return out if out else str(msg.content[0])
    

# --- Kimi ---------------------------------------------------------------------

class MoonshotModelClient(ModelClient):
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

            key = self._api_key or os.environ.get("MOONSHOT_API_KEY")
            if not key:
                key = getpass.getpass("Enter Moonshot API key: ")
                os.environ["MOONSHOT_API_KEY"] = key
            else:
                src = "constructor arg" if self._api_key else "MOONSHOT_API_KEY env var"

            self._client = OpenAI(api_key=key, base_url="https://api.moonshot.ai/v1")
        return self._client

    def _use_thinking(self) -> bool:
        result = self.reasoning_effort != "disabled"
        return result

    def _generate(self, messages: list[dict[str, str]]) -> str:
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            preview = content[:200] + ("..." if len(content) > 200 else "")

        if self.reasoning_effort and self.reasoning_effort not in KIMI_REASONING_EFFORTS:
            err = (
                f"Kimi models accept --reasoning-effort values: "
                f"{', '.join(KIMI_REASONING_EFFORTS)}. "
                f"Got {self.reasoning_effort!r}."
            )
            raise ValueError(err)

        client = self._ensure_client()
        thinking = self._use_thinking()
        temperature = 1.0 if thinking else 0.6
        extra_body = {} if thinking else {"thinking": {"type": "disabled"}}

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                extra_body=extra_body,
            )
        except Exception as e:
            raise

        usage = getattr(response, "usage", None)
        if usage:
            print(f"[DEBUG]   usage: prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens} total_tokens={usage.total_tokens}")

        choices = getattr(response, "choices", [])
        for i, choice in enumerate(choices):
            finish = getattr(choice, "finish_reason", "?")
            raw_content = getattr(choice.message, "content", None)
            print(f"[DEBUG]   choices[{i}] finish_reason={finish!r}")
            print(f"[DEBUG]   choices[{i}] raw content ({len(raw_content or '')} chars):")
            print(raw_content)

        result = (response.choices[0].message.content or "").strip()
        return result
    


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

    def _claude_uses_on_off_thinking(self) -> bool:
        m = self.model.lower()
        return not ("4-6" in m or "4-7" in m)

    def _apply_claude_thinking(self, kw: dict[str, Any]) -> None:
        if not self.reasoning_effort:
            return

        if self.reasoning_effort not in CLAUDE_REASONING_EFFORTS:
            raise ValueError(
                "Claude Bedrock accepts --reasoning-effort values: "
                f"{', '.join(CLAUDE_REASONING_EFFORTS)}. "
                f"Got {self.reasoning_effort!r}. "
                "These are passed through using Anthropic's adaptive thinking API; "
                "choose a value supported by your Claude model."
            )

        kw.pop("temperature", None)

        if self._claude_uses_on_off_thinking():
            if self.reasoning_effort not in CLAUDE_ON_OFF_EFFORTS:
                raise ValueError(
                    f"Claude 4.5 Bedrock models accept --reasoning-effort values: "
                    f"{', '.join(CLAUDE_ON_OFF_EFFORTS)}. "
                    f"Got {self.reasoning_effort!r}. "
                    "Use Claude 4.6+ / Opus 4.7-style models for adaptive "
                    "effort values such as low, medium, high, max, or xhigh."
                )
            if self.reasoning_effort == "off":
                kw["thinking"] = {"type": "disabled"}
                return
            kw["max_tokens"] = max(
                int(kw["max_tokens"]),
                CLAUDE_ON_THINKING_BUDGET + 1024,
            )
            kw["thinking"] = {
                "type": "enabled",
                "budget_tokens": CLAUDE_ON_THINKING_BUDGET,
            }
            return

        if self.reasoning_effort == "off":
            kw["thinking"] = {"type": "disabled"}
            return

        kw["thinking"] = {"type": "adaptive"}
        if self.reasoning_effort != "on":
            kw["output_config"] = {"effort": self.reasoning_effort}

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

        if self.reasoning_effort:
            self._apply_claude_thinking(kw)
        else:
            kw["temperature"] = self.temperature
            kw["thinking"] = {"type": "disabled"}

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
    reasoning_effort: str | None = None,
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

    if reasoning_effort:
        effort = reasoning_effort.strip().lower()
        if prov in ("anthropic", "bedrock") and effort not in CLAUDE_REASONING_EFFORTS:
            raise ValueError(
                f"Claude accepts --reasoning-effort values: "
                f"{', '.join(CLAUDE_REASONING_EFFORTS)}. Got {reasoning_effort!r}. "
                "Pass the exact Anthropic value supported by your selected Claude model."
            )
        if prov == "openai":
            openai_client = OpenAIModelClient(model, reasoning_effort=reasoning_effort)
            if not openai_client._openai_uses_reasoning_parameter():
                raise ValueError(
                    f"OpenAI model {model!r} does not support --reasoning-effort. "
                    "Only GPT-5 and o-series models use OpenAI reasoning.effort. "
                    "Remove --reasoning-effort or choose a reasoning-capable model."
                )
            if (
                openai_client._openai_uses_reasoning_parameter()
                and effort not in OPENAI_REASONING_EFFORTS
            ):
                raise ValueError(
                    f"OpenAI reasoning models accept --reasoning-effort values: "
                    f"{', '.join(OPENAI_REASONING_EFFORTS)}. Got {reasoning_effort!r}. "
                    "GPT-4 and lower ignore this flag."
                )

    if prov == "openai":
        return OpenAIModelClient(model, reasoning_effort=reasoning_effort)
    if prov == "openai_legacy" : 
        return LegacyOpenAIModelClient(model)
    if prov == "anthropic":
        return AnthropicModelClient(model, reasoning_effort=reasoning_effort)
    if prov == "bedrock":
        return BedrockAnthropicClient(
            model,
            aws_region=aws_region,
            reasoning_effort=reasoning_effort,
        )
    if prov == "moonshot":
        return MoonshotModelClient(model, reasoning_effort=reasoning_effort)
    raise ValueError(f"Unknown provider {provider!r}.")
