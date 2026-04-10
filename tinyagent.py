"""
TinyAgent: small wrapper around chat models.

- `provider` (constructor) or `TINYAGENT_PROVIDER` env: `openai`, `anthropic`,
  `google`, `ollama`, `hf` (local Hugging Face `transformers`).
- If omitted, provider is inferred from the model name when possible.
- Keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` or `GEMINI_API_KEY`.
"""
import os
import json
import getpass
from dotenv import load_dotenv

load_dotenv()


_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API key: ")
        from openai import OpenAI

        _openai_client = OpenAI()
    return _openai_client


# @title
class TinyAgent:
    def __init__(self, model, tokenizer=None, debug=False, provider=None):
        self.model = model
        self.tokenizer = tokenizer
        self.provider = provider
        self.messages = []
        self.max_tokens = 3072
        self.debug = debug
        self.reasoning_effort = "low"
        self.temperature = 1
        self._hf_pipeline = None
        self._hf_tokenizer = None
        self.thinking_budget = 0

    def clear_messages(self):
        self.messages = list()

    def add_message(self, message_type, message):
        content_type = "input_text"
        if message_type == "assistant":
            content_type = "output_text"
        self.messages.append(
            {"role": message_type, "content": [{"type": content_type, "text": message}]}
        )

    def add_system_message(self, message):
        self.add_message("system", message)

    def add_user_message(self, message):
        self.add_message("user", message)

    def add_assistant_message(self, message):
        self.add_message("assistant", message)

    def add_instruction(self, message):
        self.add_message("system", "Instruction: " + message)

    def add_data(self, message):
        self.add_message("user", "Data: " + message)

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def set_debug(self, debug):
        self.debug = debug

    def set_reasoning_effort(self, reasoning_effort="low", budget_tokens=0):
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = budget_tokens

    def call(self, prompt="", response_type="text", cache=True):
        reply = None

        # --- Resolve provider (explicit > env > model name heuristics) ---
        env_prov = (os.environ.get("TINYAGENT_PROVIDER") or "").strip().lower()
        explicit = (self.provider or "").strip().lower() if self.provider else ""
        if explicit:
            provider = explicit
        elif env_prov:
            provider = env_prov
        else:
            lm = self.model.lower()
            if "claude" in lm:
                provider = "anthropic"
            elif "gemini" in lm:
                provider = "google"
            elif "/" in self.model and not lm.startswith("gpt"):
                provider = "hf"
            elif any(
                x in self.model
                for x in ("gpt-", "gpt-4", "gpt-5", "o1", "o3", "o4")
            ):
                provider = "openai"
            else:
                provider = "openai"

        _aliases = {"huggingface": "hf", "transformers": "hf", "gemini": "google"}
        provider = _aliases.get(provider, provider)

        # --- OpenAI Responses API input (structured content blocks) ---
        openai_messages = self.messages.copy()
        if prompt:
            openai_messages.append(
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
            )

        # --- Flat chat messages for Anthropic / Ollama / HF / ad-hoc prompts ---
        flat = []
        for m in self.messages:
            role = m["role"]
            c = m["content"]
            if isinstance(c, str):
                text = c
            else:
                text = ""
                for part in c:
                    if isinstance(part, dict):
                        text += part.get("text", "")
                    elif isinstance(part, str):
                        text += part
            flat.append({"role": role, "content": text})
        if prompt:
            flat.append({"role": "user", "content": prompt})

        if cache:
            self.add_user_message(prompt)

        # --- Provider calls (all request logic stays in call) ---
        if provider == "openai":
            client = _get_openai_client()
            if "gpt-5" in self.model:
                response = client.responses.create(
                    model=self.model,
                    input=openai_messages,
                    reasoning={"effort": self.reasoning_effort},
                    text={
                        "format": {"type": response_type},
                        "verbosity": "low",
                    },
                )
                reply = response.output_text
            elif (
                "gpt-4" in self.model
                or "o3" in self.model
                or "o4" in self.model
                or "o1" in self.model
                or "gpt-" in self.model
            ):
                response = client.responses.create(
                    model=self.model,
                    input=openai_messages,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=1,
                    text={"format": {"type": response_type}},
                )
                reply = response.output_text
            else:
                response = client.responses.create(
                    model=self.model,
                    input=openai_messages,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=1,
                    text={"format": {"type": response_type}},
                )
                reply = response.output_text

        elif provider == "anthropic":
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                api_key = getpass.getpass("Enter Anthropic API key: ")
                os.environ["ANTHROPIC_API_KEY"] = api_key
            aclient = anthropic.Anthropic()
            system_parts = [m["content"] for m in flat if m["role"] == "system"]
            system = "\n\n".join(system_parts) if system_parts else None
            anth_msgs = []
            for m in flat:
                if m["role"] == "system":
                    continue
                r = m["role"]
                if r not in ("user", "assistant"):
                    r = "user"
                anth_msgs.append({"role": r, "content": m["content"]})
            if self.thinking_budget != 0 : 
                thinking_type = "enabled"
                msg = aclient.messages.create(
                model=self.model,  # e.g. "anthropic.claude-sonnet-4-5"
                max_tokens=self.max_tokens,
                system=system,
                messages=anth_msgs,
                temperature=self.temperature,
                thinking = {
                    "type" : thinking_type,
                    "budget_tokens" : self.thinking_budget
                }
            )
            else : 
                thinking_type = "disabled"
                msg = aclient.messages.create(
                model=self.model,  # e.g. "anthropic.claude-sonnet-4-5"
                max_tokens=self.max_tokens,
                system=system,
                messages=anth_msgs,
                temperature=self.temperature,
                thinking = {
                    "type" : thinking_type,
                }
            )
            text_parts = []
            for block in msg.content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                    text_parts.append(block.text)
            reply = "".join(text_parts).strip()
            if not reply and msg.content:
                reply = str(msg.content[0])

        elif provider == "bedrock":
            import anthropic

            aclient = anthropic.AnthropicBedrock(aws_region="us-east-1")

            system_parts = [m["content"] for m in flat if m["role"] == "system"]
            system = "\n\n".join(system_parts) if system_parts else None
            anth_msgs = []
            for m in flat:
                if m["role"] == "system":
                    continue
                r = m["role"]
                if r not in ("user", "assistant"):
                    r = "user"
                anth_msgs.append({"role": r, "content": m["content"]})

            if self.thinking_budget != 0 : 
                thinking_type = "enabled"
                msg = aclient.messages.create(
                model=self.model,  # e.g. "anthropic.claude-sonnet-4-5"
                max_tokens=self.max_tokens,
                system=system,
                messages=anth_msgs,
                temperature=self.temperature,
                thinking = {
                    "type" : thinking_type,
                    "budget_tokens" : self.thinking_budget
                }
            )
            else : 
                thinking_type = "disabled"
                msg = aclient.messages.create(
                model=self.model,  # e.g. "anthropic.claude-sonnet-4-5"
                max_tokens=self.max_tokens,
                system=system,
                messages=anth_msgs,
                temperature=self.temperature,
                thinking = {
                    "type" : thinking_type,
                }
            )


            text_parts = []
            for block in msg.content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                    text_parts.append(block.text)
            reply = "".join(text_parts).strip()
            if not reply and msg.content:
                reply = str(msg.content[0])


        elif provider == "google":
            from google import genai
            from google.genai import types

            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
                "GEMINI_API_KEY"
            )
            if not api_key:
                api_key = getpass.getpass("Enter Google AI (Gemini) API key: ")
                os.environ["GOOGLE_API_KEY"] = api_key
            client = genai.Client(api_key=api_key)
            system_parts = [m["content"] for m in flat if m["role"] == "system"]
            gemini_contents = []
            for m in flat:
                if m["role"] == "system":
                    continue
                role = "user"
                if m["role"] == "assistant":
                    role = "model"
                gemini_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=m["content"])],
                    )
                )
            gen_cfg = {
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if system_parts:
                gen_cfg["system_instruction"] = "\n\n".join(system_parts)
            if response_type == "json_object":
                gen_cfg["response_mime_type"] = "application/json"
            response = client.models.generate_content(
                model=self.model,
                contents=gemini_contents or "",
                config=types.GenerateContentConfig(**gen_cfg),
            )
            reply = response.text

        elif provider == "ollama":
            import ollama

            opts = {"model": self.model, "messages": flat}
            if response_type == "json_object":
                opts["format"] = "json"
            resp = ollama.chat(**opts)
            reply = resp["message"]["content"]

        elif provider == "hf":
            from transformers import AutoTokenizer, pipeline

            if self._hf_pipeline is None:
                self._hf_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=(self.tokenizer or self.model),
                )
            if self._hf_tokenizer is None:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model)
            tok = self._hf_tokenizer
            json_suffix = ""
            if response_type == "json_object":
                json_suffix = "\n\nReply with valid JSON only. No markdown or explanation."
            if hasattr(tok, "apply_chat_template") and getattr(
                tok, "chat_template", None
            ):
                prompt_text = tok.apply_chat_template(
                    flat, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = "\n\n".join(
                    f"{m['role']}: {m['content']}" for m in flat
                )
            prompt_text = prompt_text + json_suffix
            out = self._hf_pipeline(
                prompt_text,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                return_full_text=False,
            )
            reply = (out[0].get("generated_text") or "").strip()

        else:
            raise ValueError(f"Unknown provider: {provider!r}")

        if self.debug:
            print(reply)
        if cache:
            self.add_assistant_message(reply)
        return reply

    def load_json(self, s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    def call_json(self, prompt=""):
        self.add_system_message("Reply must be JSON format.")
        reply = self.call(prompt=prompt, response_type="json_object")
        if not reply:
            print("Empty reply")
            return None

        reply = reply.strip()
        if reply.startswith("```json"):
            reply = reply[len("```json") :].strip()
            if reply.endswith("```"):
                reply = reply[:-3].strip()

        if not (reply.startswith("{") and reply.endswith("}")):
            print("Not JSON structure")
            return None

        try:
            return self.load_json(reply)
        except Exception:
            print("Error parsing JSON")
            return None
