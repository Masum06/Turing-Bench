# ══════════════════════════════════════════════════════════════════════════════
#  DEFINE YOUR MODEL HERE
#
#  Fill in the predict() function below. It receives the two dialogue
#  transcripts as plain strings and must return either "A" or "B".
#
#  Use SYSTEM_PROMPT and USER_TEMPLATE.format(dialogueA=..., dialogueB=...)
#  to build your prompt (found in config.py).
#
#  A few copy-paste starter examples are included as comments beneath
#  the function. Some of these use tinyagent.py, a wrapper class that 
#  handles model configuration automatically.
#
#  Thread safety: if USE_THREADS = True, predict() will be called from
#  multiple threads simultaneously. Stateless API clients (OpenAI, Groq, etc.)
#  are safe by default. For local models, set USE_THREADS = False or ensure
#  your pipeline/model object is thread-safe.
# ══════════════════════════════════════════════════════════════════════════════
from config import SYSTEM_PROMPT, USER_TEMPLATE, MAX_RETRIES, BASE_DELAY
from tinyagent import *


def predict(dialogueA: str, dialogueB: str) -> str:
    """
    Return "A" if dialogueA is the human-human conversation, "B" otherwise.
    Replace the body of this function with your own model call.
    """
    
    raise NotImplementedError(
        "Please fill in the predict() function with your model. "
        "See the examples in the comments below."
    )

# EXAMPLE A — OpenAI-compatible API (OpenAI, Together, Groq, Ollama, etc.)
# Works with any provider that follows the OpenAI chat completion format.
# Safe with USE_THREADS = True
"""
Terminal: pip install openai

import time
from openai import RateLimitError, APIError

MAX_RETRIES = 5
BASE_DELAY  = 1.0   # seconds — doubles each attempt: 1, 2, 4, 8, 16

def predict(dialogueA: str, dialogueB: str) -> str:
    prompt = USER_TEMPLATE.format(dialogueA=dialogueA, dialogueB=dialogueB)

    for attempt in range(MAX_RETRIES):
        try:
            agent = TinyAgent(model="gpt-5", provider="openai")
            agent.set_max_tokens(1024)
            agent.set_reasoning_effort("medium")
            agent.add_system_message(SYSTEM_PROMPT)
            return agent.call_json(prompt=prompt)
        except RateLimitError:
            wait = BASE_DELAY * (2 ** attempt)
            print(f"Rate limited (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait:.1f}s...")
            time.sleep(wait)
        except APIError as e:
            wait = BASE_DELAY * (2 ** attempt)
            print(f"API error: {e} (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(f"predict() failed after {MAX_RETRIES} attempts")
"""

# EXAMPLE B — Hugging Face transformers (local model)
# Set USE_THREADS = False for local models
"""
Terminal: pip install transformers torch

from transformers import pipeline

pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

def predict(dialogueA: str, dialogueB: str) -> str:
    prompt = SYSTEM_PROMPT + "\\n\\n" + USER_TEMPLATE.format(
        dialogueA=dialogueA, dialogueB=dialogueB
    )
    out = pipe(prompt, max_new_tokens=5, temperature=0.0)[0]["generated_text"]
    return out
"""

# EXAMPLE C — Ollama (local server, any model pulled via `ollama pull`)
# Set USE_THREADS = False for local models
"""
Terminal: pip install ollama

import ollama

def predict(dialogueA: str, dialogueB: str) -> str:
    prompt = USER_TEMPLATE.format(dialogueA=dialogueA, dialogueB=dialogueB)
    resp = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp["message"]["content"]
"""