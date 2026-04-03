# ══════════════════════════════════════════════════════════════════════════════
#  DEFINE YOUR MODEL HERE
#
#  Fill in the predict() function below. It receives the two dialogue
#  transcripts as plain strings and must return either "A" or "B".
#
#  Use SYSTEM_PROMPT and USER_TEMPLATE.format(dialogueA=..., dialogueB=...)
#  to build your prompt.
#
#  A few copy-paste starter examples are included as comments beneath
#  the function.
#
#  Thread safety: if USE_THREADS = True, predict() will be called from
#  multiple threads simultaneously. Stateless API clients (OpenAI, Groq, etc.)
#  are safe by default. For local models, set USE_THREADS = False or ensure
#  your pipeline/model object is thread-safe.
# ══════════════════════════════════════════════════════════════════════════════
from config import SYSTEM_PROMPT, USER_TEMPLATE, MAX_RETRIES, BASE_DELAY

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

import os
import time
from openai import OpenAI, RateLimitError, APIError

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],   # or your provider's key
    base_url="https://api.openai.com/v1",   # swap for Groq/Together/etc.
)

MAX_RETRIES = 5
BASE_DELAY  = 1.0   # seconds — doubles each attempt: 1, 2, 4, 8, 16

def predict(dialogueA: str, dialogueB: str) -> str:
    prompt = USER_TEMPLATE.format(dialogueA=dialogueA, dialogueB=dialogueB)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",             # swap for any model name
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=1024,
                temperature=1,
            )
            return resp.choices[0].message.content
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