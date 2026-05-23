# `--provider custom` examples

Runs use `predict.predict(dialogue_a, dialogue_b)`. See [`model_clients.py`](model_clients.py) for hosted APIs.

## Minimal stub (returns all A — for wiring tests only)

```python
def predict(dialogue_a: str, dialogue_b: str) -> str:
    return "A"
```

```bash
python run_judge.py --provider custom --input sample.csv --limit 5 --output smoke.csv --no-threads
```

## Call your HTTP API yourself

```python
import json
from config import SYSTEM_PROMPT, USER_TEMPLATE


def predict(dialogue_a: str, dialogue_b: str) -> str:
    user = USER_TEMPLATE.format(dialogueA=dialogue_a, dialogueB=dialogue_b)
    text = your_http_chat(system=SYSTEM_PROMPT, user=user)
    data = json.loads(text)
    verdict = data.get("result", {}).get("verdict")
    if verdict not in ("A", "B"):
        raise ValueError(f"No verdict: {data!r}")
    return verdict
```

## Hosted APIs

For OpenAI, Anthropic, or Bedrock, run with `-m <model-id>` (provider is inferred). See the main [README](README.md).

To support a new model string, add an entry to the `MODEL_SUBSTRING_TO_PROVIDER` list in [`model_clients.py`](model_clients.py) (first match wins).
