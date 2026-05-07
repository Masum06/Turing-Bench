"""
Optional hook used when you run::

    python run_judge.py --provider custom ...

Implement::

    predict(dialogue_a, dialogue_b) -> "A" | "B"

You may instead return a legacy JSON dict in the shape:
    {"result": {"verdict": "A"}}
"""


def predict(dialogue_a: str, dialogue_b: str) -> str | dict:
    """Return which transcript contains the human witness (A or B)."""
    raise NotImplementedError(
        "Either use a hosted model (e.g. python run_judge.py -m gpt-4o) or "
        "implement predict() and run with --provider custom."
    )
