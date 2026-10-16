import time
from huggingface_hub import InferenceClient
from config import (
    HF_TOKEN, HF_MODEL, HELPFUL_SYSTEM, TERSE_SYSTEM,
    MIN_CHOSEN_LENGTH, MIN_REJECTED_LENGTH, REQUEST_DELAY,
)

client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

_MAX_RETRIES = 3


def _chat(messages: list, max_tokens: int) -> str | None:
    for attempt in range(_MAX_RETRIES):
        try:
            text = client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            ).choices[0].message.content.strip()
            if text:
                return text
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def generate_responses(prompt: str) -> tuple[str, str] | None:
    helpful = _chat(
        messages=[
            {"role": "system", "content": HELPFUL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
    )
    time.sleep(REQUEST_DELAY)

    terse = _chat(
        messages=[
            {"role": "system", "content": TERSE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
    )
    time.sleep(REQUEST_DELAY)

    if helpful is None or terse is None:
        return None
    if len(helpful) < MIN_CHOSEN_LENGTH or len(terse) < MIN_REJECTED_LENGTH:
        return None
    if helpful == terse:
        return None

    return helpful, terse
