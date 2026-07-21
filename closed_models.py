"""Closed-model benchmark registry with published pricing."""

CLOSED_MODELS = {
    "gpt-5.6-sol": {
        "model_id": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "provider": "OpenAI",
        "pricing": {"input": 5.0, "output": 30.0, "cache_read": 0.50},
        "slug_confirmed": True,
        "intelligence": None,
    },
}
