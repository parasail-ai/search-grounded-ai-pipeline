"""OpenAI-backed closed-model agent for grounded comparisons."""

import os

from openai import OpenAI

from base_agent import OpenAICompatibleAgent


class ClosedModelAgent(OpenAICompatibleAgent):
    """OpenAI agent using the same You.com tool-use loop as Parasail."""

    def __init__(self, model: str):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key, timeout=120.0)
        super().__init__(
            client=client,
            model=model,
            max_tokens_param="max_completion_tokens",
            # GPT-5.x rejects function tools on /v1/chat/completions unless
            # reasoning_effort is 'none' (otherwise: use /v1/responses).
            extra_body={"reasoning_effort": "none"},
        )
