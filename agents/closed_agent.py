"""OpenAI-backed closed-model agent for grounded comparisons."""

import os

from openai import OpenAI

from base_agent import OpenAICompatibleAgent
from search_tool import get_system_prompt


class ClosedModelAgent(OpenAICompatibleAgent):
    """OpenAI agent using the same You.com Search tool loop as Parasail.

    The model is forced to call web_search on the first turn so it cannot
    answer from training data for current questions.
    """

    def __init__(self, model: str):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key, timeout=120.0)
        system_prompt = get_system_prompt(model) + (
            "\n\nCRITICAL: Before answering any question that asks about "
            "current events, recent news, or anything that may have changed "
            "since your training cutoff, you MUST call the web_search tool. "
            "Do not answer from training data when live information is available."
        )
        super().__init__(
            client=client,
            model=model,
            system_prompt=system_prompt,
            max_tokens_param="max_completion_tokens",
            # GPT-5.x rejects function tools on /v1/chat/completions unless
            # reasoning_effort is 'none' (otherwise: use /v1/responses).
            extra_body={"reasoning_effort": "none"},
            force_search_first=True,
        )
