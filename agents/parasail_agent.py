"""
parasail_agent.py — Parasail agent with You.com web search.

Parasail (parasail.io) is a cloud GPU inference platform hosting open-weight
models across 40+ data centers. It exposes two OpenAI-compatible endpoints:

  /v1/chat/completions  — standard tool-use loop (all models, serverless + dedicated)
  /v1/responses         — OpenAI Responses API shape (serverless only; stateless —
                          no previous_response_id chaining, full history required)

Both paths use the same base URL and API key.

Usage:
    from agents.parasail_agent import ParasailAgent

    # chat.completions path (default — works on all models):
    agent = ParasailAgent(model="meta-llama/llama-3.3-70b-instruct")
    result = agent.ask("What happened in AI news this week?")
    print(result["answer"])

Requirements:
    pip install openai python-dotenv
    export PARASAIL_API_KEY="..."   # from saas.parasail.io/keys
    export YDC_API_KEY="..."

Parasail serverless model IDs (verify current list via GET /v1/models):
    parasail-deepseek-r1
    parasail-llama-33-70b-fp8
    meta-llama/Meta-Llama-3-8B-Instruct   (dedicated)
    Qwen/Qwen3-VL-8B-Instruct             (dedicated)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from base_agent import OpenAICompatibleAgent, _empty_stats

_PARASAIL_BASE_URL = "https://api.parasail.io/v1"


class ParasailAgent(OpenAICompatibleAgent):
    """Parasail agent using You.com Search via chat.completions tool-use loop.

    Accepts any Parasail model slug. Drop-in replacement for OpenRouterAgent
    — same interface, different base URL and API key env var.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ):
        if not model:
            raise ValueError(
                "model is required — pass a Parasail model slug, "
                "e.g. 'meta-llama/llama-3.3-70b-instruct'"
            )
        resolved_key = api_key or os.environ.get("PARASAIL_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "PARASAIL_API_KEY not set. "
                "Pass api_key= or export PARASAIL_API_KEY."
            )
        client = OpenAI(
            api_key=resolved_key,
            base_url=_PARASAIL_BASE_URL,
            timeout=120.0,
        )
        super().__init__(client=client, model=model)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import MODELS, model_id as resolve_model_id

    args = sys.argv[1:]
    if not args:
        print("Usage: python3 parasail_agent.py <model-key-or-slug> <question>")
        print("  e.g. python3 parasail_agent.py gpt-oss-20b 'Latest AI news?'")
        print(f"  Keys: {list(MODELS)}")
        sys.exit(1)

    raw_model = args[0]
    # Accept either a short registry key ("gpt-oss-20b") or a full slug
    model_slug = resolve_model_id(raw_model) if raw_model in MODELS else raw_model
    question = " ".join(args[1:]) or input("Ask something: ")
    agent = ParasailAgent(model=model_slug)
    result = agent.ask(question)
    print(result["answer"])
    if result["sources"]:
        print(f"\nSources: {result['sources']}")
