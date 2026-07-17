"""
demo.py — Interactive REPL for Parasail + You.com Search.

Usage:
    python3 demo.py <model-slug> [question]

    # Interactive mode:
    python3 demo.py meta-llama/llama-3.3-70b-instruct

    # One-shot:
    python3 demo.py meta-llama/llama-3.3-70b-instruct "Latest AI news?"

Environment:
    PARASAIL_API_KEY  — from parasail.io dashboard → API Keys
    YDC_API_KEY       — from you.com/platform
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from agents.parasail_agent import ParasailAgent
from search_tool import format_tool_log


def run(model: str, question: str) -> None:
    agent = ParasailAgent(model=model)

    print(f"\nModel:    {model}")
    print(f"Question: {question}\n")
    print("─" * 60)

    t0 = time.perf_counter()

    def on_progress(msg: str):
        print(f"  › {msg}")

    result = agent.ask(question, on_progress=on_progress)

    elapsed = time.perf_counter() - t0
    print("─" * 60)
    print(f"\n{result['answer']}\n")

    if result["tool_calls"]:
        print("Search calls:")
        for entry in result["tool_calls"]:
            print(format_tool_log(entry))

    if result["sources"]:
        print("\nSources:")
        for i, url in enumerate(result["sources"], 1):
            print(f"  [{i}] {url}")

    print(f"\nTokens: {result['tokens_used']} | Latency: {elapsed:.1f}s | Searches: {result['search_calls']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 demo.py <model-slug> [question]")
        print("  e.g. python3 demo.py meta-llama/llama-3.3-70b-instruct")
        sys.exit(1)

    model_slug = args[0]
    question = " ".join(args[1:]) if len(args) > 1 else None

    if question:
        run(model_slug, question)
    else:
        print(f"Parasail demo — model: {model_slug}")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            run(model_slug, q)
