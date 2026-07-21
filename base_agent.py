"""
base_agent.py — Tool-use loop implementation for OpenAI-compatible APIs.

Classes:
    BaseAgent               — ABC defining the stream() + ask() contract
    OpenAICompatibleAgent   — OpenAI chat.completions (Parasail, Llama, etc.)

Public interface:
    stream(question) → Generator[dict]
        Yields structured event dicts as the loop progresses.

    ask(question, on_progress=None) → dict
        Synchronous wrapper around stream(). Fires on_progress(str) at
        each step. Returns the final stats dict.

Event shapes from stream():
    {"event": "tool_call",     "round": int, "search_num": int, "query": str, "params": dict}
    {"event": "search_result", "round": int, "search_num": int, "result_count": int,
                               "latency_ms": float, "sources": list, "search_uuid": str}
    {"event": "answer",        "text": str, "sources": list}
    {"event": "done",          "stats": dict}

Stats dict — canonical shape returned by ask() on all agents:
    answer           str    Final answer text.
    sources          list   Ordered source URLs cited in the answer.
    tool_calls       list   YDC search log entries.
    model            str    Model ID as passed by the caller.
    model_confirmed  str    Model ID as confirmed by the API response (may differ).
    interface        str    Which search path ran.
    tokens_used      int    Total tokens consumed (input + output).
    token_breakdown  dict   {"input": int, "output": int, "search_context": int}
    search_calls     int    Number of search round-trips executed.
    api_calls        int    Number of LLM API calls made (>1 for multi-round loops).
    latency_ms       float  Wall-clock time from first request to final answer.

    See _empty_stats() for the authoritative definition.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Generator

from search_tool import (
    INTEGRATION_INTERFACE,
    get_system_prompt,
    MAX_TOKENS,
    MAX_TOOL_ROUNDS,
    TOOL_SCHEMA,
    build_tool_log_entry,
    execute_search,
    extract_urls,
    extract_search_uuid,
    is_verbose,
    format_tool_log,
)

logger = logging.getLogger(__name__)


# ─── Shared utilities ─────────────────────────────────────────────────────────

def _empty_stats(model: str = "") -> dict:
    """Return a fresh stats dict — the canonical shape for all agent return values."""
    return {
        "answer": "",
        "sources": [],
        "tool_calls": [],
        "model": model,
        "model_confirmed": None,
        "interface": INTEGRATION_INTERFACE,
        "tokens_used": 0,
        "token_breakdown": {"input": 0, "output": 0, "search_context": 0},
        "search_calls": 0,
        "api_calls": 0,
        "latency_ms": 0.0,
        "connect_ms": 0,
        "hit_round_limit": False,
        "search_uuid": "",
        "not_supported": False,
    }


def _event_to_message(event: dict) -> str:
    """Convert a structured event dict to a display string for on_progress callbacks."""
    e = event.get("event")
    if e == "tool_call":
        query = event.get("query", "")
        params = event.get("params", {})
        param_str = " • ".join(f"{k}={v}" for k, v in params.items() if v)
        label = query + (" • " + param_str if param_str else "")
        return f"Search {event.get('search_num', '?')}: {label}"
    if e == "search_result":
        return f"Results received ({event.get('latency_ms', 0):.0f}ms)"
    if e == "answer":
        return "Generating answer..."
    return ""


# ─── Base class ───────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base for all grounded LLM agents.

    Subclasses implement stream(). The ask() method is provided here and
    must not be overridden — it is the single conversion point from the
    streaming interface to the synchronous dict interface.
    """

    def __init__(self, model: str, system_prompt: str = None):
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else get_system_prompt(model)

    @abstractmethod
    def stream(self, question: str, max_rounds: int = None, prior_messages: list = None) -> Generator[dict, None, None]:
        """Run the tool-use loop, yielding structured events at each step.

        max_rounds:     cap the tool-use loop. None uses MAX_TOOL_ROUNDS from search_tool.py.
        prior_messages: list of prior {"role", "content"} turns for multi-turn chat.

        Must always yield "answer" followed by "done" as the last two events.
        """
        ...

    def ask(self, question: str, on_progress=None, max_rounds: int = None) -> dict:
        """Synchronous wrapper around stream().

        Consumes all events, fires on_progress(str) for each step, and
        returns the stats dict from the final "done" event.
        """
        stats = None
        try:
            for event in self.stream(question, max_rounds=max_rounds):
                if on_progress:
                    msg = _event_to_message(event)
                    if msg:
                        try:
                            on_progress(msg)
                        except Exception:
                            pass
                if event["event"] == "done":
                    stats = event["stats"]
        except Exception as e:
            logger.error("stream() raised unexpectedly (model=%s): %s", self.model, e)
            if stats is None:
                stats = _empty_stats(self.model)
                stats["answer"] = ""
        return stats


# ─── OpenAI chat.completions ──────────────────────────────────────────────────

class OpenAICompatibleAgent(BaseAgent):
    """Tool-use loop using OpenAI's chat.completions API.

    Compatible with any inference provider that speaks the OpenAI
    chat.completions format (Parasail, Together AI, Fireworks, Groq, etc.).

    Args:
        client:           An openai.OpenAI instance pointed at the provider's base URL.
        model:            Model ID string.
        max_tokens_param: "max_tokens" for most providers; "max_completion_tokens"
                          for providers that deprecated the older param name.
        extra_body:       Provider-specific kwargs passed through to create().
    """

    def __init__(
        self,
        client,
        model: str,
        system_prompt: str = None,
        max_tokens_param: str = "max_tokens",
        extra_body: dict | None = None,
    ):
        super().__init__(model, system_prompt)
        self.client = client
        self.max_tokens_param = max_tokens_param
        self.extra_body = extra_body

    def stream(self, question: str, max_rounds: int = None, prior_messages: list = None) -> Generator[dict, None, None]:
        stats = _empty_stats(self.model)
        messages = [
            {"role": "system", "content": self.system_prompt},
            *(prior_messages or []),
            {"role": "user", "content": question},
        ]
        t0 = time.perf_counter()
        baseline_input = 0
        search_num = 0
        _max_rounds = max_rounds if max_rounds is not None else MAX_TOOL_ROUNDS
        response = None

        for round_num in range(_max_rounds):
            try:
                create_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": [TOOL_SCHEMA],
                    self.max_tokens_param: MAX_TOKENS,
                }
                if self.extra_body:
                    create_kwargs["extra_body"] = self.extra_body
                t_connect = time.perf_counter()
                response = self.client.chat.completions.create(**create_kwargs)
                if round_num == 0:
                    stats["connect_ms"] = round((time.perf_counter() - t_connect) * 1000)
            except Exception as e:
                logger.error("API call failed (model=%s, round=%d): %s", self.model, round_num, e)
                stats["answer"] = f"Error: {e}"
                stats["latency_ms"] = (time.perf_counter() - t0) * 1000
                yield {"event": "answer", "text": stats["answer"], "sources": []}
                yield {"event": "done", "stats": stats}
                return

            if response.usage:
                call_input = getattr(response.usage, "prompt_tokens", 0)
                call_output = getattr(response.usage, "completion_tokens", 0)
                stats["token_breakdown"]["input"] += call_input
                stats["token_breakdown"]["output"] += call_output
                if round_num == 0:
                    baseline_input = call_input
            stats["api_calls"] += 1
            if round_num == 0:
                stats["model_confirmed"] = getattr(response, "model", None)

            if not response.choices:
                logger.error("API returned empty choices (model=%s, round=%d)", self.model, round_num)
                break
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                break

            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                if tool_call.function.name != "web_search":
                    logger.warning("LLM called unknown tool: %s", tool_call.function.name)
                    continue

                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    logger.error("Malformed tool arguments (model=%s): %s", self.model, tool_call.function.arguments[:200])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Error: could not parse tool arguments",
                    })
                    continue

                search_num += 1
                yield {
                    "event": "tool_call",
                    "round": round_num + 1,
                    "search_num": search_num,
                    "query": args.get("query", ""),
                    "params": {k: v for k, v in args.items() if k != "query"},
                }

                search_t0 = time.perf_counter()
                result = execute_search(args)
                elapsed_ms = (time.perf_counter() - search_t0) * 1000

                entry = build_tool_log_entry(args, result, elapsed_ms)
                stats["tool_calls"].append(entry)
                sources = extract_urls(result)
                stats["sources"].extend(sources)
                uuid = extract_search_uuid(result)
                if uuid:
                    stats["search_uuid"] = uuid
                stats["search_calls"] += 1

                if is_verbose():
                    print(format_tool_log(entry))

                yield {
                    "event": "search_result",
                    "round": round_num + 1,
                    "search_num": search_num,
                    "result_count": entry["result_count"],
                    "latency_ms": elapsed_ms,
                    "sources": sources,
                    "search_uuid": uuid or "",
                }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            stats["hit_round_limit"] = True
            logger.warning("Tool loop hit max_rounds=%d (model=%s) — forcing synthesis call", _max_rounds, self.model)
            try:
                synthesis_kwargs = {
                    "model": self.model,
                    "messages": messages + [{"role": "user", "content": "Based on all the search results above, please provide a comprehensive answer."}],
                    self.max_tokens_param: MAX_TOKENS,
                }
                if self.extra_body:
                    synthesis_kwargs["extra_body"] = self.extra_body
                response = self.client.chat.completions.create(**synthesis_kwargs)
                if response.usage:
                    stats["token_breakdown"]["input"] += getattr(response.usage, "prompt_tokens", 0)
                    stats["token_breakdown"]["output"] += getattr(response.usage, "completion_tokens", 0)
                stats["api_calls"] += 1
            except Exception as e:
                logger.error("Synthesis call failed (model=%s): %s", self.model, e)

        if not response or not response.choices:
            stats["answer"] = "Error: no response from model"
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000
            yield {"event": "answer", "text": stats["answer"], "sources": []}
            yield {"event": "done", "stats": stats}
            return

        message = response.choices[0].message
        stats["answer"] = message.content or (message.model_extra or {}).get("reasoning", "") or ""
        total_in = stats["token_breakdown"]["input"]
        total_out = stats["token_breakdown"]["output"]
        stats["tokens_used"] = total_in + total_out
        stats["token_breakdown"]["search_context"] = (
            max(0, total_in - baseline_input * stats["api_calls"])
            if stats["api_calls"] > 1 else 0
        )
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000

        yield {"event": "answer", "text": stats["answer"], "sources": stats["sources"]}
        yield {"event": "done", "stats": stats}
