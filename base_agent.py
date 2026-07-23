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
    ANSWER_MAX_TOKENS,
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
        force_search_first: bool = False,
    ):
        super().__init__(model, system_prompt)
        self.client = client
        self.max_tokens_param = max_tokens_param
        self.extra_body = extra_body
        self.force_search_first = force_search_first

    def stream(self, question: str, max_rounds: int = None, prior_messages: list = None, max_searches: int = None) -> Generator[dict, None, None]:
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
        budget_prompt_added = False
        cumulative_answer = ""
        answer_emit_chars = 0
        final_aggregate = None

        def consume_stream(response_stream, round_num):
            nonlocal cumulative_answer, answer_emit_chars
            content_parts = []
            reasoning_parts = []
            tool_calls = {}
            finish_reason = None
            model = None
            usage = None
            saw_choice = False
            first_chunk = True

            for chunk in response_stream:
                if first_chunk:
                    first_chunk = False
                    if round_num == 0:
                        stats["connect_ms"] = round((time.perf_counter() - t_connect) * 1000)
                if model is None:
                    model = getattr(chunk, "model", None)
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                if not getattr(chunk, "choices", None):
                    continue

                saw_choice = True
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                content = getattr(delta, "content", None) or ""
                reasoning = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    or (getattr(delta, "model_extra", None) or {}).get("reasoning", "")
                    or ""
                )
                if content:
                    content_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)

                streamed_piece = content or reasoning
                if streamed_piece:
                    cumulative_answer += streamed_piece
                    if len(cumulative_answer) - answer_emit_chars >= 60:
                        answer_emit_chars = len(cumulative_answer)
                        yield {
                            "event": "answer",
                            "text": cumulative_answer,
                            "sources": stats["sources"],
                        }

                for fragment in (getattr(delta, "tool_calls", None) or []):
                    index = getattr(fragment, "index", 0)
                    call = tool_calls.setdefault(index, {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    fragment_id = getattr(fragment, "id", None)
                    if fragment_id:
                        call["id"] = fragment_id
                    function = getattr(fragment, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None)
                        arguments = getattr(function, "arguments", None)
                        if name:
                            call["function"]["name"] += name
                        if arguments:
                            call["function"]["arguments"] += arguments

            return {
                "content": "".join(content_parts),
                "reasoning": "".join(reasoning_parts),
                "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
                "finish_reason": finish_reason,
                "model": model,
                "usage": usage,
                "saw_choice": saw_choice,
            }

        for round_num in range(_max_rounds):
            searches_left = None if max_searches is None else max(0, max_searches - stats["search_calls"])
            try:
                create_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    self.max_tokens_param: ANSWER_MAX_TOKENS if searches_left == 0 else MAX_TOKENS,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if searches_left == 0:
                    # Search budget exhausted — drop tools and explicitly instruct the
                    # model to answer from what it has. (Just omitting tools or setting
                    # tool_choice="none" makes some models emit raw tool-call tokens
                    # into the answer instead of synthesizing.)
                    if not budget_prompt_added:
                        messages.append({
                            "role": "user",
                            "content": "You have reached your web search limit. Do NOT "
                                       "search again. Using only the information already "
                                       "gathered above, write the final answer now.",
                        })
                        budget_prompt_added = True
                else:
                    create_kwargs["tools"] = [TOOL_SCHEMA]
                    if self.force_search_first and round_num == 0:
                        create_kwargs["tool_choice"] = "required"
                if self.extra_body:
                    create_kwargs["extra_body"] = self.extra_body
                t_connect = time.perf_counter()
                stream_chunks = consume_stream(
                    self.client.chat.completions.create(**create_kwargs),
                    round_num,
                )
                while True:
                    try:
                        yield next(stream_chunks)
                    except StopIteration as stop:
                        aggregate = stop.value
                        break
            except Exception as e:
                logger.error("API call failed (model=%s, round=%d): %s", self.model, round_num, e)
                stats["answer"] = f"Error: {e}"
                stats["latency_ms"] = (time.perf_counter() - t0) * 1000
                yield {"event": "answer", "text": stats["answer"], "sources": []}
                yield {"event": "done", "stats": stats}
                return

            if aggregate["usage"]:
                call_input = getattr(aggregate["usage"], "prompt_tokens", 0)
                call_output = getattr(aggregate["usage"], "completion_tokens", 0)
                stats["token_breakdown"]["input"] += call_input
                stats["token_breakdown"]["output"] += call_output
                if round_num == 0:
                    baseline_input = call_input
            stats["api_calls"] += 1
            if round_num == 0:
                stats["model_confirmed"] = aggregate["model"]

            if not aggregate["saw_choice"]:
                logger.error("API returned empty choices (model=%s, round=%d)", self.model, round_num)
                break
            if aggregate["finish_reason"] != "tool_calls" or not aggregate["tool_calls"]:
                stats["answer"] = aggregate["content"] or aggregate["reasoning"] or ""
                final_aggregate = aggregate
                break

            messages.append({
                "role": "assistant",
                "content": aggregate["content"] or None,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": tool_call["function"],
                    }
                    for tool_call in aggregate["tool_calls"]
                ],
            })

            for tool_call in aggregate["tool_calls"]:
                function = tool_call["function"]
                if function["name"] != "web_search":
                    logger.warning("LLM called unknown tool: %s", function["name"])
                    continue

                if searches_left is not None and searches_left <= 0:
                    # Over the search budget for this request — decline extra calls
                    # but still answer each tool_call id so the message log stays valid.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": "Search limit reached — answer using the information already gathered.",
                    })
                    continue

                try:
                    args = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    logger.error("Malformed tool arguments (model=%s): %s", self.model, function["arguments"][:200])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
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
                if searches_left is not None:
                    searches_left -= 1

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
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })
        else:
            stats["hit_round_limit"] = True
            logger.warning("Tool loop hit max_rounds=%d (model=%s) — forcing synthesis call", _max_rounds, self.model)
            try:
                synthesis_kwargs = {
                    "model": self.model,
                    "messages": messages + [{"role": "user", "content": "Based on all the search results above, please provide a comprehensive answer."}],
                    self.max_tokens_param: ANSWER_MAX_TOKENS,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if self.extra_body:
                    synthesis_kwargs["extra_body"] = self.extra_body
                t_connect = time.perf_counter()
                stream_chunks = consume_stream(
                    self.client.chat.completions.create(**synthesis_kwargs),
                    _max_rounds,
                )
                while True:
                    try:
                        yield next(stream_chunks)
                    except StopIteration as stop:
                        aggregate = stop.value
                        break
                if aggregate["usage"]:
                    stats["token_breakdown"]["input"] += getattr(aggregate["usage"], "prompt_tokens", 0)
                    stats["token_breakdown"]["output"] += getattr(aggregate["usage"], "completion_tokens", 0)
                stats["api_calls"] += 1
                if aggregate["model"] and stats["model_confirmed"] is None:
                    stats["model_confirmed"] = aggregate["model"]
                if aggregate["saw_choice"]:
                    stats["answer"] = aggregate["content"] or aggregate["reasoning"] or ""
                    final_aggregate = aggregate
            except Exception as e:
                logger.error("Synthesis call failed (model=%s): %s", self.model, e)

        if final_aggregate is None or not final_aggregate["saw_choice"]:
            stats["answer"] = "Error: no response from model"
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000
            yield {"event": "answer", "text": stats["answer"], "sources": []}
            yield {"event": "done", "stats": stats}
            return

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
