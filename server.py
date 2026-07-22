"""
server.py — Parasail + You.com Search playground.

Run:
    python3 server.py          # http://localhost:8080
    python3 server.py --port 9000

Endpoints:
    GET  /                     → index.html
    GET  /api/models           → JSON list of available models
    POST /api/ask              → SSE stream of search + answer events
"""

import json
import os
import sys
import time
import threading
import copy
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from logger_setup import configure, get_logger
configure(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

import inspect
from agents.parasail_agent import ParasailAgent
from agents.closed_agent import ClosedModelAgent
from agents.pipeline_agent import enrich, brief_stream, email_stream, _llm_stream
from search_tool import SYSTEM_PROMPT, execute_search
from models import DEFAULT_MODEL, get_live_model_ids, list_models_from_api, MODELS
from closed_models import CLOSED_MODELS
from costs import calculate_costs, format_cost, YDC_SEARCH_COST_PER_CALL

RETRY_DELAY_S = 5          # seconds to wait before one 429 retry
MAX_REQUEST_BODY = 512_000  # 512 KB — sufficient for any question/brief payload

APP_DIR = Path(__file__).parent
PORT = int(os.getenv("PORT", "8091"))
HOST = os.getenv("HOST", "0.0.0.0")
EXPOSE_SOURCE = os.getenv("EXPOSE_SOURCE", "true").lower() == "true"
CLOSED_CACHE_PATH = APP_DIR / "cache" / "closed_runs.json"
_CLOSED_CACHE_VERSION = "2"    # bump when the closed agent behavior changes
_CACHE_LOCK = threading.Lock()


def _cache_key(question: str, model_id: str, search_enabled: bool) -> str:
    normalized = " ".join(question.split()).casefold()
    return json.dumps([normalized, model_id, bool(search_enabled), _CLOSED_CACHE_VERSION], separators=(",", ":"))


def _load_closed_cache() -> dict:
    try:
        return json.loads(CLOSED_CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_closed_cache(cache: dict):
    CLOSED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CLOSED_CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(cache, indent=2))
    temp_path.replace(CLOSED_CACHE_PATH)


def _closed_cache_get(key: str):
    with _CACHE_LOCK:
        return _load_closed_cache().get(key)


def _closed_cache_put(key: str, value: dict):
    with _CACHE_LOCK:
        cache = _load_closed_cache()
        cache[key] = value
        _save_closed_cache(cache)


def _event_costs(stats: dict, pricing: dict) -> dict:
    costs = calculate_costs(stats, pricing)
    return {
        "total": round(costs["total"], 6),
        "llm": round(costs["llm"], 6),
        "search": round(costs["search"], 6),
        "livecrawl_overage": round(costs["livecrawl_overage"], 6),
        "total_fmt": format_cost(costs["total"]),
    }


def _event_pricing(pricing: dict) -> dict:
    return {
        "input_per_m": pricing["input"],
        "output_per_m": pricing["output"],
        "cache_read_per_m": pricing.get("cache_read", 0),
        "search_per_call": YDC_SEARCH_COST_PER_CALL,
    }


def _projected_closed_stats(open_stats: dict, closed_cfg: dict) -> dict:
    closed_stats = copy.deepcopy(open_stats)
    closed_stats["model"] = closed_cfg["model_id"]
    closed_stats["model_confirmed"] = closed_cfg["model_id"]
    closed_stats["answer"] = "Projected closed-model answer omitted; cost estimated from open-model token usage."
    return closed_stats


def _is_parasail_overload(error) -> bool:
    message = str(error or "")
    return "429" in message or "overloaded" in message.lower()


def _available_model_keys(live_ids=None) -> list:
    if live_ids is None:
        return list(MODELS)
    live_keys = [key for key, cfg in MODELS.items() if cfg["model_id"] in live_ids]
    return live_keys or list(MODELS)


def _resolve_model_key(requested=None) -> str:
    live_ids = get_live_model_ids()
    available = _available_model_keys(live_ids)
    if requested in available:
        return requested
    if DEFAULT_MODEL in available:
        return DEFAULT_MODEL
    return available[0]


def _model_catalog_payload(live_ids=None) -> dict:
    matched_keys = [
        key for key, cfg in MODELS.items()
        if live_ids is not None and cfg["model_id"] in live_ids
    ]
    fallback = live_ids is None or not matched_keys
    available_keys = matched_keys or list(MODELS)
    hidden = [key for key in MODELS if key not in available_keys]
    if hidden:
        logger.info("model catalog hiding offline curated models: %s", ", ".join(hidden))
    if fallback:
        reason = "lookup failed" if live_ids is None else "no curated models matched"
        logger.warning("model catalog fallback: serving all curated models (%s)", reason)
    return {
        "models": [
            {
                "key": key,
                "model_id": MODELS[key]["model_id"],
                "display_name": MODELS[key]["display_name"],
                "provider": MODELS[key]["provider"],
                "pricing": MODELS[key]["pricing"],
                "live": live_ids is not None and MODELS[key]["model_id"] in live_ids,
            }
            for key in available_keys
        ]
    }


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info(fmt, *args)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(APP_DIR / "index.html", "text/html")
        elif self.path == "/api/config":
            from search_tool import (
                MAX_TOOL_ROUNDS, MAX_TOKENS,
                SEARCH_TIMEOUT_SECONDS, MAX_LIVECRAWL_CHARS,
            )
            self._json({
                "max_tool_rounds":       MAX_TOOL_ROUNDS,
                "max_tokens":            MAX_TOKENS,
                "search_timeout_s":      SEARCH_TIMEOUT_SECONDS,
                "max_livecrawl_chars":   MAX_LIVECRAWL_CHARS,
                "llm_api_timeout_s":     120,
                "ydc_cost_per_call":     YDC_SEARCH_COST_PER_CALL,
                "system_prompt":         SYSTEM_PROMPT,
            })
        elif self.path == "/api/source":
            if not EXPOSE_SOURCE:
                self._json({"error": "source endpoint disabled"}, 403)
                return
            self._json({
                "enrich":              inspect.getsource(enrich),
                "brief_stream":        inspect.getsource(brief_stream),
                "_llm_stream":         inspect.getsource(_llm_stream),
                "email_stream":        inspect.getsource(email_stream),
                "list_models_from_api": inspect.getsource(list_models_from_api),
                "models":              MODELS,
                "system_prompt":       SYSTEM_PROMPT,
                "execute_search":      inspect.getsource(execute_search),
                "parasail_agent":      inspect.getsource(ParasailAgent),
                "_sse_ask":            inspect.getsource(Handler._sse_ask),
                "_sse_direct":         inspect.getsource(Handler._sse_direct),
            })
        elif self.path == "/api/models":
            self._json(_model_catalog_payload(get_live_model_ids()))
        else:
            self.send_error(404)

    def _read_body(self):
        """Read and parse the POST body. Returns parsed dict or None on error (sends 400)."""
        try:
            length = min(int(self.headers.get("Content-Length", 0)), MAX_REQUEST_BODY)
        except ValueError:
            self._json({"error": "invalid Content-Length"}, 400)
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self._json({"error": "invalid request body"}, 400)
            return None

    def do_POST(self):
        if self.path == "/api/ask":
            body = self._read_body()
            if body is None:
                return
            model_key = _resolve_model_key(body.get("model"))
            question = body.get("question", "").strip()
            if not question:
                self._json({"error": "question required"}, 400)
                return
            search_enabled = body.get("search_enabled", True)
            logger.info("ask  model=%s search=%s q=%r", model_key, search_enabled, question[:80])
            self._sse_ask(model_key, question, search_enabled=search_enabled)
        elif self.path == "/api/chat":
            body = self._read_body()
            if body is None:
                return
            model_key = _resolve_model_key(body.get("model"))
            question = body.get("question", "").strip()
            prior_messages = body.get("prior_messages", [])
            search_enabled = body.get("search_enabled", True)
            if not question:
                self._json({"error": "question required"}, 400)
                return
            logger.info("chat model=%s search=%s turns=%d q=%r", model_key, search_enabled, len(prior_messages)//2, question[:80])
            self._sse_ask(model_key, question, prior_messages=prior_messages, search_enabled=search_enabled)
        elif self.path == "/api/compare":
            body = self._read_body()
            if body is None:
                return
            model_key = _resolve_model_key(body.get("model"))
            question = body.get("question", "").strip()
            search_enabled = bool(body.get("search_enabled", True))
            closed_key = body.get("closed_model", "gpt-5.6-sol")
            if not question:
                self._json({"error": "question required"}, 400)
                return
            skip_cache = bool(body.get("skip_cache") or body.get("no_cache"))
            prior_messages = body.get("prior_messages") or []
            self._sse_compare(model_key, question, search_enabled, closed_key, skip_cache=skip_cache, prior_messages=prior_messages)
        elif self.path == "/api/pipeline/enrich":
            body = self._read_body()
            if body is None:
                return
            company = body.get("company", "").strip()
            if not company:
                self._json({"error": "company required"}, 400)
                return
            num_results = int(body.get("num_results", 5))
            livecrawl   = bool(body.get("livecrawl", False))
            logger.info("pipeline/enrich company=%r num=%d livecrawl=%s", company, num_results, livecrawl)
            ydc_key = os.environ.get("YDC_API_KEY", "")
            result = enrich(company, ydc_key, num_results=num_results, livecrawl=livecrawl)
            result["ydc_cost"] = YDC_SEARCH_COST_PER_CALL
            self._json(result)

        elif self.path == "/api/pipeline/brief":
            body = self._read_body()
            if body is None:
                return
            company = body.get("company", "").strip()
            hits    = body.get("hits", [])
            model_key = body.get("model", "")
            model_cfg = MODELS.get(model_key, {})
            model_id  = model_cfg.get("model_id") or model_key
            logger.info("pipeline/brief company=%r model=%s", company, model_id)
            self._sse_llm_stream(brief_stream(company, hits, model_id), pricing=model_cfg.get("pricing"))

        elif self.path == "/api/pipeline/email":
            body = self._read_body()
            if body is None:
                return
            company = body.get("company", "").strip()
            brief   = body.get("brief", "")
            model_key = body.get("model", "")
            model_cfg = MODELS.get(model_key, {})
            model_id  = model_cfg.get("model_id") or model_key
            logger.info("pipeline/email company=%r model=%s", company, model_id)
            self._sse_llm_stream(email_stream(company, brief, model_id), pricing=model_cfg.get("pricing"))

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _serve_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_event(self, data: dict):
        line = f"data: {json.dumps(data)}\n\n"
        self.wfile.write(line.encode())
        self.wfile.flush()

    def _sse_ask(self, model_key: str, question: str, prior_messages: list = None, search_enabled: bool = True):
        model_cfg = MODELS.get(model_key)
        if not model_cfg:
            self._json({"error": f"unknown model: {model_key}"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._cors_headers()
        self.end_headers()

        model_id = model_cfg["model_id"]
        self._send_event({"event": "init", "model": model_id, "display_name": model_cfg["display_name"]})

        if not search_enabled:
            self._sse_direct(model_id, model_cfg, question, prior_messages or [])
            return

        for attempt in range(2):
            try:
                agent = ParasailAgent(model=model_id)
                for event in agent.stream(question, prior_messages=prior_messages or []):
                    if event.get("event") == "done":
                        stats = event.get("stats", {})
                        costs = calculate_costs(stats, model_cfg["pricing"])
                        tb = stats.get("token_breakdown", {})
                        event["costs"] = {
                            "total":             round(costs["total"], 6),
                            "llm":               round(costs["llm"], 6),
                            "search":            round(costs["search"], 6),
                            "livecrawl_overage": round(costs["livecrawl_overage"], 6),
                            "total_fmt":         format_cost(costs["total"]),
                        }
                        event["token_breakdown"] = {
                            "input":          tb.get("input", 0),
                            "output":         tb.get("output", 0),
                            "search_context": tb.get("search_context", 0),
                        }
                        event["pricing"] = _event_pricing(model_cfg["pricing"])
                        logger.info(
                            "done model=%s tokens=%s searches=%s cost=%s",
                            model_id,
                            stats.get("tokens_used", "?"),
                            stats.get("search_calls", "?"),
                            format_cost(costs["total"]),
                        )
                    elif event.get("event") == "tool_call":
                        logger.debug("search round=%s q=%r", event.get("round"), event.get("query", "")[:60])
                    self._send_event(event)
                break  # clean exit — no retry needed

            except BrokenPipeError:
                logger.info("client disconnected (stop or navigate away)")
                break

            except Exception as e:
                err_str = str(e)
                is_429 = _is_parasail_overload(e)

                if is_429 and attempt == 0:
                    logger.warning("429 from Parasail on attempt 1 — retrying in %ds", RETRY_DELAY_S)
                    try:
                        self._send_event({
                            "event": "retrying",
                            "message": f"Parasail is busy — retrying in {RETRY_DELAY_S}s…",
                            "after_s": RETRY_DELAY_S,
                        })
                    except BrokenPipeError:
                        break
                    time.sleep(RETRY_DELAY_S)
                    continue

                if is_429:
                    logger.error("429 persisted after retry — model=%s", model_id)
                    user_msg = "Parasail is busy right now. Please try again in a few moments."
                else:
                    logger.error("stream error model=%s: %s", model_id, e, exc_info=True)
                    user_msg = err_str
                try:
                    self._send_event({"event": "error", "message": user_msg})
                except BrokenPipeError:
                    pass
                break

    def _sse_compare(self, model_key: str, question: str, search_enabled: bool, closed_key: str, skip_cache: bool = False, prior_messages: list = None):
        open_cfg = MODELS.get(model_key)
        closed_cfg = CLOSED_MODELS.get(closed_key)
        if not open_cfg:
            self._json({"error": f"unknown model: {model_key}"}, 400)
            return
        if not closed_cfg:
            self._json({"error": f"unknown closed model: {closed_key}"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._cors_headers()
        self.end_headers()

        open_stats = None
        closed_stats = None
        try:
            self._send_event({
                "event": "compare_init",
                "model": open_cfg["model_id"],
                "display_name": open_cfg["display_name"],
                "closed_model": closed_cfg["model_id"],
                "closed_display_name": closed_cfg["display_name"],
            })
            open_error = None
            for attempt in range(2):
                try:
                    open_agent = ParasailAgent(model=open_cfg["model_id"])
                    for event in open_agent.stream(question, max_rounds=None if search_enabled else 0, prior_messages=prior_messages):
                        if event.get("event") == "done":
                            open_stats = event.get("stats", {})
                            event["costs"] = _event_costs(open_stats, open_cfg["pricing"])
                            event["token_breakdown"] = open_stats.get("token_breakdown", {})
                            event["pricing"] = _event_pricing(open_cfg["pricing"])
                        event["side"] = "open"
                        self._send_event(event)
                    open_answer = str((open_stats or {}).get("answer", ""))
                    if open_answer.startswith("Error:") and _is_parasail_overload(open_answer) and attempt == 0:
                        logger.warning("429 from Parasail compare open run — retrying in %ds", RETRY_DELAY_S)
                        self._send_event({
                            "event": "retrying",
                            "side": "open",
                            "message": f"Parasail is busy — retrying in {RETRY_DELAY_S}s…",
                            "after_s": RETRY_DELAY_S,
                        })
                        open_stats = None
                        time.sleep(RETRY_DELAY_S)
                        continue
                    break
                except BrokenPipeError:
                    raise
                except Exception as exc:
                    if _is_parasail_overload(exc) and attempt == 0:
                        logger.warning("429 from Parasail compare open run — retrying in %ds", RETRY_DELAY_S)
                        self._send_event({
                            "event": "retrying",
                            "side": "open",
                            "message": f"Parasail is busy — retrying in {RETRY_DELAY_S}s…",
                            "after_s": RETRY_DELAY_S,
                        })
                        time.sleep(RETRY_DELAY_S)
                        continue
                    open_error = exc
                    break

            open_answer = str((open_stats or {}).get("answer", ""))
            if open_error or open_stats is None or open_answer.startswith("Error:"):
                if open_error:
                    logger.error("compare open run failed: %s", open_error, exc_info=True)
                else:
                    logger.error("compare open run returned an error: %s", open_answer)
                if _is_parasail_overload(open_error or open_answer):
                    user_msg = "Parasail is busy right now. Please try again in a few moments."
                else:
                    user_msg = str(open_error or open_answer or "Open model returned no response")
                self._send_event({"event": "error", "message": user_msg})
                return

            cache_key = _cache_key(question, closed_cfg["model_id"], search_enabled)
            cached = None if skip_cache else _closed_cache_get(cache_key)
            closed_source = "cached" if cached else None
            if cached:
                closed_stats = cached["stats"]
                self._send_event({
                    "event": "answer",
                    "side": "closed",
                    "text": closed_stats.get("answer", ""),
                    "sources": closed_stats.get("sources", []),
                })
                self._send_event({
                    "event": "done",
                    "side": "closed",
                    "stats": closed_stats,
                    "costs": cached["costs"],
                    "token_breakdown": closed_stats.get("token_breakdown", {}),
                    "pricing": _event_pricing(closed_cfg["pricing"]),
                })
            elif os.getenv("ALLOW_LIVE_CLOSED", "true").lower() != "false" and os.environ.get("OPENAI_API_KEY"):
                closed_agent = ClosedModelAgent(model=closed_cfg["model_id"])
                live_events = []
                live_error = None
                try:
                    for event in closed_agent.stream(question, max_rounds=None if search_enabled else 0, prior_messages=prior_messages):
                        if event.get("event") == "done":
                            closed_stats = event.get("stats", {})
                        event["side"] = "closed"
                        live_events.append(event)
                except Exception as exc:
                    live_error = exc

                has_live_answer = (
                    closed_stats is not None
                    and not str(closed_stats.get("answer", "")).startswith("Error:")
                )
                if has_live_answer and live_error is None:
                    closed_source = "live"
                    for event in live_events:
                        if event.get("event") == "done":
                            event["costs"] = _event_costs(closed_stats, closed_cfg["pricing"])
                            event["token_breakdown"] = closed_stats.get("token_breakdown", {})
                            event["pricing"] = _event_pricing(closed_cfg["pricing"])
                        self._send_event(event)
                    if not skip_cache:
                        _closed_cache_put(cache_key, {
                            "stats": closed_stats,
                            "costs": _event_costs(closed_stats, closed_cfg["pricing"]),
                        })
                else:
                    if live_error:
                        logger.warning("closed model failed; using projected fallback: %s", live_error)
                    else:
                        logger.warning("closed model returned an error; using projected fallback")
                    closed_source = "projected"
                    closed_stats = _projected_closed_stats(open_stats, closed_cfg)
                    closed_costs = _event_costs(closed_stats, closed_cfg["pricing"])
                    self._send_event({
                        "event": "answer",
                        "side": "closed",
                        "text": closed_stats["answer"],
                        "sources": [],
                    })
                    self._send_event({
                        "event": "done",
                        "side": "closed",
                        "stats": closed_stats,
                        "costs": closed_costs,
                        "token_breakdown": closed_stats["token_breakdown"],
                        "pricing": _event_pricing(closed_cfg["pricing"]),
                    })
            else:
                closed_source = "projected"
                closed_stats = _projected_closed_stats(open_stats, closed_cfg)
                closed_costs = _event_costs(closed_stats, closed_cfg["pricing"])
                self._send_event({
                    "event": "answer",
                    "side": "closed",
                    "text": closed_stats["answer"],
                    "sources": [],
                })
                self._send_event({
                    "event": "done",
                    "side": "closed",
                    "stats": closed_stats,
                    "costs": closed_costs,
                    "token_breakdown": closed_stats["token_breakdown"],
                    "pricing": _event_pricing(closed_cfg["pricing"]),
                })

            open_costs = _event_costs(open_stats, open_cfg["pricing"])
            closed_costs = _event_costs(closed_stats, closed_cfg["pricing"])
            open_llm = open_costs["llm"]
            closed_llm = closed_costs["llm"]
            self._send_event({
                "event": "comparison",
                "open_cost": open_costs,
                "closed_cost": closed_costs,
                "open_inference_cost": open_llm,
                "closed_inference_cost": closed_llm,
                "open_token_breakdown": open_stats.get("token_breakdown", {}),
                "closed_token_breakdown": closed_stats.get("token_breakdown", {}),
                "open_latency_ms": open_stats.get("latency_ms", 0),
                "closed_latency_ms": closed_stats.get("latency_ms", 0),
                "savings_multiple": round(closed_llm / open_llm, 2) if open_llm else 0,
                "closed_answer": closed_stats.get("answer", ""),
                "closed_source": closed_source,
                "open_stats": open_stats,
                "closed_stats": closed_stats,
                "closed_pricing": closed_cfg["pricing"],
                "intelligence": closed_cfg.get("intelligence"),
            })
        except BrokenPipeError:
            logger.info("compare client disconnected")
        except Exception as e:
            logger.error("compare stream error: %s", e, exc_info=True)
            try:
                self._send_event({"event": "error", "message": str(e)})
            except BrokenPipeError:
                pass


    def _sse_direct(self, model_id: str, model_cfg: dict, question: str, prior_messages: list):
        """Direct LLM call with no tools — used when You.com Search is disabled."""
        from openai import OpenAI
        from models import PARASAIL_BASE_URL
        client = OpenAI(
            api_key=os.environ.get("PARASAIL_API_KEY", ""),
            base_url=PARASAIL_BASE_URL,
            timeout=120.0,
        )
        messages = list(prior_messages) + [{"role": "user", "content": question}]
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages, max_tokens=1500, stream=True,
            )
            answer = ""
            input_tokens = output_tokens = 0
            for chunk in resp:
                delta = chunk.choices[0].delta if chunk.choices else None
                text = (delta.content or "") if delta else ""
                if text:
                    answer += text
                    self._send_event({"event": "token", "text": text})
                if chunk.usage:
                    input_tokens  = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
            self._send_event({"event": "answer", "text": answer, "sources": []})
            costs = calculate_costs(
                {"tokens_used": input_tokens + output_tokens, "search_calls": 0,
                 "token_breakdown": {"input": input_tokens, "output": output_tokens, "search_context": 0}},
                model_cfg["pricing"],
            )
            self._send_event({
                "event": "done",
                "stats": {
                    "tokens_used": input_tokens + output_tokens,
                    "search_calls": 0,
                    "latency_ms": round((time.time() - t0) * 1000),
                    "connect_ms": 0,
                    "model_confirmed": model_id,
                    "token_breakdown": {"input": input_tokens, "output": output_tokens, "search_context": 0},
                },
                "costs": {
                    "total": round(costs["total"], 6),
                    "llm":   round(costs["llm"],   6),
                    "search": 0,
                    "livecrawl_overage": 0,
                    "total_fmt": format_cost(costs["total"]),
                },
                "token_breakdown": {"input": input_tokens, "output": output_tokens, "search_context": 0},
                "pricing": {
                    "input_per_m":     model_cfg["pricing"]["input"],
                    "output_per_m":    model_cfg["pricing"]["output"],
                    "search_per_call": 0,
                },
            })
        except BrokenPipeError:
            pass
        except Exception as e:
            logger.error("direct llm error model=%s: %s", model_id, e)
            try:
                self._send_event({"event": "error", "message": str(e)})
            except BrokenPipeError:
                pass

    def _sse_llm_stream(self, generator, pricing: dict = None):
        """Stream token events from a pipeline_agent generator, injecting pricing into done event."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._cors_headers()
        self.end_headers()
        try:
            for ev in generator:
                if ev.get("event") == "done" and pricing:
                    ev["pricing"] = {
                        "input_per_m":     pricing.get("input", 0),
                        "output_per_m":    pricing.get("output", 0),
                    }
                    ev["ydc_cost_per_search"] = YDC_SEARCH_COST_PER_CALL
                self._send_event(ev)
        except BrokenPipeError:
            logger.info("pipeline client disconnected")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", type=str, default=HOST)
    args = parser.parse_args()

    server = ThreadedHTTPServer((args.host, args.port), Handler)
    logger.info("Parasail playground → http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
