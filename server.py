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
from agents.pipeline_agent import enrich, brief_stream, email_stream, _llm_stream
from search_tool import SYSTEM_PROMPT, execute_search
from models import list_models_from_api, MODELS
from costs import calculate_costs, format_cost, YDC_SEARCH_COST_PER_CALL

RETRY_DELAY_S = 5          # seconds to wait before one 429 retry
MAX_REQUEST_BODY = 512_000  # 512 KB — sufficient for any question/brief payload

APP_DIR = Path(__file__).parent
PORT = 8091
EXPOSE_SOURCE = os.getenv("EXPOSE_SOURCE", "true").lower() == "true"


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
            self._json({
                "models": [
                    {
                        "key": k,
                        "model_id": v["model_id"],
                        "display_name": v["display_name"],
                        "provider": v["provider"],
                        "pricing": v["pricing"],
                    }
                    for k, v in MODELS.items()
                ]
            })
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
            model_key = body.get("model", "gpt-oss-20b")
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
            model_key = body.get("model", "gpt-oss-20b")
            question = body.get("question", "").strip()
            prior_messages = body.get("prior_messages", [])
            search_enabled = body.get("search_enabled", True)
            if not question:
                self._json({"error": "question required"}, 400)
                return
            logger.info("chat model=%s search=%s turns=%d q=%r", model_key, search_enabled, len(prior_messages)//2, question[:80])
            self._sse_ask(model_key, question, prior_messages=prior_messages, search_enabled=search_enabled)
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
                        event["pricing"] = {
                            "input_per_m":     model_cfg["pricing"]["input"],
                            "output_per_m":    model_cfg["pricing"]["output"],
                            "search_per_call": YDC_SEARCH_COST_PER_CALL,
                        }
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
                is_429 = "429" in err_str or "overloaded" in err_str.lower()

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
                    continue  # attempt 2

                # Final failure
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
    args = parser.parse_args()

    server = ThreadedHTTPServer(("127.0.0.1", args.port), Handler)
    logger.info("Parasail playground → http://localhost:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
