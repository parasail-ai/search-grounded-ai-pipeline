# Search-Grounded AI Pipeline

Open-weight LLMs have a hard knowledge cutoff — typically 12 to 20 months behind today. This repo shows how to close that gap at inference time by giving the model live web search as a tool, and how to wire that pattern into a real business workflow.

The inference provider here is [Parasail](https://parasail.io) — a cloud GPU platform hosting open-weight models via an OpenAI-compatible API. The search layer is [You.com Search API](https://api.you.com). Both are swappable.

---

## What's in the demo

Three tabs, each showing a different shape of the same pattern:

**Grounded Inference** — Ask any question. The LLM decides when to call You.com as a tool, retrieves live results, and reasons over them before answering. Toggle You.com Web Search off to see the model answer from training data alone — the difference is the point.

**Research Chat** — Multi-turn conversation with the same grounded loop. Prior turns are passed as context so the model builds on what it already found across messages.

**Outreach Pipeline** — A three-stage agentic pipeline: You.com fetches live company intel → one LLM distills it into sales signals → a second LLM drafts a personalized outreach email. End-to-end from a company name in under 30 seconds.

---

## How it works

When You.com Web Search is on:

```
User question
    → LLM receives question + web_search tool definition
    → LLM calls web_search(query="...")
    → You.com Search API returns live results
    → Results injected into LLM context
    → LLM reasons over results and answers
    → Repeat if LLM decides another search round is needed
```

When You.com Web Search is off:

```
User question
    → Direct POST to /v1/chat/completions
    → LLM answers from training data only
```

The LLM never sees the raw API — it just calls a tool named `web_search`. The tool schema, execution, and result formatting all live in `search_tool.py`.

---

## Prerequisites

- Python 3.11+
- A [Parasail API key](https://parasail.io) (inference)
- A [You.com API key](https://api.you.com) (search)

---

## Quickstart

```bash
git clone https://github.com/pc1438/search-grounded-ai-pipeline
cd search-grounded-ai-pipeline

pip install -r requirements.txt

cp .env.example .env
# Add your PARASAIL_API_KEY and YDC_API_KEY to .env

python3 server.py
# Open http://localhost:8091
```

**`EXPOSE_SOURCE`** — When set to `true` (the default), a "View Source" button in the UI lets viewers inspect the full server code and prompts in-browser. This is intentional for demos — it shows partners exactly how the pipeline works. Set it to `false` if you're running on a public URL and don't want to expose your server code.

```
EXPOSE_SOURCE=false  # disable for shared/cloud deployments
```

---

## Swapping the inference provider

The agent layer uses the OpenAI-compatible `chat.completions` API. Any provider that speaks that format works — Together AI, Fireworks, Groq, Anyscale, or your own vLLM deployment.

Change two lines in `agents/parasail_agent.py`:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-key",
    base_url="https://your-provider.com/v1",  # ← change this
)
```

And update the model registry in `models.py` with your provider's model IDs and pricing.

---

## Project structure

| File | What it does |
|------|-------------|
| `server.py` | HTTP server — routes requests, streams SSE events to the UI |
| `index.html` | Single-page frontend — all three tabs, no build step |
| `search_tool.py` | You.com Search API wrapper — single source of truth for the tool schema, system prompt, and `execute_search()` |
| `base_agent.py` | Tool-use loop — the `OpenAICompatibleAgent` class that drives the search → LLM → search cycle |
| `models.py` | Model registry — display names, model IDs, and pricing for the Parasail-hosted models |
| `costs.py` | Cost calculation — token + search call pricing, formatted for the UI |
| `agents/parasail_agent.py` | Parasail-specific agent — client setup, model defaults, wires into `OpenAICompatibleAgent` |
| `agents/pipeline_agent.py` | Three-stage pipeline — `enrich()` (You.com search), `brief_stream()` (signals LLM), `email_stream()` (pitch LLM) |
| `logger_setup.py` | Structured logging configuration |
| `.env.example` | Environment variable template |

---

## Key design decisions

**`search_tool.py` as single source of truth** — the tool schema, system prompt, search constants, and `execute_search()` all live here. Agents import from it rather than defining their own. Changing search behavior means changing one file.

**Three-stage pipeline as separate endpoints** — `enrich`, `brief`, and `email` are independent server endpoints. The UI controls sequencing, not the server. Each stage can be inspected, re-run, or replaced independently.

**Non-streaming LLM calls in the pipeline** — Parasail's reasoning models put output in `model_extra['reasoning']`, not `delta.content`, making streaming unreliable. The pipeline collects the full response and re-emits it in 8-character chunks to preserve the streaming UX without losing output.

**Open-weight models via OpenAI-compatible API** — no vendor lock-in at the model layer. The same tool-use loop works with any provider that speaks `chat.completions`. Swapping providers is a two-line change.

---

## Running on a shared URL

If you expose this demo on a public or shared URL:

- Set `EXPOSE_SOURCE=false` in `.env` — the default is `true` for local demos, but you probably don't want to expose server source code publicly.
- Bind to localhost and put a reverse proxy (nginx, Caddy) in front with TLS.
- Add HTTP basic auth or an allow-list at the proxy layer — the server itself has no authentication.
- The `/api/source` endpoint is disabled when `EXPOSE_SOURCE=false`.
