"""
pipeline_agent.py — Account Intelligence Pipeline, 3 discrete stages.

Each stage is an independent function so the server can expose them as
separate endpoints and let the UI control when each step starts.

Stage 1  enrich(company)          → search results via You.com
Stage 2  brief(company, results)  → bullet intel brief via LLM (SSE)
Stage 3  email(company, brief)    → outreach email via LLM (SSE)
"""

import logging
import os
import re
import time

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

_PARASAIL_BASE_URL = "https://api.parasail.io/v1"
_YDC_SEARCH_URL    = "https://ydc-index.io/v1/search"
_SEARCH_TIMEOUT    = 20


def _parasail_client() -> OpenAI:
    key = os.environ.get("PARASAIL_API_KEY", "")
    if not key:
        raise ValueError("PARASAIL_API_KEY not set")
    return OpenAI(api_key=key, base_url=_PARASAIL_BASE_URL, timeout=60.0)


# ── Stage 1: Enrich ───────────────────────────────────────────────────────────

_LIVECRAWL_CHARS = 1500  # max chars of cleaned livecrawl prose per hit sent to the brief LLM


def _clean_markdown(text: str, max_chars: int = _LIVECRAWL_CHARS) -> str:
    # Strip markdown syntax and nav boilerplate so the brief LLM gets clean prose,
    # not raw markup or page navigation clutter.
    text = re.sub(r'<[^>]+>', ' ', text)                  # HTML tags (crawl bleed-through)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)           # image refs
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # links → link text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)  # bold/italic
    text = re.sub(r'`[^`]+`', '', text)                   # inline code
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)   # bullet markers
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)   # numbered lists
    text = re.sub(r'^>{1,}\s*', '', text, flags=re.MULTILINE)   # blockquotes
    text = re.sub(r'-{3,}|\*{3,}|_{3,}', '', text)       # horizontal rules
    # Drop short lines that look like nav labels (< 40 chars, no sentence punctuation)
    lines = [l for l in text.splitlines() if len(l.strip()) >= 40 or any(c in l for c in '.,:;!?')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()        # collapse blank lines
    return text[:max_chars].rstrip() + ('…' if len(text) > max_chars else '')


def enrich(company: str, ydc_key: str, num_results: int = 5, livecrawl: bool = False) -> dict:
    # You.com Smart API: news + web results in one call, deduplicated by URL
    """
    Search You.com for live company intel.

    Returns:
        {
            "company": str,
            "hits": [{"title", "url", "description", "page_age"}, ...],
            "latency_ms": int,
            "error": str | None,
        }
    """
    today = time.strftime("%B %d, %Y")
    # Anchor the query to today's date so the model doesn't surface stale results
    query = (
        f"{company} news funding product launch hiring 2026 "
        f"recent last 4 weeks as of {today}"
    )
    t0 = time.time()
    try:
        params = {"query": query, "num_web_results": max(1, min(num_results, 20))}
        if livecrawl:
            params["livecrawl"] = "web"
            params["livecrawl_formats"] = "markdown"
        resp = requests.get(
            _YDC_SEARCH_URL,
            headers={"X-API-Key": ydc_key},
            params=params,
            timeout=_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # Response shape: {"results": {"news": [...], "web": [...]}, "metadata": {...}}
        results = data.get("results", {})
        # When livecrawl is on, web hits carry contents.markdown; news hits never do.
        # Prioritize web hits so livecrawl content isn't crowded out by news results.
        web_hits  = results.get("web",  [])
        news_hits = results.get("news", [])
        hits = (web_hits + news_hits) if livecrawl else (news_hits + web_hits)
        # Deduplicate by URL
        seen, deduped = set(), []
        for h in hits:
            u = h.get("url", "")
            if u and u not in seen:
                seen.add(u)
                description = h.get("description", "")
                hit = {
                    "title":       h.get("title", ""),
                    "url":         u,
                    "description": description,
                    "page_age":    h.get("page_age", ""),
                }
                if livecrawl:
                    contents = h.get("contents") or {}
                    full = contents.get("markdown") or contents.get("html") or ""
                    if full:
                        # Strip markdown syntax + nav boilerplate, then truncate to _LIVECRAWL_CHARS
                        hit["livecrawl_content"] = _clean_markdown(full)
                deduped.append(hit)

        elapsed = round((time.time() - t0) * 1000)
        logger.info("enrich company=%r hits=%d livecrawl=%s ms=%d", company, len(deduped), livecrawl, elapsed)
        return {"company": company, "hits": deduped[:num_results], "latency_ms": elapsed,
                "livecrawl": livecrawl, "num_results": num_results, "error": None}

    except Exception as e:
        logger.error("enrich failed company=%r: %s", company, e)
        return {"company": company, "hits": [], "latency_ms": 0, "error": str(e)}


# ── Stage 2: Brief ────────────────────────────────────────────────────────────

_BRIEF_SYSTEM = (
    "You are a sales intelligence analyst working for Parasail — an AI inference cloud. "
    "Parasail sells: serverless open-model inference (2M+ Hugging Face models, OpenAI-compatible API), "
    "dedicated endpoints, batch inference, and flexible GPU capacity — all at up to 30x lower cost than "
    "hyperscalers, with no hidden quantization, same-day deployments, and a dedicated Slack support channel. "
    "Ideal customers are AI-native teams overpaying OpenAI/Anthropic, teams managing fragmented inference vendors, "
    "or teams needing open-weight model variety without GPU lock-in.\n\n"
    "From the search results, produce 4–6 bullet points that surface SIGNALS relevant to a Parasail sales pitch: "
    "AI/ML initiatives underway, LLM or inference spend, cloud cost pressures, open-source model usage, "
    "recent AI product launches, hiring for ML engineers, or partnerships with AI vendors. "
    "Each bullet starts with '•' on its own line. "
    "Include specific figures, dates, and names. Tie each point to why it matters for Parasail. "
    "No preamble, no explanation, no headers. "
    "Do not narrate your reasoning. Do not say what you are about to do. "
    "First character of your response must be '•'."
)


_CHUNK_SIZE = 8  # characters per synthetic SSE chunk when replaying non-streaming response

def _llm_stream(system: str, user: str, model: str, label: str, start_marker: str = None):
    # Uses stream=False: Parasail's reasoning models put output in model_extra['reasoning'],
    # not delta.content, so streaming chunks are unreliable. We collect the full response
    # then re-emit in 8-char chunks to preserve the streaming UX in the browser.
    """
    Calls the LLM without streaming (works with reasoning models), then re-streams
    the response to the client in small chunks.
    start_marker: if set, everything before the first occurrence is stripped
                  (handles reasoning models that narrate before producing output).
    Yields: {"event": "token", "text": str} … then {"event": "done", …}
    """
    client = _parasail_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=1500,
            temperature=0.3,
            stream=False,
        )
        msg  = resp.choices[0].message
        # GPT-OSS 20B (reasoning model): content is None; output is in model_extra['reasoning']
        text = msg.content or (msg.model_extra or {}).get("reasoning", "") or ""
        preamble = ""
        if start_marker and start_marker in text:
            idx = text.index(start_marker)
            # Everything before the marker is the model's chain-of-thought narration
            preamble = text[:idx].strip()
            text = text[idx:]
        # Always emit thinking event so the drawer tab is available
        yield {"event": "thinking", "text": preamble}
        usage = resp.usage
        input_tokens  = getattr(usage, "prompt_tokens",     0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        logger.info("%s done input=%d output=%d", label, input_tokens, output_tokens)
        # Re-stream in small chunks so the client sees progressive output
        for i in range(0, len(text), _CHUNK_SIZE):
            yield {"event": "token", "text": text[i:i + _CHUNK_SIZE]}
        yield {"event": "done", "input_tokens": input_tokens, "output_tokens": output_tokens}
    except Exception as e:
        logger.error("%s failed: %s", label, e)
        yield {"event": "error", "message": str(e)}


def brief_stream(company: str, hits: list, model: str):
    # Surfaces AI/ML signals relevant to a Parasail sales pitch from live search results.
    # Uses livecrawl_content when available (cleaned full-page prose), else falls back to description.
    context_parts = [
        f"[{i}] {h.get('title','')}\n{h.get('livecrawl_content') or h.get('description','')}\n{h.get('url','')}"
        for i, h in enumerate(hits[:6], 1)
    ]
    context = "\n\n".join(context_parts) or "No search results available."
    yield from _llm_stream(
        system=_BRIEF_SYSTEM,
        user=f"Company: {company}\n\nSearch results:\n{context}",
        model=model,
        label=f"brief[{company}]",
        start_marker="•",
    )


# ── Stage 3: Email ────────────────────────────────────────────────────────────

_EMAIL_SYSTEM = (
    "You are writing a cold outreach email on behalf of Parasail (parasail.io) — an AI inference cloud. "
    "Parasail's pitch: serverless inference across 2M+ open-weight models, OpenAI-compatible API, "
    "up to 30x cheaper than hyperscalers, no hidden quantization, same-day deployments, dedicated Slack support. "
    "Flexible tiers: Serverless, Dedicated Serverless, Dedicated GPU, and Batch (lowest cost for evals/embeddings). "
    "Do not narrate your reasoning. Do not explain what you are about to write. "
    "Output ONLY the email, starting immediately with 'Subject:'. "
    "Format exactly:\n"
    "Subject: <subject line referencing something specific about this company>\n\n"
    "<opener: 1-2 sentences referencing a specific AI signal from the intel brief — show you did your homework>\n\n"
    "<value prop: 2-3 sentences connecting their specific situation to what Parasail solves — cost, scale, model variety, or speed>\n\n"
    "<CTA: one concrete ask — 15-min call, free trial, or benchmark offer>\n\n"
    "No fluff. No sign-off placeholder. Sound human, not templated. "
    "First word of your response must be 'Subject'."
)


def email_stream(company: str, brief: str, model: str):
    # Writes a Parasail cold-outreach email grounded in the signals from brief_stream
    yield from _llm_stream(
        system=_EMAIL_SYSTEM,
        user=f"Company: {company}\n\nIntel Brief:\n{brief}",
        model=model,
        label=f"email:{company}",
        start_marker="Subject:",
    )
