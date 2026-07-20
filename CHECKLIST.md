# Pre-checkin Validation Checklist

Run through these before sharing the repo with a partner.

## Environment
- [ ] `.env` is NOT committed (verify with `git status`)
- [ ] `.env.example` has placeholder values only — no real API keys
- [ ] `CODE_REVIEW.md` and `BACKLOG.md` are in `.gitignore` and not tracked

## Startup
- [ ] `pip install -r requirements.txt` completes cleanly in a fresh venv
- [ ] `cp .env.example .env` + real keys → `python3 server.py` starts without errors
- [ ] `http://localhost:8091` loads in browser

## Grounded Inference tab
- [ ] Ask a question → search runs → answer streams → stats bar shows Tokens / Searches / Cost
- [ ] Latency appears inline next to the search bar after completion
- [ ] Toggle You.com off → re-ask → no search calls, direct LLM answer
- [ ] "Code & Prompt" drawer opens and shows source + prompt tabs
- [ ] Cost popup opens when clicking the cost cell

## Research Chat tab
- [ ] Send a message → hero collapses, response streams, stats bar appears
- [ ] Follow-up message uses conversation history (model refers to prior turn)
- [ ] "New Chat" button resets thread and re-shows hero
- [ ] Stop button cancels in-flight stream

## Outreach Pipeline tab
- [ ] Enter a company name → Step 1 (You.com) runs, search results appear
- [ ] "Surface Sales Signals" → Step 2 (brief) streams
- [ ] "Draft Pitch Email" → Step 3 (email) streams
- [ ] Re-run button on Step 1 re-fetches without reloading the page
- [ ] Pipeline cost breakdown popup shows per-stage costs

## Config / Models tab
- [ ] Model list loads and shows pricing
- [ ] Config panel renders without errors

## Security spot-check
- [ ] Try entering `<img src=x onerror=alert(1)>` as a company name → must render as escaped text, no alert
- [ ] `EXPOSE_SOURCE=true` → View Source button visible; `EXPOSE_SOURCE=false` → button hidden
