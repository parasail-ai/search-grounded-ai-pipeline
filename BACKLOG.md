# Parasail × You.com Playground — Backlog

> Living doc. Add items as they come up; strike through or delete when shipped.

---

## Pipeline Demos (not yet built)

### 2. Competitive Landscape Monitor
**Pitch:** Give a product or company name → You.com surfaces recent news, job postings, pricing changes, reviews → model synthesizes a "what changed this week" digest.
**Audience:** Product, strategy, investor relations — not just sales.
**Why it's compelling:** Shows the time-to-value angle without indexing on token cost. A human analyst would spend hours on this; the pipeline does it in seconds.
**Model split:** Fast summarization model for the digest (low cost, high speed).

### 3. Research-to-Draft for Technical Teams
**Pitch:** Paste a GitHub repo URL or technical topic → You.com livecrawls docs, recent blog posts, Stack Overflow, release notes → model produces an onboarding doc or "what you need to know before touching this" brief.
**Audience:** Engineering — speaks directly to the hackathon attendees.
**Why it's compelling:** Demonstrates livecrawl specifically, not just search. Personal pain point for every developer joining a new codebase.
**Model split:** Larger context model for ingestion; lighter model for the structured output.

---

## Playground UX Improvements

### Pipeline tab
- [ ] Export all results as CSV or JSON (company, brief, email per row)
- [ ] Allow re-running individual steps (re-enrich, regenerate brief, regenerate email) without starting over
- [ ] Show token count and cost per step so users can see the "research model vs email model" cost difference
- [ ] Add a "Load example accounts" button so the demo is one-click for first-time users
- [ ] Let users customize the brief prompt and email prompt (advanced mode toggle)

### Ask tab
- [ ] Show TTFC as a sparkline or visual latency bar — harder to miss than a number
- [ ] "Compare models" mode: run the same question across two models side by side

### Chat tab
- [ ] Export full conversation as markdown
- [ ] Persistent sessions (save/load named conversations)

### General
- [ ] Dark mode (the CSS vars are already structured for it — just needs the `@media (prefers-color-scheme: dark)` block)
- [ ] Mobile layout pass — input bars and stat strips need responsive breakpoints
- [ ] Retry UI: surface the 5s countdown as an animated progress bar, not just a text notice

---

## Hackathon Narrative / Positioning

**Core message to land:**
> Neither Parasail nor You.com alone closes the loop. Parasail gives you cheap, fast, task-matched inference on open-weight models. You.com gives you live web context. Together you get an agent that knows what happened this morning *and* can reason about it — something a static LLM can't do and a plain search engine can't do.

**"Time-to-value" framing (from the team):**
- Don't lead with token cost. Lead with: *how long would a human analyst take to do this?*
- Account intelligence pipeline: 45–60 min of research → ~30 seconds
- Competitive monitor: weekly analyst report → on-demand, always fresh
- Technical research-to-draft: new engineer onboarding doc → generated before the first PR

**Model selection as a value lever:**
- The pipeline deliberately uses different models per stage — a cheap summarization model for the brief, a more capable generation model for the email
- This is a Parasail-specific story: you can mix and match open-weight models per task, which you can't do with a single closed-model API
- Show the cost delta in the UI to make it visceral

---

## Tech Debt / Cleanup

- [ ] `stat-cell-value` font sizing still inconsistent between Ask and Chat stat strips — do a visual pass
- [ ] The `_chatHistory` array grows unboundedly in a long chat session — add a max-turns cap or summarization
- [ ] Server has no request timeout — a slow Parasail response blocks the thread indefinitely; add a timeout on the agent stream
- [ ] `search_tool.py` INTEGRATION_INTERFACE global is never set in server.py — log line will always say `direct_api`
- [ ] `.env` has exposed API keys (flagged during session) — rotate both before sharing the repo
