# Open vs. Closed — Cost/Intelligence Tradeoff (Design Proposal)

Goal: across the Grounded Inference, Chat, and Outreach Pipeline tabs, make it obvious that
open-weight models on Parasail (e.g. GLM 5.2) deliver **comparable intelligence at a fraction of
the cost** vs. closed frontier models (e.g. latest GPT, Claude). Show it both **live** (per query)
and via a **backlog of pre-computed samples**.

---

## 1. The core mechanic — an honest cost engine

We can't call closed models here (not on Parasail, no keys). So we don't fake a closed run —
instead we take the **real token usage** from the open-model run and price it out under each
closed model's **published per-1M pricing**. Same workload, apples-to-apples:

```
projected_closed_cost = calculate_costs(open_run.token_stats, CLOSED_MODELS[m].pricing)
savings_multiple      = projected_closed_cost / open_run.actual_cost
```

`costs.calculate_costs(stats, pricing)` already does exactly this — we just feed it a different
price sheet. Every dollar shown for the open model is real; closed numbers are clearly labeled
"estimated at published API prices, same tokens," with a dated source footnote.

Pair cost with an **intelligence proxy** so "you keep intelligence" is backed by a number, not a
vibe. Proposal: use a single neutral index (e.g. Artificial Analysis Intelligence Index) that
covers GLM/MiniMax/Kimi/DeepSeek *and* GPT/Claude/Gemini, cited + dated. Alternative: a task-
relevant benchmark (coding for Kimi K2.7 Code, etc.).

### New data: a closed-model reference registry
Mirror `models.py` with a small, clearly-labeled **reference** table (not callable):
```python
CLOSED_MODELS = {
  "gpt-5":        {"display": "GPT-5",          "pricing": {...}, "intelligence": NN, "vendor": "OpenAI"},
  "claude-opus":  {"display": "Claude Opus 4",  "pricing": {...}, "intelligence": NN, "vendor": "Anthropic"},
  # ...
}
```
Source of truth for closed pricing + scores is the main open question (see §5).

---

## 2. A shared component reused across all three tabs — "Parity Card"

One component, dropped in wherever a run finishes. Anatomy:

```
┌────────────────────────────────────────────────────────────┐
│  Same answer. 20× cheaper.                                   │
│                                                              │
│  GLM 5.2 (open)        ██                       $0.007       │
│  GPT-5 (closed)        ████████████████████     $0.140       │
│  ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔  cost / query │
│                                                              │
│  Intelligence   GLM 5.2 ▓▓▓▓▓▓▓▓�. 68   GPT-5 ▓▓▓▓▓▓▓▓▓ 71   │
│  (near-parity — you keep the smarts)                         │
│                                                              │
│  est. closed cost = same tokens × published price · src …    │
└────────────────────────────────────────────────────────────┘
```
Two bars tell the whole story: **cost** (dramatic gap) + **intelligence** (near-parity). Headline
is auto-generated from the multiple.

---

## 3. Per-tab treatment

### Tab 2 — Grounded Inference (live, single-shot)
- Today: stat bar (Tokens / Searches / Cost) + answer + sources.
- Add the **Parity Card** right under the stat bar after each run, comparing the model just used
  vs. 1–2 closed references.
- Optional: a "vs" toggle to pick which closed model to compare against.

### Tab 3 — Chat (live, compounding)
- Chat already has a running stat bar (Turns / Tokens / Searches / Cost).
- Add a **running savings tally**: "This session: $0.02 on GLM 5.2 · ~$0.41 on GPT-5 — you've
  saved $0.39 (20×) over 6 turns." The multi-turn context makes cost *compound*, so the gap grows
  as the conversation does — a great "at scale" narrative in miniature.

### Tab 4 — Outreach Pipeline (the scale/ROI story — hero candidate)
- This is batch: N companies × (enrich + brief + email). Perfect for the business case.
- Add a **batch total + projection**: "50 companies: $0.90 open vs $18 closed." Plus a
  scale slider — "at 1,000 companies/month → **$X/mo saved**" — turning per-query pennies into a
  budget-level number a buyer feels.

---

## 4. Backlog of samples (no spend required)
A **gallery of pre-computed runs** so the comparison is compelling instantly, even offline:
- Capture real token counts once per sample query, store as JSON (`samples/*.json`): question,
  answer, sources, token stats, open model used.
- Render sample cards; clicking one populates the answer area + Parity Card from stored data
  (closed costs computed live from the reference price sheet, so they stay consistent).
- Curate 4–6 across domains (news, competitive research, coding via Kimi K2.7 Code, an outreach
  batch) to show breadth.

---

## 5. Open questions (need your input before building)
1. **Which closed models to benchmark against?** You mentioned "fable" — I'm not sure which model
   that is (typo?). Proposed set: latest GPT (GPT-5.x), Claude (Opus/Sonnet), maybe Gemini.
   Confirm the list + which is the "primary" comparison.
2. **Intelligence metric + source of truth.** OK to use Artificial Analysis Intelligence Index
   (neutral, covers open + closed), cited + dated? Or a specific benchmark?
3. **Closed pricing source.** Use each vendor's public API price (dated), or do you have an
   internal Parasail comparison sheet (marketing-site `src/data/pricing`?) you'd rather trust?
4. **Hero emphasis.** Lead with live single-query parity (Tab 2), or the batch ROI projection
   (Tab 4)? Both get built; which is the headline?
5. **Live vs. backlog priority** for a first cut.
