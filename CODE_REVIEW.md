# Code Review — Parasail × You.com Search Playground

**Reviewer:** Senior staff engineer · full-codebase review
**Scope:** whole repo (~5.9k LoC source): Python `http.server` backend, agent layer, pipeline, costs, models, vanilla-JS frontend
**Method:** all Python files read line-by-line directly; `index.html` (3,533 lines) covered by a focused parallel pass. Every finding verified against source (grep) before inclusion.

---

## Verdict

This is **clean, purposeful hackathon code** — the architecture is deliberately layered (search tool → base agent → provider agent → server → UI), the `search_tool.py` single-source-of-truth pattern is sound, and the pipeline decomposition into three independent stages is correct. FINDINGS.md and BACKLOG.md show good engineering instincts: real quirks were documented as they were discovered.

The main issues cluster around **two themes**:

1. **`pipeline_agent.py` was written independently of `search_tool.py`** and now duplicates You.com search in a second, slightly divergent implementation — different timeout (20s vs 30s), different livecrawl char limit (1,500 vs 3,000), different query param name (`num_web_results` vs `count`). `search_tool.py` is explicitly documented as "SOURCE OF TRUTH" but the pipeline bypasses it entirely.

2. **`base_agent.py` was ported from a larger repo and not trimmed** — two agent classes (`AnthropicAgent`, `OpenAIResponsesAgent`) and several utilities in `search_tool.py` are fully implemented but never used in this repo. A `_get_agent_default_model` helper silently always fails because it reads a file from the parent repo that doesn't exist here.

Security posture is mostly fine for a localhost hackathon tool, but two issues are blockers before any shared or cloud deployment.

| # | Severity | Finding | Area |
|---|----------|---------|------|
| 1 | 🔴 Blocker | `/api/source` exposes full server source, system prompts, and pricing to any caller | Security |
| 2 | 🔴 Blocker | No request body size cap → OOM/DoS via oversized `Content-Length` | Security |
| 3 | 🟠 Major | 7 XSS injection points in `index.html` — unescaped user input, SSE fields, API data, LLM output all go into `innerHTML` | Security (frontend) |
| 4 | 🟠 Major | `pump()` in `ask()` swallows non-abort errors → UI permanently stuck | Correctness (frontend) |
| 5 | 🟠 Major | `pipeline_agent.py` re-implements You.com search instead of using `search_tool.py` — divergent config | Cohesion |
| 6 | 🟠 Major | `AnthropicAgent` + `OpenAIResponsesAgent` + `_get_agent_default_model` ported from a larger repo, unused here | Dead code / porting debt |
| 7 | 🟡 Minor | Duplicate imports of `ParasailAgent` and `MODELS` in `server.py` | Housekeeping |
| 8 | 🟡 Minor | `json.loads` in all 5 POST handlers unguarded — malformed body crashes the thread | Correctness |
| 9 | 🟡 Minor | Six dead exports in `search_tool.py` (porting debt) | Dead code |
| 10 | 🟡 Minor | `prior_messages` not in `BaseAgent.stream()` abstract signature | Design |
| 11 | 🟡 Minor | `enrich()` error return is missing keys present in the success return | Correctness |
| 12 | 🟡 Minor | `_llm_stream` docstring is unreachable (comment precedes it; string is a dead literal) | Docs |
| 13 | 🟡 Minor | `_PARASAIL_BASE_URL` defined identically in two agent files | Duplication |
| 14 | 🟡 Minor | SSE read loop copy-pasted 4× with inconsistent error handling | Cohesion (frontend) |
| — | ⬛ Known | `INTEGRATION_INTERFACE` never set from `server.py`; `.env` contains live keys | BACKLOG-tracked |
| — | ✅ | Notable strengths (see end) | — |

> **Frontend findings** (§9) are appended below the backend sections.

---

## 🔴 Blockers

### 1. `/api/source` exposes full server internals to any caller
`server.py:69-82` — The endpoint serializes and returns the complete Python source of:
- `_sse_ask`, `_sse_direct` (the full SSE handler logic and timeout values)
- `execute_search`, `brief_stream`, `email_stream`, `_llm_stream` (the LLM/search implementations)
- `ParasailAgent` (API key env var names, base URL, timeout)
- `list_models_from_api` (how the model list is fetched)
- `SYSTEM_PROMPT` and `MODELS` (the verbatim system prompt and all model IDs + pricing)

This is a deliberate transparency feature for the hackathon demo (show your code to the audience). That's fine in a controlled presentation — but there is no auth guard, so any user who finds the playground URL can call `GET /api/source` and get a complete map of the system. Before any shared or cloud deployment, this endpoint must either be removed or gated behind an admin check.

**Fix:** Remove the endpoint, or add a server-side toggle (e.g. `EXPOSE_SOURCE=false` in `.env`) and default it off.

### 2. No request body size cap across all POST handlers
`server.py:101,112,124,139,150` — every POST handler follows the pattern:
```python
length = int(self.headers.get("Content-Length", 0))
body = json.loads(self.rfile.read(length))
```
There is no cap on `length`. A request with `Content-Length: 104857600` (100 MB) causes the server to allocate 100 MB per request. `ThreadedHTTPServer` with `daemon_threads=True` spawns a thread per connection, so a few concurrent oversized requests exhaust memory. BACKLOG.md already calls out the related issue ("Server has no request timeout") — the body size problem is the same threat surface.

**Fix:** Cap `length` at a reasonable limit (e.g. 512 KB for question/brief payloads):
```python
MAX_BODY = 512_000
length = min(int(self.headers.get("Content-Length", 0)), MAX_BODY)
```
Also wrap `json.loads(self.rfile.read(length))` in a try/except (see §6).

---

## 🟠 Major

### 3. `pipeline_agent.py` re-implements You.com search, bypassing the single source of truth
`search_tool.py:1` explicitly documents itself as **"SOURCE OF TRUTH for shared constants"**. `pipeline_agent.py` ignores this and reimplements a second HTTP caller to the same endpoint, producing three silent divergences:

| Config | `search_tool.py` | `pipeline_agent.py` |
|--------|-----------------|---------------------|
| Timeout | `SEARCH_TIMEOUT_SECONDS = 30` | `_SEARCH_TIMEOUT = 20` |
| Livecrawl char limit | `MAX_LIVECRAWL_CHARS = 3000` | `_LIVECRAWL_CHARS = 1500` |
| Result count param | `count` | `num_web_results` |
| URL constant | `SEARCH_ENDPOINT` | `_YDC_SEARCH_URL` (same string, duplicated) |

The third difference (`count` vs `num_web_results`) isn't just a naming inconsistency — `search_tool.py`'s API docs show `count` as the canonical param; `num_web_results` is a different (likely SmartAPI) param name. One of these may be silently returning the wrong number of results.

The fix is to call `execute_search()` from `search_tool.py` inside `enrich()`, or at minimum extract the HTTP call into a shared helper and have both use the same constants. The `_clean_markdown()` / livecrawl processing in `enrich()` would remain pipeline-specific (it strips markdown for a different downstream purpose than the raw search path).

### 4. Dead agent classes and helpers ported from a larger repo
`base_agent.py` contains two fully-implemented agent classes that are never instantiated anywhere in this repo:

- **`AnthropicAgent`** (`base_agent.py:225-344`, ~120 lines) — Claude/Anthropic Messages API loop. No Anthropic agent file exists in `agents/`.
- **`OpenAIResponsesAgent`** (`base_agent.py:535-735`, ~200 lines) — OpenAI Responses API with `previous_response_id` chaining and parallel tool execution. No Responses API agent file exists in `agents/`.

`ParasailAgent` inherits only `OpenAICompatibleAgent` — the other two classes are dead.

Also dead: `_get_agent_default_model` (`base_agent.py:87-105`) — it reads `../comparison/pricing.json` (a path from the parent repo this code was ported from), which doesn't exist here. The function catches all exceptions and falls back silently, so it always returns the hardcoded string. It is never called anywhere in this repo. The docstring ("single source of truth for model selection") misleadingly implies it's functional.

**Fix:** Either delete these three constructs entirely (they're in `base_agent.py` but no code path reaches them), or add a comment making the "kept for future use" intent explicit. Delete `_get_agent_default_model` — it silently fails and no one calls it.

---

## 🟡 Minor

### 5. Duplicate imports in `server.py`
`server.py:32-39`:
```python
from agents.parasail_agent import ParasailAgent   # line 32
import inspect
from agents.pipeline_agent import enrich, brief_stream, email_stream, _llm_stream
from search_tool import SYSTEM_PROMPT, execute_search
from agents.parasail_agent import ParasailAgent   # line 36 — duplicate
from models import list_models_from_api, MODELS   # line 37
from costs import calculate_costs, format_cost, YDC_SEARCH_COST_PER_CALL
from models import MODELS                          # line 39 — duplicate
```
`ParasailAgent` and `MODELS` are both imported twice. No runtime error (Python deduplicates), but it signals a merge artifact and makes the import block harder to read. **Fix:** remove the duplicates on lines 36 and 39.

### 6. `json.loads` unguarded in all POST handlers
`server.py:102,113,125,140,151` — none of the five POST handlers wrap `json.loads(self.rfile.read(length))` in a try/except. A malformed body (non-JSON, truncated, wrong encoding) raises `json.JSONDecodeError` which propagates uncaught to `do_POST`, causing the server to return a garbled error instead of a clean 400. Same applies to `int(self.headers.get("Content-Length", ...))` if the header value is non-numeric.

**Fix:**
```python
try:
    body = json.loads(self.rfile.read(length))
except (json.JSONDecodeError, ValueError):
    self._json({"error": "invalid request body"}, 400)
    return
```

### 7. Six dead exports in `search_tool.py`
The following symbols are defined in `search_tool.py` but imported by zero other modules in this repo. They are porting artifacts from the parent codebase:

| Symbol | Lines | Description |
|--------|-------|-------------|
| `set_interface()` | 45–55 | Sets `INTEGRATION_INTERFACE` global; never called from `server.py` |
| `NATIVE_SEARCH_BASELINE_TOKENS` | 111 | Token baseline constant; never imported |
| `inject_citations_at_positions()` | 439–454 | Inserts `[N]` markers at char offsets; never called |
| `inject_citations()` | 457–462 | Replaces `[N]` with markdown links; never called |
| `make_progress_reporter()` | 465–473 | Closure factory for progress callbacks; never called |
| `make_elapsed_timer()` | 476–481 | Timer factory; never called |

Note: BACKLOG.md already flags `set_interface` specifically ("INTEGRATION_INTERFACE global is never set in `server.py`"). Removing or clearly marking all six avoids future confusion about what's live.

### 8. `prior_messages` not in `BaseAgent.stream()` abstract signature
`base_agent.py:183`:
```python
@abstractmethod
def stream(self, question: str, max_rounds: int = None) -> Generator[dict, None, None]:
```
`OpenAICompatibleAgent.stream()` (line 382) adds `prior_messages: list = None` — used by the Chat tab's multi-turn history. `BaseAgent` and `AnthropicAgent`/`OpenAIResponsesAgent` don't declare it. This means code typed as `BaseAgent` wouldn't know about `prior_messages`, and adding a new agent class from the base would silently not support Chat unless the developer reads `OpenAICompatibleAgent`'s implementation.

**Fix:** Add `prior_messages: list = None` to `BaseAgent.stream()` and document it.

### 9. `enrich()` error return is missing keys present in the success return
`pipeline_agent.py:123-128`:
```python
# Success:
return {"company": company, "hits": deduped[:num_results], "latency_ms": elapsed,
        "livecrawl": livecrawl, "num_results": num_results, "error": None}

# Failure:
return {"company": company, "hits": [], "latency_ms": 0, "error": str(e)}
```
The error path omits `livecrawl` and `num_results`. Any caller that accesses `result["livecrawl"]` or `result["num_results"]` on the error branch gets a `KeyError`. The server adds `result["ydc_cost"]` on line 135 after the call, which also means the error-path dict gets `ydc_cost` injected but is still missing the other two keys. **Fix:** include all keys in the error return.

### 10. `_llm_stream` docstring is a dead string literal
`pipeline_agent.py:153-157`:
```python
def _llm_stream(system: str, user: str, model: str, label: str, start_marker: str = None):
    # Uses stream=False: ...  ← this comment is the first line of the function body
    """                        ← this triple-quoted string is NOT the docstring; it's a dead literal
    Calls the LLM without streaming...
    """
```
In Python, the docstring must be the *first statement* in the function body. Because a `#` comment precedes the triple-quoted string, the string becomes a free-standing expression that is evaluated and discarded — `help(_llm_stream)` returns nothing. The intent was clearly to have both a comment (explaining the stream=False decision) and a docstring (describing the function). **Fix:** make the design comment part of the docstring, or move the docstring before the comment.

### 11. `_PARASAIL_BASE_URL` defined identically in two files
`agents/parasail_agent.py:42` and `agents/pipeline_agent.py:22` both define:
```python
_PARASAIL_BASE_URL = "https://api.parasail.io/v1"
```
If Parasail ever changes its base URL, only one file will get updated. **Fix:** define it once in `models.py` (which already owns Parasail-specific config) or a `config.py`, and import it in both agent files.

---

## Known / BACKLOG-tracked (not new findings)

These are already in BACKLOG.md or FINDINGS.md — listed here for completeness:

- **`INTEGRATION_INTERFACE` never set** — `server.py` never calls `set_interface(...)`, so all tool log entries record `"direct_api"` (the module-level default). Accidentally correct for now, but fragile.
- **`.env` contains live API keys** — gitignored and not committed, but BACKLOG.md flags the need to rotate before sharing the repo.
- **No server-side request timeout** — the `ThreadedHTTPServer` thread blocks indefinitely on slow Parasail responses. Mitigation: add a `timeout=` to `ParasailAgent.__init__`'s `OpenAI` client (it already has `timeout=120.0`, which helps) and ensure the thread pool doesn't starve. This is tracked in BACKLOG.md.

---

## §9. Frontend (`index.html`)

> Reviewed across all three tabs (Ask, Chat, Pipeline) — security, dead code, correctness, cohesion, and naming.

### XSS — 7 injection points, all from unescaped dynamic content into `innerHTML`

The file has a proper `_esc()` helper (verified in use in some places) but it is **not applied consistently**. Every finding below results in the corresponding data rendering as HTML, not as safe text.

**S-1 🔴 Major — `company` textarea value injected raw into `innerHTML`**
`~lines 2635, 2707, 2729, 2748, 2832, 2981`

`company` comes directly from `document.getElementById('pipeline-companies').value`. It is string-interpolated into template literals that are assigned to `.innerHTML` and passed to `insertAdjacentHTML`. A value like `<img src=x onerror=alert(1)>` executes immediately. The same unescaped variable also appears inside inline `onclick` handlers in the generated HTML:
```js
onclick="_rerunEnrich('${company}','${slug}')"
```
This allows a company input containing `'` or `)` to break out of the string and inject arbitrary JS.

**S-2 🔴 Major — SSE `ev.query`, `ev.search_num`, `ev.round`, params unescaped in `insertAdjacentHTML`**
`~lines 1958–1980 (Ask), 2465–2469 (Chat)`

Unescaped SSE fields from the server's tool-call and search-result events are assembled into HTML strings and inserted with `insertAdjacentHTML('beforeend', ...)`. If the You.com API returns a query containing HTML (or the server is MITM'd), it executes as markup.

**S-3 🔴 Major — SSE `ev.message` from retrying/error events unescaped**
`~lines 2087–2101 (Ask), 2492–2501, 2528–2530 (Chat)`
```js
results.insertAdjacentHTML('beforeend',
  `<div class="retrying-notice">${ev.message}</div>`);
// and:
bubble.innerHTML = `...<span>✕</span> ${ev.message}</div>`;
```
`ev.message` is a raw SSE string. Any HTML inside it renders as markup.

**S-4 🔴 Major — Source URLs unescaped in `href` and text (`javascript:` vector)**
`~lines 2013–2018`
```js
sourcesEl.innerHTML = ev.sources.map((url, i) =>
  `<a class="source-url" href="${url}">${url}</a>`
).join('');
```
`url` is API-supplied. A `javascript:` URL in `href` executes on click. `rel="noopener"` does not prevent this.

**S-5 🔴 Major — `renderMarkdown` passes raw HTML through to `innerHTML`**
`~lines 1768–1812 (definition), 1990 (Ask), 2487 (Chat)`

The markdown renderer applies bold/italic/code regex replacements without first HTML-escaping the input. Any `<script>`, `<img onerror=...>`, or other raw HTML in an LLM answer passes through untouched into `answerEl.innerHTML`. Because this renders every LLM answer, it's the highest-traffic execution path.

**S-6 🔴 Major — You.com hit fields (`title`, `description`, `url`, `page_age`) unescaped in pipeline cards**
`~lines 2682–2688, 2969–2974`

All four You.com result fields are string-interpolated directly into `innerHTML` template literals in the pipeline's search-result cards. `h.url` in `href` again allows `javascript:` injection; `h.title` allows script injection via an HTML-returning API response.

**S-7 🔴 Major — `_renderEmail` injects unescaped LLM output**
`~lines 2900–2907 (definition), 2870 (call)`
```js
function _renderEmail(raw) {
  return `<div class="email-body">${body}</div>`;  // body is raw LLM text
}
body.innerHTML = _renderEmail(fullText);
```
`white-space: pre-wrap` CSS gives visual safety but does not prevent HTML execution.

**S-8 🟡 Minor — `cfg.system_prompt` missing `_esc()`**
`~line 1683`
```js
document.getElementById('config-body').innerHTML =
  `...<div class="config-prompt">${cfg.system_prompt}</div>...`;
```
The system prompt is server-controlled, so risk is low today. But `_esc()` is used in adjacent config fields — this one was missed.

---

### Correctness bugs

**C-1 🟠 Major — Non-abort network error in `ask()` pump → UI permanently stuck**
`~lines 1756–1757`

`ask()` uses a recursive callback `pump()` to read the SSE stream:
```js
reader.read().then(({done, value}) => { ... pump(); })
  .catch(e => { if (e.name !== 'AbortError') throw e; });
```
Any non-`AbortError` (network drop, server reset, response body truncation) is re-thrown as an unhandled promise rejection. `finalize()` is never called — the Ask button stays disabled, the spinner keeps running, and the Stop button stays visible. The `chatSend`, `generateBrief`, and `generateEmail` equivalents all use `async/await` in a `try/finally` and correctly reach their respective `finalize` in every branch. This is the one outlier.

**C-2 🟡 Minor — `chatNew()` reset is missing `llm`, `search`, `livecrawl` fields**
`~lines 2349–2354`

`_chatTotals` is initialized with `{turns, input, ctx, output, searches, cost, llm, search, livecrawl}`. The reset in `chatNew()` omits `llm`, `search`, and `livecrawl`. Starting a new conversation leaves those three fields carrying values from the prior session — the cost popup breakdown is silently wrong.

**C-3 🟡 Minor — Stale `traceEl`/`answerEl`/`sourcesEl` after a second `ask()` call**
`~lines 1731, 1834, 1942–1944`

On each `ask()` call, `results.innerHTML = ''` detaches the old DOM nodes. `traceEl`, `answerEl`, and `sourcesEl` are module-level variables re-assigned only on the `init` SSE event. If any events from the prior (aborted) request arrive between the DOM clear and the new `init`, they write into detached nodes (silent failure). If `init` is skipped due to a network error before the first event, later events dereference null.

**C-4 🟡 Minor — `_dotDone(slug, 1)` called twice**
`~lines 2711, 2740`

Both `_enrichCompany` and `generateBrief` call `_dotDone(slug, 1)` for the same step dot. Idempotent but redundant — a copy-paste artifact.

---

### Dead code

**D-1 🟡 Minor — `toggleCardSection` defined but never called** (`~line 3088`)
The pipeline collapse behavior uses `togglePipeStep`. `toggleCardSection` has no call site.

**D-2 🟡 Minor — `.chat-searches-toggle` CSS dead** (`~line 525`, comment says "replaced by footer chips")

**D-3 🟡 Minor — `.streaming-border` CSS class never applied** (`~line 395`)
Live streaming uses `.streaming` (applied to `.chat-assistant-card`).

**D-4 🟡 Minor — `thinkingText` local assigned but never read** (`~lines 3387–3388`)

**D-5 🟡 Minor — `meta` dead local in `chatSend`** (`~line 2418`, comment acknowledges it)

**D-6 🟡 Minor — `_chatPricing = pricing` assigned twice in `updateChatStats`** (`~lines 2320, 2331`)

---

### Cohesion

**COH-1 🟡 Minor — SSE read loop copy-pasted 4×**
`~lines 1744 (ask), 2448 (chatSend), 2776 (generateBrief), 2858 (generateEmail)`

The same three-line SSE framing logic (`buf.split('\n\n')`, `buf = parts.pop()`, `part.startsWith('data:')`) is duplicated four times. Critically, `ask()` uses the older callback-based `pump()` style while the other three use `async/await` — the divergence is why C-1 exists. If the SSE framing ever needs a fix, four sites need updating.

**COH-2 🟡 Minor — `_renderEmail` called on every streaming token (O(n²) DOM writes)**
`~lines 2867–2871`

```js
if (ev.event === 'token') {
  fullText += ev.text;
  const body = document.getElementById(`pstep3-body-${slug}`);
  if (body) body.innerHTML = _renderEmail(fullText);
}
```
`_renderEmail` rebuilds the full innerHTML on each 8-character chunk. For a 500-word email (~3,000 chars), this triggers ~375 full DOM replacements. `generateBrief`'s equivalent correctly appends to a `<textarea>.value` incrementally.

---

### Naming / doc drift

**N-1 🟡 Minor — STEP_INFO[1] code snippet hardcodes year `2025`** (`~line 3169`)
The pipeline backend sends `2026`; the displayed code snippet still shows `2025`.

**N-2 🟡 Minor — Chat Search toggle sets button label to "Ask →"** (`~line 1821`)
`_toggleSearch` is shared between Ask and Chat tabs but the disabled-search label (`'Ask →'`) is Ask-tab language. When Search is disabled in Chat, the Send button reads "Ask →".

---

## ✅ Notable strengths (keep doing this)

- **`search_tool.py` single-source-of-truth pattern** — the intent is exactly right: one file owns `SYSTEM_PROMPT`, `MAX_TOKENS`, `MAX_TOOL_ROUNDS`, all tool schemas, and `execute_search`. Every agent imports from there rather than defining its own. The only violation is `pipeline_agent.py` (§3 above).
- **`_empty_stats()` canonical return shape** — defining the stats dict once and having all agents return it means the server's cost/token display code is robust against missing keys regardless of which agent ran.
- **Three-stage pipeline design** — `enrich` / `brief_stream` / `email_stream` are independent functions exposed as separate endpoints, so the UI controls sequencing without the server managing state. Clean and testable.
- **`_clean_markdown()` in the pipeline** — thoughtful decision to strip markdown before sending to the brief LLM, not just truncate. The regex chain handles the common noise patterns (nav boilerplate, link syntax, bullet markers) and includes a prose-length filter for short nav lines. Well-motivated by the FINDINGS.md engineering note.
- **FINDINGS.md + BACKLOG.md discipline** — engineering discoveries (GPT-OSS 20B reasoning field, `stream_options` breakage) and known tech debt are documented in-repo as they're found. This is the right practice for a fast-moving codebase.
- **Key hygiene** — `PARASAIL_API_KEY` and `YDC_API_KEY` are read from env at call time, never hardcoded, and the `.gitignore` excludes `.env`.
- **`_sse_ask` retry loop** — the 429 retry with a user-visible countdown event is a clean pattern; it surfaces the backpressure to the UI without silently hanging.
