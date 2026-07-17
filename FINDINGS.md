# Engineering Findings — Parasail × You.com Playground

> Discoveries made during the hackathon build. These inform future pipeline work
> and are worth sharing with anyone integrating Parasail's open-weight models.

---

## 1. GPT-OSS 20B is a Reasoning Model — Output Lives in `model_extra['reasoning']`

**What we found:** When streaming from `openai/gpt-oss-20b` via the Parasail API, every chunk
has `choices[0].delta.content = None`. The actual generated text is in a non-standard field
`choices[0].delta.reasoning` (exposed via `model_extra` in the OpenAI SDK). This is true for
both streaming and non-streaming calls — `message.content` is `None`; `message.model_extra['reasoning']`
holds the full response.

**Why it happens:** GPT-OSS 20B is deployed in a chain-of-thought / reasoning mode on Parasail.
The model shows its thinking process in the `reasoning` field before (or instead of) producing a
`content` response. This is similar to OpenAI's o-series models but with a different field name.

**How we fixed it:**
```python
msg  = resp.choices[0].message
text = msg.content or (msg.model_extra or {}).get("reasoning", "") or ""
```

**What to watch for:**
- Other open-weight models on Parasail may behave differently — always log `chunk.choices` on the
  first chunk during integration to confirm where text lands.
- Models like Qwen3 have a `/no_think` toggle; GPT-OSS doesn't expose this via the API.

---

## 2. `stream_options={"include_usage": True}` Broke Token Streaming

**What we found:** Adding `stream_options={"include_usage": True}` to the streaming call caused
ALL intermediate chunks to have `choices[0].delta.content = None` (or empty string `''`). The
token usage appeared correctly in the final `chunk.usage` field, but no content was ever streamed.

**Why it happens:** Parasail's implementation appears to buffer the full response when
`stream_options` is requested, delivering it in a non-standard way that the OpenAI SDK interprets
as empty `delta.content` on all chunks. The model generates correctly (token counts confirm this),
but the chunks don't carry the text progressively.

**How we fixed it:** Removed `stream_options`. Instead, we switched to a non-streaming call
(`stream=False`) and then re-emit the completed response as synthetic SSE chunks client-side:
```python
resp = client.chat.completions.create(model=model, ..., stream=False)
text = msg.content or msg.model_extra.get("reasoning", "")
for i in range(0, len(text), CHUNK_SIZE):
    yield {"event": "token", "text": text[i:i + CHUNK_SIZE]}
```

This gives the user a streaming feel while being fully compatible with reasoning models.

---

## 3. `max_tokens=500` Cut Off Reasoning Before Any Output Appeared

**What we found:** With `max_tokens=500`, the pipeline steps returned only partial reasoning
("We need to produce a concise brief for a sales rep...") and never reached the actual answer.
The model exhausted its token budget mid-thought.

**Why it happens:** Reasoning models consume tokens for their chain-of-thought *before* producing
output. With a hard cap of 500, the model runs out while still in the thinking phase. The final
`content` (or `reasoning` text after the thought-boundary) never renders.

**How we fixed it:** Bumped `max_tokens` to 1500. Rule of thumb for reasoning model pipeline
steps: budget at least 3× the expected output length to account for the thinking overhead.

---

## 4. System Prompt Must Explicitly Suppress Reasoning Narration

**What we found:** Even with higher `max_tokens`, the model's reasoning output leaked through
as the response — long "We need to think about this..." preambles before the actual bullets or
email. The original system prompts ("Output ONLY a bulleted list...") weren't enough.

**Why it happens:** The `reasoning` field contains the model's full internal monologue. Soft
instructions don't stop it. The model interprets "output only" as applying to the final
`content` field, not the `reasoning` chain.

**How we fixed it:** Added hard constraints with a concrete first-character rule:
```
"Do not narrate your reasoning. Do not say what you are about to do."
"First character of your response must be '•'."   # brief
"First word of your response must be 'Subject'."  # email
```

The concrete first-character instruction anchors the model's output format reliably.

---

## Summary Table

| Finding | Root Cause | Fix |
|---|---|---|
| Brief/email shows blank | Content in `reasoning`, not `content` | Read `model_extra['reasoning']` |
| Token streaming broken | `stream_options` incompatible | Switch to `stream=False` + re-emit |
| Output cut off mid-thought | `max_tokens=500` too low | Raise to 1500+ |
| Reasoning narration leaks | Soft prompt not enough | Hard first-char anchor in system prompt |
