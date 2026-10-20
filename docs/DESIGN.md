# tacit: design (v0.1)

> A team knowledge base that writes itself from your Claude Code sessions, and only says what it can prove.

## 1. Problem

Teams make decisions in chat with their coding agent every day: *"we pin pydantic < 3 because of X"*, *"never run
migrations from CI"*, *"the flaky test is a race in the cache warmup"*. These live in transcripts nobody re-reads, so
the next session (or teammate) rediscovers them the hard way. CLAUDE.md files help, but they're written by hand and
go stale.

Memory tools that *do* write automatically have a trust problem: an LLM summarizing a transcript will happily
invent a "decision" that was only floated, or recall one that never happened.

## 2. What tacit does

1. **Observes**, without blocking. After each Claude Code turn, a hook queues the session. A background worker reads
   the new, complete transcript records later, in batches.
2. **Extracts grounded knowledge.** The observer model proposes typed **cards** in five buckets (below), each citing
   **evidence ids** from a trusted catalog of the batch's transcript events (one per text block, tool call and tool
   result). tacit copies the quote from the transcript itself, so the model never types a quote it could make up.
   Secrets are redacted (by pattern, on the whole block) when events are built, before anything is stored or sent. A
   card whose ids don't resolve in the batch is rejected and logged with a reason code. It is never silently saved.
3. **Settles into a ledger and weaves it.** Cards are rows in SQLite, not loose markdown. A proposal becomes a new
   card, a repeat (which **reinforces** the existing card), or a new version (which **supersedes** it). Cards are
   linked (`supersedes`, `relates_to`, `contradicts`), so *"we switched from X to Y"* connects the new decision to the
   old one. Every outcome is recorded.
4. **Whispers, grounded and woven.** *Selection:* the observer may pick only a card its own search returned in this
   turn (`deliver_whisper`), and only that card's real snippets. *Weaving:* tacit follows the card's history and
   links and renders the final text with dates and quotes, e.g. *"tacit · world_update (as of 2026-10-14, v1): Mika
   is the PM for payments"*, followed by the exact quotes it came from. A fast sync lane (no LLM) adds clear full-text matches before a prompt.
5. **Asks a human when it matters.** Links the observer suggests between cards (`same_as`, `supersedes`,
   `relates_to`, `contradicts`) wait in `tacit review` until a person accepts or dismisses them. Decisions are ledger
   entries too, and they follow the same bucket rules as the observer's writes.

## 2b. Buckets (each with its own fields and lifecycle)

| Bucket | Example | Required fields | Lifecycle |
|---|---|---|---|
| `preference` | "Mika wants answers short, bullets first" | `who` | newest supersedes |
| `workflow_habit` | "branch before committing; run tests first" | `trigger` | reinforced on repeat |
| `world_update` | "Mika is now the PM for payments" (was Ken) | `source` = `user` or `tool`; optional `replaces`, `valid_from` | newest supersedes; **evidence must come from the user or a tool result, never the assistant** |
| `decision` | "pin pydantic < 3" | `rationale` | stays until superseded |
| `gotcha` | "flaky test = cache warmup race" | `symptom`, `fix` | reinforced when it bites again |

`world_update` is a **knowledge-cutoff patch**: things that happened after the model's training, which it would
otherwise get confidently wrong.

**Versioning (SCD type 2 / bitemporal):** cards are never edited in place. Superseding closes the old version
(`superseded_at`, `superseded_by`) and inserts a new version on the same `lineage`, linked by `supersedes`. Each card
keeps two times: `valid_from` (when it became true in the world) and `recorded_at` (when tacit learned it). So
"current knowledge" and "what did we believe on 10-10?" are both plain queries, and history is never lost.

## 3. What's new compared with upstream

[letta-ai/claude-subconscious](https://github.com/letta-ai/claude-subconscious) had the brilliant base idea: a
background agent, driven by Claude Code hooks, that watches sessions and whispers back. tacit keeps that shape (hook
lifecycle, transcript cursor, detached worker) and adds the parts below.

| # | Idea | Upstream | tacit | Why it matters |
|---|---|---|---|---|
| 1 | **Structured memory with relations** | memory as markdown text blocks | SQLite rows + a `links` table | Related knowledge is woven together and queryable; repeats and replacements are explicit |
| 2 | **Purpose-built agent tools** | generic bash / file-edit tools | `search`, `propose_<bucket>`, `reinforce`, `suggest_link`, `decline_intent`, `deliver_whisper`; no file editing | Fewer agent steps, fewer broken edits, every call validated in code |
| 3 | **Grounded writes** | the model writes what it wants | cards cite evidence ids from the batch; quotes are copied from the transcript | No invented memories or quotes |
| 4 | **Grounded whispers** | free-form guidance | only cards a search returned this turn, with quotes and dates | No false recall |
| 5 | **Two lanes** | one async agent | sync recall (no LLM, fast) + async consolidation | Prompts never wait on an LLM |
| 6 | **Trusted intents** | n/a | "remember that…" must get an answer (a card or an explicit decline) | User intent is never silently dropped |
| 7 | **Outcome ledger + ownership-based batch outcome** | n/a | every batch / attempt / proposal outcome recorded; only owned work holds a batch; cursor held until settled | Retries converge instead of duplicating |
| 8 | **Human in the loop** | n/a | `tacit review` inbox for suggested links | People own the final word on shared knowledge |
| 9 | **Observability as a feature** | n/a | health checks + a daily chart report from the ledger | Failure and reinforcement rates are measured, not guessed |
| 10 | **Privacy by design** | n/a | local store, pattern-based redaction before API calls, stats export numbers only | Transcripts contain keys and private code |

## 4. Architecture

```
 Claude Code                       tacit (local)
 ───────────                       ─────────────
 Stop hook ──── queue session ──▶  sessions (transcript path, cursor, byte offset)
                                      │
                                      ▼  background worker (detached, one at a time)
                                   observer ── OpenAI-compatible model (tool calls)
                                      │
                                      ▼
                                   validator ── grounding check, bucket rules
                                      │
                                      ▼
                                   ledger (SQLite) ── cards · evidence · links
                                      │                batches · attempts · outcomes · intents · reviews
                     ┌────────────────┼──────────────────┐
                     ▼                ▼                  ▼
 UserPromptSubmit ◀─ whispers     tacit review       tacit stats / health
   (queued + sync lane,           (human inbox)      + report.html
    no LLM)                                          (numbers only)
```

- **Language:** Python 3.12 + `uv`. Stdlib `sqlite3` with **FTS5** for search (no vector DB in v0.1), with its own
  tokens so Japanese and Chinese text is searchable. No LLM SDK.
- **Models:** the observer talks to any **OpenAI-compatible chat-completions endpoint** with tool calling, through a
  small stdlib HTTP client. The model and SDK aren't the point; the harness around them is. Swapping providers is
  config only. Whisper retrieval uses no LLM (FTS + recency), so prompts stay fast.
- **Packaging:** a Claude Code plugin (`hooks.json` + a `tacit` CLI).
- **Observability:** `tacit stats` rolls the ledger up per day (batches, retryable share and reasons, proposal
  outcomes, rejections by code, observer turn time, whispers, intents). `tacit health` gives ok / warn / alert.
  `tacit report` writes a self-contained `report.html` (inline SVG). Their output is numbers and fixed labels only,
  never card text.

## 5. Scope of v0.1

**In:**
- Hooks (Stop → queue, UserPromptSubmit → whisper) + detached worker with a lock
- Observer with tool use + grounding validator + redaction
- Ledger schema, reinforce / supersede, suggested links → review
- Ownership-based batch outcome + held cursor + retry
- `tacit review`, `tacit search`, `tacit stats` / `health` / `report`
- Tests with a scripted fake model (grounding, ledger convergence, crash/retry, no secrets reach the API payload)

**Out (named on purpose, as later work):** embeddings, a web UI, team sync / multi-user, a `.tacitignore` file,
editing CLAUDE.md, any non-Claude-Code agent.

## 6. Decisions

- **Python 3.12 + uv**, stdlib SQLite/FTS5, no LLM SDK: an OpenAI-compatible client, provider-agnostic by config.
- **Built and tested on made-up transcripts** with a scripted fake model, then smoke-tested live on the same made-up
  sessions (see [demo/](../demo/)).
- **MIT** license.
