"""Transcript events: a batch's raw Claude Code JSONL lines become trusted evidence events, plus the catalog the observer
sees.

Grounding works by reference, not by quotation: every text block, tool call and tool result gets a backend-owned
`evidence_id`, and the observer may only cite those ids. tacit then copies the quote from the transcript itself, so the
model never types a quote it could make up.

One event per content block, with ids stable across retries of the same lines.
"""
import json
from dataclasses import dataclass

from tacit.redact import redact, redact_value
from tacit.store import Evidence, stable_id

MAX_EVIDENCE_CHARS = 12_000

KINDS = ("user_text", "assistant_text", "assistant_tool_use", "tool_result")


@dataclass(frozen=True)
class Event:
    evidence_id: str
    line: int
    message_id: str
    block: int
    role: str  # user | assistant | tool
    kind: str
    quote: str
    captured_at: str
    tool_name: str | None = None
    tool_use_id: str | None = None

    def as_evidence(self):
        return Evidence(self.line, self.role, self.quote, self.evidence_id, self.kind)


def block_text(value):
    """(text, redaction counts) for one block: JSON for structured tool input/output, sensitive fields scrubbed, then
    the whole text redacted, and only then trimmed and capped. Text is None when empty."""
    counts = {}
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(redact_value(value, counts), indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text, found = redact(text.strip())
    for kind, n in found.items():
        counts[kind] = counts.get(kind, 0) + n
    if not text:
        return None, counts
    if len(text) > MAX_EVIDENCE_CHARS:
        text = f"{text[:MAX_EVIDENCE_CHARS]}\n... [evidence truncated at {MAX_EVIDENCE_CHARS} chars]"
    return text, counts


def events(batch, redactions=None, malformed=None):
    """Evidence events for a batch. Records that aren't user/assistant messages are skipped.

    `redactions`, if given, is a dict that collects {kind: count} of scrubbed secrets (numbers only, for stats).
    `malformed`, if given, is a list that collects the line numbers of records that aren't JSON objects."""
    result = []
    tool_names = {}
    for offset, raw in enumerate(batch.lines):
        line = batch.start_line + offset
        try:
            message = json.loads(raw)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            if malformed is not None and raw.strip():
                malformed.append(line)
            continue
        if message.get("type") not in ("user", "assistant"):
            continue
        message_id = message.get("uuid") or f"line:{line}"
        captured_at = message.get("timestamp") or "1970-01-01T00:00:00Z"
        body = message.get("message") if isinstance(message.get("message"), dict) else {}
        content = body.get("content", message.get("content"))
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            kind = tool_name = tool_use_id = value = None
            if block.get("type") == "text":
                kind = "user_text" if message["type"] == "user" else "assistant_text"
                role = message["type"]
                value = block.get("text")
            elif message["type"] == "assistant" and block.get("type") == "tool_use" and block.get("name"):
                kind, role = "assistant_tool_use", "assistant"
                tool_name = str(block["name"])
                tool_use_id = block.get("id") or None
                if tool_use_id:
                    tool_names[tool_use_id] = tool_name
                value = block.get("input")
            elif block.get("type") == "tool_result":
                kind, role = "tool_result", "tool"
                tool_use_id = block.get("tool_use_id") or None
                tool_name = tool_names.get(tool_use_id)
                value = block.get("content")
            if not kind or value is None:
                continue
            quote, counts = block_text(value)
            if not quote:
                continue
            if redactions is not None:
                for k, n in counts.items():
                    redactions[k] = redactions.get(k, 0) + n
            evidence_id = stable_id("ev", {"session": batch.session, "message_id": message_id, "block": index,
                                           "kind": kind, "tool_use_id": tool_use_id})
            result.append(Event(evidence_id, line, message_id, index, role, kind, quote, captured_at, tool_name, tool_use_id))
    return result


def escape(value):
    return (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


SEMANTICS = """<evidence_semantics>
Trusted evidence proves only what appeared in the Claude Code transcript and where; a claim inside a tool result is not
automatically true.
Judge meaning yourself. Ignore routine edits, installs, test runs, file reads and shell noise unless they carry durable
team knowledge: a decision and its reason, a preference, a workflow habit, a gotcha and its fix, or a world update the
model could not have known.
Cite evidence only by the exact evidence_id values below. Never invent an id or retype a quote.
</evidence_semantics>"""


def catalog(evs):
    """The trusted evidence catalog appended to the observer's message."""
    entries = []
    for e in evs:
        attrs = [f'evidence_id="{escape(e.evidence_id)}"', f'role="{e.role}"', f'kind="{e.kind}"', f'line="{e.line}"']
        if e.tool_name:
            attrs.append(f'tool_name="{escape(e.tool_name)}"')
        attrs.append(f'captured_at="{escape(e.captured_at)}"')
        entries.append(f"  <evidence {' '.join(attrs)}>{escape(e.quote)}</evidence>")
    body = "\n".join(entries)
    return f'{SEMANTICS}\n\n<evidence_catalog trusted="transcript_provenance_only">\n{body}\n</evidence_catalog>'
