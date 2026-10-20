"""The observer: one model turn over one batch, with purpose-built tools and an ownership-based verdict.

    observe(store, batch, client) -> Result

The turn-level signal answers one question only: did the observer complete a turn over this batch? Transport
failures (provider error, unknown tool, missing required search, too many rounds, a crash) fail the turn. Tool errors
inside a completed turn are feedback to the model and telemetry, never a verdict; the batch is decided by the ledger
(see runtime.Runtime.finalize).
"""
import json
import logging
from dataclasses import dataclass, field

from tacit import transcript, whisper
from tacit.buckets import BUCKETS
from tacit.llm import LLMError
from tacit.runtime import LINK_RELATIONS, Runtime, detect_intents
from tacit.store import stable_id

log = logging.getLogger("tacit.observer")

MAX_ROUNDS = 12


class TurnFailed(Exception):
    pass


class Feedback(ValueError):
    """Model-input validation: returned to the model so it can correct itself in this turn; never holds the batch."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    execute: object
    owned: bool = False  # its failure leaves batch-owned work unsettled

    def schema(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class ToolError:
    name: str
    kind: str  # invalid_arguments | execution
    message: str
    resolved: bool = False  # a later call to the same tool succeeded in this turn


@dataclass
class Result:
    settled: bool
    retry_reasons: list
    tool_errors: list = field(default_factory=list)
    rounds: int = 0
    whisper: str | None = None
    redactions: dict = field(default_factory=dict)


def run_conversation(client, messages, tools, required=(), session=None, max_rounds=MAX_ROUNDS):
    """The client-tool loop. Returns (tool_errors, rounds). Raises TurnFailed when the turn itself didn't complete."""
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    done, errors = set(), []
    for round_ in range(1, max_rounds + 1):
        try:
            reply = client.chat(messages, schemas, session=session)
        except LLMError as error:
            raise TurnFailed(f"provider: {error}") from None
        calls = reply.get("tool_calls") or []
        messages.append({"role": "assistant", "content": reply.get("content"), **({"tool_calls": calls} if calls else {})})
        if not calls:
            missing = sorted(set(required) - done)
            if missing:
                raise TurnFailed(f"turn ended before required tool(s): {', '.join(missing)}")
            return errors, round_
        for call in calls:
            name = (call.get("function") or {}).get("name")
            tool = by_name.get(name)
            if tool is None:
                raise TurnFailed(f"model requested unknown tool: {name}")
            kind = "invalid_arguments"
            try:
                args = json.loads((call["function"].get("arguments") or "{}"))
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be a JSON object")
                kind = "execution"
                content = json.dumps(tool.execute(args), ensure_ascii=False)
                done.add(name)
                for earlier in errors:  # the model corrected its own malformed call
                    if earlier.name == name and earlier.kind == "invalid_arguments":
                        earlier.resolved = True
            except Exception as error:  # noqa: BLE001 - every tool error goes back to the model
                errors.append(ToolError(name, kind, str(error)))
                content = json.dumps({"error": str(error)}, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": content})
    raise TurnFailed(f"tool loop exceeded {max_rounds} rounds")


# ---------------------------------------------------------------------------------------------------- tool surface

STRING = {"type": "string", "minLength": 1}
IDS = {"type": "array", "minItems": 1, "uniqueItems": True, "items": STRING}


FIELD_HELP = {
    "who": "The person (or team) whose preference this is.",
    "trigger": "When the habit applies, e.g. 'before committing'.",
    "source": "Where the fact came from: user (they said it) or tool (a tool result showed it).",
    "replaces": "What this fact replaces, e.g. the previous holder of a role. Omit if nothing.",
    "valid_from": "When it became true (ISO date), if the evidence says so. Omit otherwise.",
    "rationale": "Why it was decided, as stated in the evidence.",
    "symptom": "What goes wrong, as observed.",
    "fix": "What resolves it, as observed.",
}


def propose_schema(bucket):
    b = BUCKETS[bucket]
    fields = {}
    for name in b.required + b.optional:
        fields[name] = {"type": "string", "enum": list(b.choices[name])} if name in b.choices else dict(STRING)
        fields[name]["description"] = FIELD_HELP[name]
    return {
        "type": "object", "additionalProperties": False, "required": ["title", "fields", "evidence_ids"],
        "properties": {
            "title": {**STRING, "description": "One line, source-faithful: what was decided / preferred / learned."},
            "body": {"type": "string", "description": "Optional short context. Never invent reasons or details."},
            "fields": {"type": "object", "additionalProperties": False, "required": list(b.required), "properties": fields},
            "evidence_ids": {**IDS, "description": "Exact evidence_id values from the evidence catalog that show this."},
            "supersedes": {**STRING, "description": "An active card (from a maintenance search) this one replaces."},
            "valid_from": {"type": "string", "description": "When it became true in the world (ISO date), if stated."},
            "intent_id": {**STRING, "description": "The intent_id from the intent catalog, when this answers one."},
        },
    }


BUCKET_HELP = {
    "preference": "how a person wants things done (who = the person). A newer preference supersedes the old one.",
    "workflow_habit": "a repeated way of working (trigger = when it applies). Seen again -> reinforce.",
    "world_update": "a fact about the world after the model's training cutoff (a new release, a new person in a role). "
                    "Evidence must come from the user or a tool result, never the assistant. Newer supersedes older.",
    "decision": "a choice the team made, with its rationale. Only record what was actually decided, not ideas floated.",
    "gotcha": "a trap and its fix (symptom + fix). Bit again -> reinforce.",
}


SYSTEM = """You are tacit's observer. You read one batch of a Claude Code session and keep the team's knowledge base
current. You never see the whole history: use search to compare against what is already known.

Work in this order:
1. search (purpose=maintenance) for the topics in the batch.
2. For each piece of durable knowledge in the batch:
   - already known with the same meaning -> reinforce it with the new evidence ids;
   - replaces a known card -> propose_<bucket> with supersedes=<card>;
   - new -> propose_<bucket>.
   One piece of knowledge is one card, in the bucket that fits best: never record the same statement in several
   buckets. Ignore routine edits, installs, test runs, file reads and chatter. Saving nothing is a fine result.
3. Every entry in the intent catalog ("remember that ...") must be answered in this turn: propose a card with its
   intent_id, or decline_intent with a reason.
4. Optional: if an existing card would help with the user's latest message, search (purpose=whisper) and
   deliver_whisper once with the most useful card and 1-3 of its snippet_ids.
5. Optional: suggest_link when two current cards are related in a way a human should confirm.

Grounding rules: cite only evidence_id values from the catalog. Never retype or paraphrase a quote as evidence. Titles
and fields must say only what the evidence shows.

Buckets:
""" + "\n".join(f"- {name}: {text}" for name, text in BUCKET_HELP.items())


def build_message(batch, events, intents):
    latest_user = next((e.quote for e in reversed(events) if e.kind == "user_text"), "")
    parts = [f'<batch session="{transcript.escape(batch.session)}" lines="{batch.start_line}-{batch.end_line - 1}">',
             f"<latest_user_message>{transcript.escape(latest_user[:2000])}</latest_user_message>",
             transcript.catalog(events)]
    entries = "\n".join(f'  <intent intent_id="{i["intent"]}" evidence_id="{i["evidence_id"]}">'
                        f'{transcript.escape(i["text"])}</intent>' for i in intents)
    parts.append(f'<intent_catalog trusted="true">\n{entries}\n</intent_catalog>')
    parts.append("</batch>")
    return "\n\n".join(parts)


class Turn:
    """Per-turn state shared by the tools: what search surfaced, and whether the whisper is resolved."""

    def __init__(self, store, batch, runtime):
        self.store, self.batch, self.runtime = store, batch, runtime
        self.surfaced = {}  # card -> {snippet_id: snippet}
        self.topics = {}  # card -> the whisper search query that found it
        self.maintenance_seen = set()
        self.whisper = None
        self.whisper_resolved = False
        self.searched = False  # a maintenance search ran in this turn

    def require_search(self):
        if not self.searched:
            raise Feedback("search with purpose=maintenance first, so you can reinforce or supersede an existing card "
                           "instead of duplicating it; then call this again")

    def search(self, args):
        purpose = args.get("purpose")
        if purpose not in ("maintenance", "whisper"):
            raise ValueError("purpose must be maintenance or whisper")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        bucket = args.get("bucket")
        if bucket is not None and bucket not in BUCKETS:
            raise ValueError(f"unknown bucket: {bucket}")
        limit = max(1, min(int(args.get("limit") or 5), 20))
        exclude = self.store.session_known(self.batch.session) if purpose == "whisper" else ()
        try:
            cards = self.store.search(query, limit=limit, bucket=bucket, exclude=exclude)
            results = [whisper.search_result(self.store, c) for c in cards]
        except Exception:
            # A failed maintenance search can hide an owned write; a failed whisper search only loses the whisper
            if purpose == "maintenance":
                self.runtime.mark_retryable()
            raise
        self.searched = self.searched or purpose == "maintenance"
        for r in results:
            if purpose == "maintenance":
                self.maintenance_seen.add(r["card"])
            else:
                self.surfaced.setdefault(r["card"], {}).update({s["snippet_id"]: s for s in r["snippets"]})
                self.topics[r["card"]] = query
        return {"results": results}

    def wrote(self, result):
        """Cards this turn created or reinforced are known to the observer, like maintenance search results."""
        if isinstance(result, dict) and result.get("card"):
            self.maintenance_seen.add(result["card"])
        return result

    def suggest_link(self, args):
        cards = args.get("cards")
        unseen = [c for c in cards or [] if c not in self.maintenance_seen]
        if unseen:
            raise ValueError("suggest_link may reference only cards returned by a purpose=maintenance search, or written, "
                             f"in this turn: {', '.join(map(str, unseen))}")
        return self.runtime.suggest_link(args)

    def deliver_whisper(self, args):
        if self.whisper_resolved:
            return {"status": "already_resolved", "reason": "deliver_whisper runs at most once per batch."}
        card = args.get("card")
        picked = args.get("snippet_ids")
        if (not isinstance(card, str) or not isinstance(picked, list) or not 1 <= len(picked) <= 3
                or len(set(map(str, picked))) != len(picked)):
            raise ValueError("deliver_whisper requires one card and 1-3 unique snippet_ids. "
                             + whisper.describe_surfaced({c: list(s) for c, s in self.surfaced.items()}))
        surfaced = self.surfaced.get(card)
        if surfaced is None or any(p not in surfaced for p in picked):
            raise ValueError("deliver_whisper may select only a card and snippets surfaced by a purpose=whisper search "
                             "in this turn. " + whisper.describe_surfaced({c: list(s) for c, s in self.surfaced.items()}))
        current = self.store.get(card)
        self.whisper_resolved = True
        if current is None or current["status"] != "active":
            return {"status": "suppressed_no_longer_current"}  # superseded or retired between search and delivery
        known = set(self.store.session_known(self.batch.session))
        text, related = whisper.render(self.store, current, [surfaced[p] for p in picked],
                                       topic=self.topics.get(card, ""), known=known)
        whisper_id = stable_id("whisper", {"session": self.batch.session, "batch": self.batch.batch})
        queued = self.store.queue_whisper(whisper_id, self.batch.session, self.batch.batch, card, text, related)
        self.whisper = text if queued else None
        return {"status": "ok" if queued else "batch_already_whispered", "whisper": whisper_id}


def build_tools(turn):
    rt = turn.runtime
    tools = [Tool("search", "Search current cards (full text).", {
        "type": "object", "additionalProperties": False, "required": ["query", "purpose"],
        "properties": {"query": STRING, "purpose": {"type": "string", "enum": ["maintenance", "whisper"],
                                                    "description": "maintenance before writing; whisper when choosing "
                                                                   "a card for deliver_whisper (excludes cards this "
                                                                   "session already knows)"},
                       "bucket": {"type": "string", "enum": list(BUCKETS)},
                       "limit": {"type": "integer", "minimum": 1, "maximum": 20}}}, turn.search)]
    for bucket in BUCKETS:
        tools.append(Tool(f"propose_{bucket}", f"Propose a {bucket} card: {BUCKET_HELP[bucket]}", propose_schema(bucket),
                          lambda args, bucket=bucket: turn.wrote(turn.require_search() or rt.propose(bucket, args)),
                          owned=True))
    tools += [
        Tool("reinforce", "The batch shows an existing card again: attach the new evidence.", {
            "type": "object", "additionalProperties": False, "required": ["card", "evidence_ids"],
            "properties": {"card": STRING, "evidence_ids": IDS}}, lambda args: turn.wrote(turn.require_search() or rt.reinforce(args)), owned=True),
        Tool("decline_intent", "Answer an intent you decided not to store, with the reason.", {
            "type": "object", "additionalProperties": False, "required": ["intent_id", "reason"],
            "properties": {"intent_id": STRING, "reason": STRING}}, rt.decline_intent, owned=True),
        Tool("suggest_link", "Suggest a relation between current cards for a human to confirm. Never edits cards.", {
            "type": "object", "additionalProperties": False, "required": ["cards", "relation", "reason"],
            "properties": {"cards": {"type": "array", "minItems": 2, "maxItems": 8, "uniqueItems": True, "items": STRING},
                           "relation": {"type": "string", "enum": list(LINK_RELATIONS)},
                           "reason": {"type": "string", "minLength": 1, "maxLength": 1000}}}, turn.suggest_link),
        Tool("deliver_whisper", "Surface one card for the user's next prompt: a card and 1-3 snippet_ids returned by a "
                                "purpose=whisper search in this turn. tacit renders the text; don't write your own.", {
            "type": "object", "additionalProperties": False, "required": ["card", "snippet_ids"],
            "properties": {"card": STRING, "snippet_ids": {"type": "array", "minItems": 1, "maxItems": 3,
                                                           "uniqueItems": True, "items": STRING}}}, turn.deliver_whisper),
    ]
    return [owned_guard(t, rt) if t.owned else t for t in tools]


def owned_guard(tool, runtime):
    """Business failures come back as ledger outcomes; anything an owned tool *throws* is an unrecorded owned-work
    failure, so it holds the batch (and is still returned to the model as a tool error)."""
    def execute(args):
        try:
            return tool.execute(args)
        except Feedback:
            raise  # the call never reached the ledger, and the model can retry it in this turn
        except Exception:
            runtime.mark_retryable()
            raise
    return Tool(tool.name, tool.description, tool.parameters, execute, owned=True)


# ---------------------------------------------------------------------------------------------------- one batch

def observe(store, batch, client, max_rounds=MAX_ROUNDS):
    """Run one observer attempt over a batch and settle it (or hold it) from the ledger."""
    if store.batch_settled(batch):
        return Result(True, [])
    attempt = store.begin_attempt(batch)
    redactions, malformed = {}, []
    events = transcript.events(batch, redactions, malformed)
    for line in malformed:
        store.add_diagnostic(batch.session, line, "malformed_record", batch.batch)
    intents = detect_intents(store, batch, events)
    runtime = Runtime(store, batch, attempt, events, intents)
    if not events:  # nothing to read: settle without spending a model call
        settled, reasons = runtime.finalize(True)
        return Result(settled, reasons, redactions=redactions)
    turn = Turn(store, batch, runtime)
    tools = build_tools(turn)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_message(batch, events, intents)}]
    required = ("search",) if any(e.kind == "user_text" for e in events) else ()
    completed, errors, rounds = False, [], 0
    try:
        errors, rounds = run_conversation(client, messages, tools, required=required, session=batch.session,
                                          max_rounds=max_rounds)
        completed = True
        owned = {t.name for t in tools if t.owned}
        for error in errors:
            # An unparseable owned-work call never reached the ledger, unless the model corrected it in this turn
            if error.kind == "invalid_arguments" and error.name in owned and not error.resolved:
                runtime.mark_retryable()
            log.info("tool error returned to model: tool=%s kind=%s", error.name, error.kind)
    except TurnFailed as error:
        log.warning("observer turn failed: %s", error)
    except Exception:  # noqa: BLE001 - a crash is a failed turn, never a settled batch
        log.exception("observer crashed")
    settled, reasons = runtime.finalize(completed)
    return Result(settled, reasons, errors, rounds, turn.whisper, redactions)
