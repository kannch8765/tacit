"""The observer's write path for one batch attempt: grounded proposals, reinforcement, link suggestions, intents, and
the batch verdict.

Rules:
- Business failures never raise. A bad proposal is recorded as `rejected` with a reason code; a storage failure is
  recorded as `retryable` and holds the batch.
- Every write has a source key (batch + proposal), so replaying the same proposal after a crash or retry is idempotent.
- A batch fails only when work it owns is not durably settled: a failed turn, a retryable outcome, an owned-work
  exception, or a "remember that ..." intent still inside its hold budget. Whispers and link suggestions are optional
  and never fail a batch.
"""
import re
import sqlite3

from tacit.buckets import BUCKETS, Rejected, validate_fields
from tacit.store import stable_id

# A trusted intent holds its batch (and the transcript cursor) for at most this many completed observer turns. Then it
# is released as `deferred`, so one ignored intent can't freeze a session.
MAX_UNRESOLVED_INTENT_TURNS = 3

INTENT_PATTERN = re.compile(r"(?im)^\s*(?:tacit[,:]?\s+)?(?:#remember\b|remember(?:\s+that\b|\s*:))\s*(?P<text>\S.*)$")

PROPOSAL_KEYS = {"title", "body", "fields", "evidence_ids", "supersedes", "valid_from", "intent_id"}
LINK_RELATIONS = ("same_as", "supersedes", "relates_to", "contradicts")


def detect_intents(store, batch, events):
    """User messages that explicitly ask to remember something. Each becomes a durable, trusted intent."""
    intents = []
    for e in events:
        if e.kind != "user_text":
            continue
        match = INTENT_PATTERN.search(e.quote)
        if match:
            intent = stable_id("intent", {"evidence_id": e.evidence_id})
            intents.append(store.add_intent(intent, batch.session, batch, e.evidence_id, e.line, match["text"].strip()))
    return intents


class Runtime:
    def __init__(self, store, batch, attempt, events, intents=()):
        self.store = store
        self.batch = batch
        self.attempt = attempt
        self.events = {e.evidence_id: e for e in events}
        self.intents = {i["intent"]: i for i in intents}
        self.held = False  # an owned-work failure that produced no outcome row

    def mark_retryable(self):
        self.held = True

    # ------------------------------------------------------------------ outcome helpers

    def _record(self, key, outcome, card=None, code=None, reason=None, intent=None):
        try:
            self.store.record_outcome(self.attempt, self.batch, key, outcome, code=code, card=card, reason=reason)
            if intent:
                self.store.record_intent_outcome(intent, self.attempt, self.batch, outcome, card=card, reason=reason)
        except sqlite3.Error as error:
            return self._retryable(key, f"failed to record outcome: {error}", intent)
        result = {"outcome": outcome}
        if card:
            result["card"] = card
        if code:
            result["code"] = code
        if reason:
            result["reason"] = reason
        return result

    def _retryable(self, key, reason, intent=None):
        self.held = True
        try:
            self.store.record_outcome(self.attempt, self.batch, key, "retryable", code="card_retryable", reason=reason)
            if intent:
                self.store.record_intent_outcome(intent, self.attempt, self.batch, "retryable", reason=reason)
        except sqlite3.Error:
            pass  # `held` still keeps the cursor
        return {"outcome": "retryable", "reason": reason}

    def _replay(self, key, previous, intent):
        """A proposal this batch already settled: copy its outcome into this attempt and report it as a duplicate."""
        if previous["outcome"] == "rejected":
            return self._record(key, "rejected", code=previous["code"], reason=previous["reason"], intent=intent)
        return self._record(key, "duplicate", card=previous["card"], intent=intent)

    def _evidence(self, bucket, ids):
        if not isinstance(ids, list) or not ids:
            raise Rejected("no_evidence")
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise Rejected("bad_evidence_ids", "evidence_ids must be unique, non-empty strings")
        events = []
        for evidence_id in ids:
            event = self.events.get(evidence_id.strip())
            if event is None:
                raise Rejected("unresolvable_evidence", f"not in this batch's evidence catalog: {evidence_id}")
            if bucket and event.role not in BUCKETS[bucket].evidence_roles:
                raise Rejected("evidence_role_not_allowed", f"{bucket} can't be grounded by {event.role} evidence")
            events.append(event)
        return events

    def _intent(self, raw):
        intent = raw.get("intent_id")
        if intent is None:
            return None
        if intent not in self.intents:
            raise Rejected("unknown_intent", f"not in this batch's intent catalog: {intent}")
        return intent

    # ------------------------------------------------------------------ tools

    def propose(self, bucket, raw):
        """Create a card, or recognise it as existing knowledge (duplicate / reinforce), or reject it with a code."""
        raw = raw if isinstance(raw, dict) else {}
        key = stable_id("src", {"batch": self.batch.batch, "bucket": bucket, "proposal": raw})
        intent = raw.get("intent_id") if raw.get("intent_id") in self.intents else None
        previous = self.store.terminal_outcome(key)
        if previous:
            return self._replay(key, previous, intent)
        try:
            unknown = sorted(set(raw) - PROPOSAL_KEYS)
            if unknown:
                raise Rejected("unknown_argument", ", ".join(unknown))
            intent = self._intent(raw)
            if bucket not in BUCKETS:
                raise Rejected("unknown_bucket", bucket)
            title = raw.get("title")
            if not isinstance(title, str) or not title.strip():
                raise Rejected("missing_title")
            events = self._evidence(bucket, raw.get("evidence_ids"))
            fields = validate_fields(bucket, raw.get("fields", {}))
            evidence = [e.as_evidence() for e in events]
            dedupe_key = stable_id("dedupe", {"bucket": bucket, "title": title.strip(), "fields": fields,
                                             "evidence": sorted(e.evidence_id for e in events)})
            existing = self.store.card_by_dedupe(dedupe_key) or self.store.find_active(bucket, title, fields)
            if existing and existing["status"] == "active":
                return self._reinforce_existing(key, existing["card"], events, intent)
            if existing:  # the exact same proposal already landed and was superseded since: nothing new to learn
                return self._record(key, "duplicate", card=existing["card"], intent=intent)
            supersedes = raw.get("supersedes")
            body = raw.get("body") or ""
            if not isinstance(body, str):
                raise Rejected("bad_body")
            said_at = sorted(e.captured_at for e in events if not e.captured_at.startswith("1970"))
            valid_from = raw.get("valid_from") or fields.get("valid_from") or (said_at[0] if said_at else None)
            card = self.store.add_card(self.batch, bucket, title, fields, evidence, body=body,
                                       valid_from=valid_from, supersedes=supersedes, dedupe_key=dedupe_key)
        except Rejected as rejected:
            return self._record(key, "rejected", code=rejected.code, reason=str(rejected), intent=intent)
        except sqlite3.Error as error:
            return self._retryable(key, str(error), intent)
        return self._record(key, "superseded" if supersedes else "accepted", card=card, intent=intent)

    def _reinforce_existing(self, key, card, events, intent=None):
        known = {e["evidence_id"] for e in self.store.evidence(card)}
        new = [e for e in events if e.evidence_id not in known]
        if not new:
            return self._record(key, "duplicate", card=card, intent=intent)
        self.store.reinforce(self.batch, card, [e.as_evidence() for e in new])
        return self._record(key, "reinforced", card=card, intent=intent)

    def reinforce(self, raw):
        """The batch shows an existing card again: keep the new evidence and strengthen the card."""
        raw = raw if isinstance(raw, dict) else {}
        card, ids = raw.get("card"), raw.get("evidence_ids")
        key = stable_id("reinforce", {"batch": self.batch.batch, "card": card,
                                      "evidence": sorted(ids) if isinstance(ids, list) and all(isinstance(i, str) for i in ids) else ids})
        previous = self.store.terminal_outcome(key)
        if previous:
            return self._replay(key, previous, None)
        try:
            target = self.store.get(card) if isinstance(card, str) else None
            if target is None:
                raise Rejected("reinforce_unknown_card", str(card))
            if target["status"] != "active":
                raise Rejected("reinforce_inactive_card", target["status"])
            return self._reinforce_existing(key, card, self._evidence(target["bucket"], ids))
        except Rejected as rejected:
            return self._record(key, "rejected", code=rejected.code, reason=str(rejected))
        except sqlite3.Error as error:
            return self._retryable(key, str(error))

    def decline_intent(self, raw):
        """The observer looked at a "remember that ..." and decided it isn't durable knowledge. Final, with a reason."""
        raw = raw if isinstance(raw, dict) else {}
        intent, reason = raw.get("intent_id"), raw.get("reason")
        if intent not in self.intents:
            return {"outcome": "rejected", "code": "unknown_intent"}
        if not isinstance(reason, str) or not reason.strip():
            return {"outcome": "rejected", "code": "missing_reason"}
        try:
            self.store.record_intent_outcome(intent, self.attempt, self.batch, "declined", reason=reason.strip())
        except sqlite3.Error as error:
            self.held = True
            return {"outcome": "retryable", "reason": str(error)}
        return {"outcome": "declined"}

    def suggest_link(self, raw):
        """Queue a relation between existing cards for human review. Optional work: never holds the batch."""
        raw = raw if isinstance(raw, dict) else {}
        cards, relation, reason = raw.get("cards"), raw.get("relation"), raw.get("reason")
        if (not isinstance(cards, list) or not 2 <= len(cards) <= 8 or len(set(map(str, cards))) != len(cards)
                or relation not in LINK_RELATIONS or not isinstance(reason, str) or not reason.strip() or len(reason) > 1000):
            return {"outcome": "rejected",
                    "reason": "needs 2-8 unique cards, a relation in " + "/".join(LINK_RELATIONS) + " and a reason (<= 1000 chars)"}
        for card in cards:
            found = self.store.get(card)
            if found is None or found["status"] != "active":
                return {"outcome": "rejected", "reason": f"not a current card: {card}"}
        ordered = sorted(cards)
        review = stable_id("review", {"cards": ordered})
        added = self.store.add_review(review, relation, ordered, reason.strip(), self.batch)
        return {"outcome": "queued" if added else "duplicate", "review": review}

    # ------------------------------------------------------------------ verdict

    def finalize(self, turn_completed):
        """Decide the batch from the ledger. Returns (settled, retry_reasons)."""
        reasons = []
        if not turn_completed:
            reasons.append("turn_failed")
        if self.held:
            reasons.append("attempt_retryable")
        unresolved = [i for i in self.intents if not self.store.intent_resolved(i)]
        # Only a turn the observer actually completed counts against an intent's budget
        blocking = self._defer(unresolved) if turn_completed else unresolved
        if blocking:
            reasons.append("intent_blocking")
        settled = self.store.finish_attempt(self.attempt, self.batch, reasons)
        recorded = self.store.attempt(self.attempt)["retry_reasons"]
        return settled, recorded.split(",") if recorded else []

    def _defer(self, intents):
        blocking = []
        for intent in intents:
            turn = self.store.intent_deferrals(intent) + 1
            released = turn >= MAX_UNRESOLVED_INTENT_TURNS
            reason = (f"unresolved after {turn} completed observer turns; released so the cursor can advance" if released
                      else f"unresolved after completed observer turn {turn} of {MAX_UNRESOLVED_INTENT_TURNS}")
            try:
                self.store.record_intent_outcome(intent, self.attempt, self.batch, "deferred", reason=reason)
            except sqlite3.Error:
                blocking.append(intent)  # without a durable deferral, keep holding
                continue
            if not released:
                blocking.append(intent)
        return blocking
