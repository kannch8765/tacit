"""The ledger: cards (versioned, SCD type 2 / bitemporal), evidence, links, and the batch → attempt → outcome trail.

Rules the store enforces, so callers can't get them wrong:
- Cards are never edited in place. Superseding closes the old version and inserts a new one on the same lineage.
- Every card has evidence, and every evidence role must be allowed by the card's bucket.
- A session's transcript cursor moves only when a batch settles, i.e. its owned work is done. A failed attempt keeps
  the cursor where it was, and the next batch for that session starts at the same line (and may be wider).
- Only complete records (ending in a newline) are read, from the settled byte offset. A half-written last line stays
  unread until its writer finishes it.
"""
import datetime
import hashlib
import json
import pathlib
import sqlite3
import uuid
from dataclasses import dataclass

from tacit.buckets import BUCKETS, Rejected, validate_fields
from tacit.text import tokens

SCHEMA = """
create table if not exists sessions (
    session text primary key, transcript_path text not null, cursor integer not null default 0, updated_at text,
    byte_offset integer not null default 0);
create table if not exists batches (
    batch text primary key, session text not null, start_line integer not null, end_line integer not null,
    created_at text not null, status text not null check (status in ('pending', 'settled')),
    end_offset integer not null default 0);
-- problems worth a look that hold nothing: codes and line numbers only, never transcript text
create table if not exists diagnostics (
    session text not null, batch text not null default '', line integer not null, code text not null,
    created_at text not null, primary key (session, batch, line, code));
create table if not exists attempts (
    attempt text primary key, batch text not null references batches, started_at text not null, finished_at text,
    status text not null check (status in ('running', 'completed', 'retryable')), retry_reasons text);
create table if not exists cards (
    card text primary key, lineage text not null, version integer not null, bucket text not null,
    title text not null, body text not null default '', fields text not null,
    status text not null check (status in ('active', 'superseded', 'in_review', 'retired')),
    valid_from text, recorded_at text not null, superseded_at text, superseded_by text,
    reinforcements integer not null default 0, last_seen_at text not null, dedupe_key text,
    unique (lineage, version));
create table if not exists evidence (
    card text not null references cards, batch text not null, line integer not null, role text not null,
    quote text not null, evidence_id text, kind text);
create table if not exists links (
    src text not null references cards, dst text not null references cards,
    kind text not null check (kind in ('supersedes', 'relates_to', 'contradicts')), created_at text not null,
    primary key (src, dst, kind));
create table if not exists outcomes (
    attempt text not null references attempts, batch text not null, proposal text not null, card text,
    outcome text not null check (outcome in
        ('accepted', 'duplicate', 'reinforced', 'superseded', 'review', 'rejected', 'retryable')),
    code text, recorded_at text not null, reason text);
create table if not exists intents (
    intent text primary key, session text not null, batch text not null, evidence_id text not null,
    line integer not null, text text not null, created_at text not null);
create table if not exists intent_outcomes (
    intent text not null references intents, attempt text not null, batch text not null,
    outcome text not null check (outcome in
        ('accepted', 'duplicate', 'reinforced', 'superseded', 'declined', 'rejected', 'retryable', 'deferred')),
    card text, reason text, recorded_at text not null);
create table if not exists reviews (
    review text primary key, relation text not null, cards text not null, reason text not null,
    status text not null check (status in ('pending', 'accepted', 'dismissed')), batch text,
    created_at text not null, decided_at text);
create table if not exists whispers (
    whisper text primary key, session text not null, batch text not null, card text not null, text text not null,
    created_at text not null, delivered_at text, related text not null default '[]');
-- `terms` holds tacit's own tokens (see text.py), so CJK text is searchable by any two-character piece
create virtual table if not exists card_search using fts5(card unindexed, terms);
create index if not exists cards_lineage on cards (lineage, version);
create index if not exists outcomes_batch on outcomes (batch);
create index if not exists outcomes_proposal on outcomes (proposal);
create index if not exists whispers_session on whispers (session, delivered_at);
"""

ROLES = ("user", "assistant", "tool")


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def stable_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_id(prefix, value):
    """Content-addressed id: the same input always gets the same id (retries stay idempotent)."""
    return f"{prefix}_{hashlib.sha256(stable_json(value).encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class Batch:
    batch: str
    session: str
    start_line: int
    end_line: int  # exclusive
    lines: list
    end_offset: int = 0  # byte offset just after the batch's last line


@dataclass(frozen=True)
class Evidence:
    line: int
    role: str
    quote: str
    evidence_id: str | None = None  # the transcript event this quote was copied from
    kind: str | None = None


class Store:
    def __init__(self, path, clock=utcnow):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("pragma journal_mode = wal")
        self.db.execute("pragma foreign_keys = on")
        self.db.executescript(SCHEMA)
        self.migrate()

    def migrate(self):
        """Columns added after v0.1's first schema, for stores created before them."""
        for table, column in (("sessions", "byte_offset"), ("batches", "end_offset")):
            if column not in {r["name"] for r in self.db.execute(f"pragma table_info({table})")}:
                with self.db:
                    self.db.execute(f"alter table {table} add column {column} integer not null default 0")

    def close(self):
        self.db.close()

    # ------------------------------------------------------------------ queue + cursor

    def enqueue(self, session, transcript_path):
        """Called by the Stop hook: remember the session's transcript. Cheap, never reads the transcript."""
        with self.db:
            self.db.execute(
                "insert into sessions (session, transcript_path, updated_at) values (?, ?, ?) "
                "on conflict (session) do update set transcript_path = excluded.transcript_path, updated_at = excluded.updated_at",
                (session, str(transcript_path), self.clock()))

    def cursor(self, session):
        row = self.db.execute("select cursor from sessions where session = ?", (session,)).fetchone()
        return row["cursor"] if row else None

    def pending_sessions(self):
        return [r["session"] for r in self.db.execute("select session from sessions order by updated_at")]

    def next_batch(self, session, max_lines=200):
        """The complete transcript records after the cursor, as a batch, or None when caught up.

        Reads from the settled byte offset, at most `max_lines` records, and stops before a last line with no newline
        yet (its writer isn't done). The batch id is a hash of (session, start, end), so a retry after a failure that
        sees new lines gets a new, wider batch starting at the same held cursor."""
        row = self.db.execute("select transcript_path, cursor, byte_offset from sessions where session = ?",
                              (session,)).fetchone()
        if row is None:
            return None
        start, offset, lines = row["cursor"], row["byte_offset"], []
        try:
            with open(row["transcript_path"], "rb") as f:
                if offset == 0 and start > 0:  # a store from before byte offsets: skip the settled lines once
                    for _ in range(start):
                        offset += len(f.readline())
                if f.seek(0, 2) < offset:
                    self.add_diagnostic(session, start, "transcript_shrank")
                    return None
                f.seek(offset)
                end_offset = offset
                while len(lines) < max_lines:
                    raw = f.readline()
                    if not raw.endswith(b"\n"):
                        break
                    lines.append(raw.rstrip(b"\r\n").decode("utf-8", errors="replace"))
                    end_offset += len(raw)
        except FileNotFoundError:
            return None
        if not lines:
            return None
        end = start + len(lines)
        batch = hashlib.sha256(f"{session}\x1f{start}\x1f{end}".encode()).hexdigest()[:16]
        with self.db:
            self.db.execute("insert or ignore into batches values (?, ?, ?, ?, ?, 'pending', ?)",
                            (batch, session, start, end, self.clock(), end_offset))
        return Batch(batch, session, start, end, lines, end_offset)

    def add_diagnostic(self, session, line, code, batch=""):
        with self.db:
            self.db.execute("insert or ignore into diagnostics values (?, ?, ?, ?, ?)",
                            (session, batch, line, code, self.clock()))

    def diagnostics(self):
        return [dict(r) for r in self.db.execute("select * from diagnostics order by created_at, line")]

    # ------------------------------------------------------------------ attempts + outcomes

    def begin_attempt(self, batch):
        attempt = new_id("a")
        with self.db:
            self.db.execute("insert into attempts (attempt, batch, started_at, status) values (?, ?, ?, 'running')",
                            (attempt, batch.batch, self.clock()))
        return attempt

    def record_outcome(self, attempt, batch, proposal, outcome, code=None, card=None, reason=None):
        """`proposal` is the proposal's source key: the same proposal in the same batch always gets the same key."""
        with self.db:
            self.db.execute("insert into outcomes values (?, ?, ?, ?, ?, ?, ?, ?)",
                            (attempt, batch.batch, proposal, card, outcome, code, self.clock(), reason))

    def terminal_outcome(self, proposal):
        """The latest final (non-retryable) outcome for a source key, from any attempt, or None."""
        row = self.db.execute("select * from outcomes where proposal = ? and outcome != 'retryable' "
                              "order by rowid desc limit 1", (proposal,)).fetchone()
        return dict(row) if row else None

    def finish_attempt(self, attempt, batch, reasons=()):
        """Settle the batch if all its owned work reached a final outcome; otherwise keep the cursor and mark the attempt
        retryable. `retryable` outcomes count as unsettled, plus any `reasons` the caller found (a failed turn, a held
        intent). `rejected` is final: a bad proposal won't improve by retrying the same batch. Returns True if the batch
        settled."""
        reasons = list(reasons) + [r["code"] or "retryable" for r in self.db.execute(
            "select code from outcomes where attempt = ? and outcome = 'retryable'", (attempt,))]
        with self.db:
            if reasons:
                self.db.execute("update attempts set status = 'retryable', finished_at = ?, retry_reasons = ? where attempt = ?",
                                (self.clock(), ",".join(sorted(set(reasons))), attempt))
                return False
            self.db.execute("update attempts set status = 'completed', finished_at = ? where attempt = ?", (self.clock(), attempt))
            self.db.execute("update batches set status = 'settled' where batch = ?", (batch.batch,))
            # Never move a cursor backwards (an older, narrower batch can't undo a newer one). The byte offset moves
            # with it, from the batches row (a Batch rebuilt elsewhere may not carry it).
            self.db.execute(
                "update sessions set byte_offset = case when ? > cursor then "
                "(select end_offset from batches where batch = ?) else byte_offset end, cursor = max(cursor, ?) "
                "where session = ?", (batch.end_line, batch.batch, batch.end_line, batch.session))
        return True

    def batch_settled(self, batch):
        row = self.db.execute("select status from batches where batch = ?", (batch.batch,)).fetchone()
        return bool(row) and row["status"] == "settled"

    def attempt(self, attempt):
        row = self.db.execute("select * from attempts where attempt = ?", (attempt,)).fetchone()
        return dict(row) if row else None

    def fail_attempt(self, attempt, reason):
        """The observer turn itself failed (API error, crash): nothing settled, the cursor stays."""
        with self.db:
            self.db.execute("update attempts set status = 'retryable', finished_at = ?, retry_reasons = ? where attempt = ?",
                            (self.clock(), reason, attempt))

    # ------------------------------------------------------------------ cards

    def _check_evidence(self, bucket, evidence, batch):
        if not evidence:
            raise Rejected("no_evidence")
        for e in evidence:
            if e.role not in ROLES:
                raise Rejected("bad_evidence_role", e.role)
            if e.role not in BUCKETS[bucket].evidence_roles:
                raise Rejected("evidence_role_not_allowed", f"{bucket} can't be grounded by {e.role}")
            if not (batch.start_line <= e.line < batch.end_line):
                raise Rejected("evidence_outside_batch", str(e.line))
            if not e.quote.strip():
                raise Rejected("empty_quote")

    def add_card(self, batch, bucket, title, fields, evidence, body="", valid_from=None, supersedes=None, dedupe_key=None):
        """Insert a new card (version 1), or a new version that supersedes an active card. Returns the card id.
        Raises Rejected with a reason code when the proposal doesn't fit."""
        fields = validate_fields(bucket, fields)
        valid_from = valid_from or fields.get("valid_from")  # bucket field feeds the bitemporal column
        if not title or not title.strip():
            raise Rejected("missing_title")
        self._check_evidence(bucket, evidence, batch)
        now = self.clock()
        card = new_id("c")
        with self.db:
            if supersedes:
                old = self.db.execute("select * from cards where card = ?", (supersedes,)).fetchone()
                if old is None:
                    raise Rejected("supersede_unknown_card", supersedes)
                if old["status"] != "active":
                    raise Rejected("supersede_inactive_card", old["status"])
                if old["bucket"] != bucket:
                    raise Rejected("supersede_other_bucket", f"{old['bucket']} -> {bucket}")
                lineage, version = old["lineage"], old["version"] + 1
            else:
                lineage, version = new_id("l"), 1
            self.db.execute(
                "insert into cards (card, lineage, version, bucket, title, body, fields, status, valid_from, recorded_at, last_seen_at, "
                "dedupe_key) values (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)",
                (card, lineage, version, bucket, title.strip(), body.strip(), json.dumps(fields, sort_keys=True, ensure_ascii=False),
                 valid_from, now, now, dedupe_key))
            if supersedes:
                self.db.execute("update cards set status = 'superseded', superseded_at = ?, superseded_by = ? where card = ?",
                                (now, card, supersedes))
                self.db.execute("insert into links values (?, ?, 'supersedes', ?)", (card, supersedes, now))
            self.db.executemany("insert into evidence values (?, ?, ?, ?, ?, ?, ?)",
                                [(card, batch.batch, e.line, e.role, e.quote, e.evidence_id, e.kind) for e in evidence])
            terms = tokens(" ".join([title, body, *fields.values()]))
            self.db.execute("insert into card_search values (?, ?)", (card, " ".join(sorted(terms))))
        return card

    def reinforce(self, batch, card, evidence):
        """Seeing a card again: strengthen it and keep the new evidence. Only active cards can be reinforced."""
        row = self.db.execute("select bucket, status from cards where card = ?", (card,)).fetchone()
        if row is None:
            raise Rejected("reinforce_unknown_card", card)
        if row["status"] != "active":
            raise Rejected("reinforce_inactive_card", row["status"])
        self._check_evidence(row["bucket"], evidence, batch)
        with self.db:
            self.db.execute("update cards set reinforcements = reinforcements + 1, last_seen_at = ? where card = ?",
                            (self.clock(), card))
            self.db.executemany("insert into evidence values (?, ?, ?, ?, ?, ?, ?)",
                                [(card, batch.batch, e.line, e.role, e.quote, e.evidence_id, e.kind) for e in evidence])

    def link(self, src, dst, kind):
        if kind == "supersedes":
            raise Rejected("supersede_via_add_card")  # versioning must go through add_card so the old version closes
        with self.db:
            self.db.execute("insert or ignore into links values (?, ?, ?, ?)", (src, dst, kind, self.clock()))

    # ------------------------------------------------------------------ intents ("remember that ...")

    def add_intent(self, intent, session, batch, evidence_id, line, text):
        with self.db:
            self.db.execute("insert or ignore into intents values (?, ?, ?, ?, ?, ?, ?)",
                            (intent, session, batch.batch, evidence_id, line, text, self.clock()))
        return dict(self.db.execute("select * from intents where intent = ?", (intent,)).fetchone())

    def record_intent_outcome(self, intent, attempt, batch, outcome, card=None, reason=None):
        with self.db:
            self.db.execute("insert into intent_outcomes values (?, ?, ?, ?, ?, ?, ?)",
                            (intent, attempt, batch.batch, outcome, card, reason, self.clock()))

    def intent_resolved(self, intent):
        """True once the intent has a final outcome (anything but retryable / deferred), in any attempt."""
        return self.db.execute("select 1 from intent_outcomes where intent = ? and outcome not in ('retryable', 'deferred')",
                               (intent,)).fetchone() is not None

    def intent_deferrals(self, intent):
        return self.db.execute("select count(*) from intent_outcomes where intent = ? and outcome = 'deferred'",
                               (intent,)).fetchone()[0]

    # ------------------------------------------------------------------ reviews (human inbox)

    def add_review(self, review, relation, cards, reason, batch):
        """Queue a suggested relation for a human. Returns False if the same suggestion is already queued."""
        with self.db:
            cursor = self.db.execute("insert or ignore into reviews values (?, ?, ?, ?, 'pending', ?, ?, null)",
                                     (review, relation, json.dumps(cards), reason, batch.batch, self.clock()))
        return cursor.rowcount == 1

    def decide_review(self, review, accept):
        """A human's decision on a suggested relation. Accepting applies it; both are recorded on the review.

        relates_to / contradicts link every pair. supersedes: the newest card (by recorded time) replaces the others.
        same_as: the oldest card stays, the others retire and their evidence reinforces it. Both of those need one
        bucket: a human confirming a relation doesn't waive a bucket's evidence rules (a decision's assistant evidence
        must never land on a world_update)."""
        row = self.db.execute("select * from reviews where review = ?", (review,)).fetchone()
        if row is None:
            raise Rejected("unknown_review", review)
        if row["status"] != "pending":
            raise Rejected("review_already_decided", row["status"])
        now = self.clock()
        cards = [self.get(c) for c in json.loads(row["cards"])]
        if accept and any(c is None or c["status"] != "active" for c in cards):
            raise Rejected("review_cards_not_current", "a card changed since the suggestion; dismiss it instead")
        if accept and row["relation"] in ("supersedes", "same_as") and len({c["bucket"] for c in cards}) > 1:
            raise Rejected("review_other_bucket", ", ".join(sorted({c["bucket"] for c in cards})))
        cards.sort(key=lambda c: (c["recorded_at"], c["card"]))
        with self.db:
            if accept and row["relation"] in ("relates_to", "contradicts"):
                self.db.executemany("insert or ignore into links values (?, ?, ?, ?)",
                                    [(a["card"], b["card"], row["relation"], now)
                                     for i, a in enumerate(cards) for b in cards[i + 1:]])
            elif accept and row["relation"] == "supersedes":
                newest = cards[-1]
                for old in cards[:-1]:
                    self.db.execute("update cards set status = 'superseded', superseded_at = ?, superseded_by = ? "
                                    "where card = ?", (now, newest["card"], old["card"]))
                    self.db.execute("insert or ignore into links values (?, ?, 'supersedes', ?)", (newest["card"], old["card"], now))
            elif accept and row["relation"] == "same_as":
                keep = cards[0]
                for dup in cards[1:]:
                    self.db.execute("update cards set status = 'retired', superseded_at = ?, superseded_by = ? where card = ?",
                                    (now, keep["card"], dup["card"]))
                    self.db.execute("insert into evidence select ?, batch, line, role, quote, evidence_id, kind "
                                    "from evidence where card = ?", (keep["card"], dup["card"]))
                    self.db.execute("update cards set reinforcements = reinforcements + 1 + ?, last_seen_at = ? where card = ?",
                                    (dup["reinforcements"], now, keep["card"]))
            self.db.execute("update reviews set status = ?, decided_at = ? where review = ?",
                            ("accepted" if accept else "dismissed", now, review))

    def reviews(self, status="pending"):
        rows = self.db.execute("select * from reviews where status = ? order by created_at", (status,))
        return [dict(r, cards=json.loads(r["cards"])) for r in rows]

    # ------------------------------------------------------------------ whispers

    def queue_whisper(self, whisper, session, batch, card, text, related=(), delivered=False):
        """Queue a whisper for the session's next prompt (or record one the sync lane already delivered). `related` are
        the linked cards woven into the text: they count as known to the session too. False if already queued."""
        now = self.clock()
        with self.db:
            cursor = self.db.execute("insert or ignore into whispers values (?, ?, ?, ?, ?, ?, ?, ?)",
                                     (whisper, session, batch, card, text, now, now if delivered else None,
                                      json.dumps(list(related))))
        return cursor.rowcount == 1

    def session_known(self, session):
        """Cards a session shouldn't be whispered: ones it was already told about, and ones grounded by its own
        transcript (it just said them)."""
        written = {r[0] for r in self.db.execute(
            "select distinct e.card from evidence e join batches b on b.batch = e.batch where b.session = ?", (session,))}
        return sorted(written | set(self.whispered_cards(session)))

    def whispered_cards(self, session):
        """Cards this session has already been told about (directly or as related context)."""
        known = set()
        for card, related in self.db.execute("select card, related from whispers where session = ?", (session,)):
            known.add(card)
            known.update(json.loads(related))
        return sorted(known)

    def pending_whispers(self, session):
        return [dict(r) for r in self.db.execute(
            "select * from whispers where session = ? and delivered_at is null order by created_at", (session,))]

    def mark_delivered(self, whispers):
        with self.db:
            self.db.executemany("update whispers set delivered_at = ? where whisper = ?",
                                [(self.clock(), w) for w in whispers])

    # ------------------------------------------------------------------ reads

    def get(self, card):
        row = self.db.execute("select * from cards where card = ?", (card,)).fetchone()
        return self._card(row) if row else None

    def current(self, bucket=None):
        sql, args = "select * from cards where status = 'active'", []
        if bucket:
            sql, args = sql + " and bucket = ?", [bucket]
        return [self._card(r) for r in self.db.execute(sql + " order by recorded_at, card", args)]

    def as_of(self, when, bucket=None):
        """What tacit believed at `when` (recorded time): versions recorded by then and not yet superseded."""
        sql = ("select * from cards where recorded_at <= ? and (superseded_at is null or superseded_at > ?) "
               "and status in ('active', 'superseded')")
        args = [when, when]
        if bucket:
            sql, args = sql + " and bucket = ?", args + [bucket]
        return [self._card(r) for r in self.db.execute(sql + " order by recorded_at, card", args)]

    # A new version and a replacement accepted in review both write one `supersedes` link, newer → older. Two views
    # read those links, never timestamps alone (two cards can share a recorded second):
    # - history: the whole family (every card reachable through supersedes links in either direction), the same
    #   list from any member, each card once, every card after the cards it replaced;
    # - predecessors: only what one card replaced, directly or transitively (the "was …" lines of a whisper).

    def history(self, card):
        """Every card in this card's replacement family, oldest first (see above). Ties: recorded time, then id."""
        if self.get(card) is None:
            return []
        family = self._supersedes_walk(card, both=True) | {card}
        marks = ",".join("?" * len(family))
        cards = {r["card"]: self._card(r) for r in self.db.execute(f"select * from cards where card in ({marks})", list(family))}
        replaced = {c: set() for c in family}  # card -> the cards it replaced
        for src, dst in self.db.execute(f"select src, dst from links where kind = 'supersedes' and src in ({marks}) "
                                        f"and dst in ({marks})", list(family) * 2):
            replaced[src].add(dst)
        order = []
        while replaced:  # topological order: a card is ready once everything it replaced is placed
            ready = [c for c, olds in replaced.items() if not olds - set(order)] or list(replaced)  # a cycle can't stall it
            nxt = min(ready, key=lambda c: (cards[c]["recorded_at"], c))
            order.append(nxt)
            del replaced[nxt]
        return [cards[c] for c in order]

    def predecessors(self, card):
        """Card ids this card replaced, directly or through older replacements, nearest first (breadth-first, newest
        sibling first)."""
        return self._supersedes_walk(card)

    def _supersedes_walk(self, card, both=False):
        """Breadth-first along `supersedes` links from newer to older (or both ways); each card once, start excluded.
        Returns an ordered list, or a set when `both`."""
        seen, order, frontier = {card}, [], [card]
        sql = ("select c.card from links l join cards c on c.card = l.dst where l.kind = 'supersedes' and l.src in ({m}) "
               "order by c.recorded_at desc, c.card")
        if both:
            sql = ("select l.dst from links l where l.kind = 'supersedes' and l.src in ({m}) "
                   "union select l.src from links l where l.kind = 'supersedes' and l.dst in ({m})")
        while frontier:
            marks = ",".join("?" * len(frontier))
            step = [r[0] for r in self.db.execute(sql.format(m=marks), frontier * (2 if both else 1))]
            frontier = [c for c in dict.fromkeys(step) if c not in seen]
            seen.update(frontier)
            order += frontier
        return set(order) if both else order

    def search(self, query, limit=5, bucket=None, exclude=()):
        """Full-text search over active cards. Returns cards, best match first."""
        terms = " OR ".join(f'"{t}"' for t in sorted(tokens(query)))
        if not terms:
            return []
        sql = ("select c.* from card_search s join cards c on c.card = s.card "
               "where card_search match ? and c.status = 'active'")
        args = [terms]
        if bucket:
            sql, args = sql + " and c.bucket = ?", args + [bucket]
        if exclude:
            sql, args = sql + f" and c.card not in ({','.join('?' * len(exclude))})", args + list(exclude)
        rows = self.db.execute(sql + " order by rank limit ?", args + [limit])
        return [self._card(r) for r in rows]

    def find_active(self, bucket, title, fields):
        """An active card with the same bucket, title (case-insensitive) and fields: the same knowledge again."""
        for row in self.db.execute("select * from cards where status = 'active' and bucket = ? and lower(title) = ?",
                                   (bucket, title.strip().lower())):
            if json.loads(row["fields"]) == fields:
                return self._card(row)
        return None

    def card_by_dedupe(self, dedupe_key):
        row = self.db.execute("select * from cards where dedupe_key = ? order by recorded_at desc limit 1",
                              (dedupe_key,)).fetchone()
        return self._card(row) if row else None

    def evidence(self, card):
        return [dict(r) for r in self.db.execute("select line, role, quote, batch, evidence_id, kind from evidence where card = ? order by rowid", (card,))]

    def links_of(self, card):
        return [dict(r) for r in self.db.execute("select src, dst, kind from links where src = ? or dst = ?", (card, card))]

    @staticmethod
    def _card(row):
        card = dict(row)
        card["fields"] = json.loads(card["fields"])
        return card
