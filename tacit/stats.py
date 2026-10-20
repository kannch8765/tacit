"""Observability from the ledger: a daily rollup and a healthcheck. Numbers and fixed names only, never card text, so
the output can be shared or charted without showing anyone's knowledge.

tacit's ledger already is the telemetry store, so there is no collector. Days are local dates in `tz`.
"""
import collections
import datetime
import json

LEVELS = ["ok", "warn", "alert"]
RESOLVED_INTENT = {"accepted", "duplicate", "reinforced", "superseded", "declined", "rejected"}
PROPOSAL_OUTCOMES = ("accepted", "reinforced", "duplicate", "superseded", "rejected", "retryable")

# name -> (warn, alert); None = that level is not used
THRESHOLDS = {
    "batch_retryable_share_24h": (0.10, 0.25),
    "proposal_rejected_share_24h": (0.30, 0.60),  # the observer keeps proposing things it can't ground
    "observer_turn_p90_s_24h": (120, 300),
    "held_sessions": (1, 3),  # sessions whose newest attempt didn't settle and is over an hour old
    "unresolved_intent_age_h": (1, 24),
    "pending_reviews": (10, 50),
}


def parse_ts(value):
    if not value:
        return None
    try:
        t = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def pct(values, q):
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def rows(store, sql, params=()):
    return [dict(r) for r in store.db.execute(sql, params)]


class Acc:
    """metrics[day][metric] with helpers for counts and percentile samples."""

    def __init__(self, tz, days):
        self.tz, self.days = tz, set(days)
        self.counts = collections.defaultdict(lambda: collections.defaultdict(float))
        self.samples = collections.defaultdict(lambda: collections.defaultdict(list))

    def day(self, ts):
        t = parse_ts(ts)
        d = t.astimezone(self.tz).date().isoformat() if t else None
        return d if d in self.days else None

    def add(self, day, metric, value=1):
        self.counts[day][metric] += value

    def sample(self, day, metric, value):
        if value is not None:
            self.samples[day][metric].append(value)

    def result(self):
        out = {day: dict(metrics) for day, metrics in self.counts.items()}
        for day, metrics in self.samples.items():
            for metric, values in metrics.items():
                out.setdefault(day, {})[f"{metric}_p50"] = pct(values, 0.5)
                out.setdefault(day, {})[f"{metric}_p90"] = pct(values, 0.9)
        for metrics in out.values():  # derived shares
            for share, part, whole in (("retryable_share", "retryable", "attempts"),
                                       ("rejected_share", "proposal:rejected", "proposals")):
                if metrics.get(whole):
                    metrics[share] = metrics.get(part, 0) / metrics[whole]
        return out


def compute(store, tz, days):
    """{day: {metric: value}} for the given local days."""
    acc = Acc(tz, days)
    for r in rows(store, "select * from attempts where status != 'running'"):
        if not (d := acc.day(r["finished_at"])):
            continue
        acc.add(d, "attempts")
        acc.add(d, "completed" if r["status"] == "completed" else "retryable")
        for reason in (r["retry_reasons"] or "").split(","):
            if reason:
                acc.add(d, f"retry_reason:{reason}")
        start, end = parse_ts(r["started_at"]), parse_ts(r["finished_at"])
        if start and end:
            acc.sample(d, "observer_turn_s", (end - start).total_seconds())
    for r in rows(store, "select outcome, code, recorded_at from outcomes"):
        if d := acc.day(r["recorded_at"]):
            acc.add(d, "proposals")
            acc.add(d, f"proposal:{r['outcome']}")
            if r["outcome"] == "rejected" and r["code"]:
                acc.add(d, f"rejection:{r['code']}")
    for r in rows(store, "select created_at from intents"):
        if d := acc.day(r["created_at"]):
            acc.add(d, "intents_captured")
    for r in rows(store, "select outcome, recorded_at from intent_outcomes"):
        if (d := acc.day(r["recorded_at"])) and r["outcome"] in RESOLVED_INTENT:
            acc.add(d, "intents_resolved")
    for r in rows(store, "select batch, created_at from whispers"):
        if d := acc.day(r["created_at"]):
            acc.add(d, "whispers")
            acc.add(d, "whisper:sync" if r["batch"] == "sync" else "whisper:async")
    for r in rows(store, "select recorded_at, version from cards"):
        if d := acc.day(r["recorded_at"]):
            acc.add(d, "cards_new" if r["version"] == 1 else "cards_new_version")
    for r in rows(store, "select created_at from reviews"):
        if d := acc.day(r["created_at"]):
            acc.add(d, "reviews_queued")
    return acc.result()


def recent_days(now, tz, n):
    today = now.astimezone(tz).date()
    return [(today - datetime.timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


def backlog(store):
    """Transcript lines past each session's cursor (reads line counts only)."""
    total, sessions = 0, 0
    for r in rows(store, "select transcript_path, cursor from sessions"):
        try:
            with open(r["transcript_path"], "rb") as f:
                lines = sum(1 for _ in f)
        except OSError:
            continue
        gap = max(0, lines - r["cursor"])
        total += gap
        sessions += gap > 0
    return {"backlog_lines": total, "backlog_sessions": sessions}


# ---------------------------------------------------------------------------------------------------- health

def level(value, warn, alert):
    if value is None:
        return "ok"
    if alert is not None and value >= alert:
        return "alert"
    if warn is not None and value >= warn:
        return "warn"
    return "ok"


def values(store, now):
    day_ago = now - datetime.timedelta(days=1)
    recent = lambda ts: (t := parse_ts(ts)) is not None and t >= day_ago  # noqa: E731
    out = {}
    attempts = [r for r in rows(store, "select * from attempts where status != 'running'") if recent(r["finished_at"])]
    out["batch_retryable_share_24h"] = sum(r["status"] != "completed" for r in attempts) / len(attempts) if attempts else None
    outcomes = [r for r in rows(store, "select outcome, recorded_at from outcomes") if recent(r["recorded_at"])]
    out["proposal_rejected_share_24h"] = sum(r["outcome"] == "rejected" for r in outcomes) / len(outcomes) if outcomes else None
    turns = [(parse_ts(r["finished_at"]) - parse_ts(r["started_at"])).total_seconds() for r in attempts
             if parse_ts(r["started_at"]) and parse_ts(r["finished_at"])]
    out["observer_turn_p90_s_24h"] = pct(turns, 0.9)

    # A session is held when its newest attempt didn't settle and has stayed that way for over an hour
    newest = {}
    for r in rows(store, "select a.*, b.session from attempts a join batches b on b.batch = a.batch order by a.started_at"):
        newest[r["session"]] = r
    out["held_sessions"] = sum(1 for r in newest.values() if r["status"] == "retryable"
                               and (t := parse_ts(r["finished_at"])) and now - t > datetime.timedelta(hours=1))

    latest = {}
    for r in rows(store, "select intent, outcome from intent_outcomes order by rowid"):
        if r["outcome"] != "deferred":
            latest[r["intent"]] = r["outcome"]
    ages = [(now - t).total_seconds() / 3600 for r in rows(store, "select intent, created_at from intents")
            if latest.get(r["intent"]) not in RESOLVED_INTENT and (t := parse_ts(r["created_at"]))]
    out["unresolved_intent_age_h"] = max(ages) if ages else None
    out["unresolved_intents"] = len(ages)
    out["pending_reviews"] = store.db.execute("select count(*) from reviews where status = 'pending'").fetchone()[0]
    return out


def check(store, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    measured = values(store, now)
    checks = []
    for name, (warn, alert) in THRESHOLDS.items():
        value = measured.get(name)
        checks.append({"name": name, "level": level(value, warn, alert),
                       "value": round(value, 4) if isinstance(value, float) else value, "warn": warn, "alert": alert})
    overall = max((c["level"] for c in checks), key=LEVELS.index, default="ok")
    return {"schema": "tacit_health_v1", "generated_at": now.isoformat(timespec="seconds"), "overall": overall,
            "checks": checks, "info": {"unresolved_intents": measured["unresolved_intents"], **backlog(store)}}


def summary_json(store, tz, days):
    """`tacit stats`: per-day numbers plus current totals."""
    totals = {k: store.db.execute(sql).fetchone()[0] for k, sql in {
        "sessions": "select count(*) from sessions",
        "cards_active": "select count(*) from cards where status = 'active'",
        "cards_superseded": "select count(*) from cards where status = 'superseded'",
        "reviews_pending": "select count(*) from reviews where status = 'pending'",
        "whispers_pending": "select count(*) from whispers where delivered_at is null",
    }.items()}
    return json.dumps({"schema": "tacit_stats_v1", "totals": {**totals, **backlog(store)},
                       "days": compute(store, tz, days)}, indent=2, sort_keys=True)
