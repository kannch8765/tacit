"""Ledger rules: bucket validation, SCD-2 versioning, reinforcement, search, and a cursor that only moves on settle."""
import datetime
import io
import json

import pytest

from tacit import cli
from tacit.buckets import Rejected, validate_fields
from tacit.store import Evidence, Store


class Clock:
    """Deterministic UTC clock: each call is one second later."""

    def __init__(self):
        self.t = datetime.datetime(2026, 10, 19, tzinfo=datetime.timezone.utc)

    def __call__(self):
        self.t += datetime.timedelta(seconds=1)
        return self.t.isoformat(timespec="seconds")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "tacit.db", clock=Clock())
    yield s
    s.close()


def transcript(path, n, start=0):
    with open(path, "a") as f:
        for i in range(start, start + n):
            f.write(json.dumps({"type": "user", "message": {"role": "user", "content": f"line {i}"}}) + "\n")
    return path


def batch_for(store, tmp_path, n=10):
    t = transcript(tmp_path / "s1.jsonl", n)
    store.enqueue("s1", t)
    return store.next_batch("s1")


def ev(line=0, role="user", quote="the quote"):
    return [Evidence(line, role, quote)]


# ---------------------------------------------------------------- buckets

@pytest.mark.parametrize("bucket, fields, code", [
    ("nope", {}, "unknown_bucket"),
    ("decision", {}, "missing_field"),
    ("decision", {"rationale": "  "}, "missing_field"),
    ("gotcha", {"symptom": "flaky"}, "missing_field"),
    ("preference", {"who": "mika", "mood": "happy"}, "unknown_field"),
    ("world_update", {"source": "assistant"}, "bad_field_value"),
    ("workflow_habit", {"trigger": 3}, "bad_field_type"),
])
def test_bucket_validation_rejects_with_codes(bucket, fields, code):
    with pytest.raises(Rejected) as error:
        validate_fields(bucket, fields)
    assert error.value.code == code


def test_world_update_must_be_grounded_outside_the_model(store, tmp_path):
    batch = batch_for(store, tmp_path)
    with pytest.raises(Rejected) as error:
        store.add_card(batch, "world_update", "PM is now X", {"source": "user"}, ev(role="assistant"))
    assert error.value.code == "evidence_role_not_allowed"
    assert store.add_card(batch, "world_update", "PM is now X", {"source": "tool"}, ev(role="tool"))


def test_evidence_must_sit_inside_the_batch(store, tmp_path):
    batch = batch_for(store, tmp_path, n=3)
    for evidence, code in ((ev(line=3), "evidence_outside_batch"), ([], "no_evidence"), (ev(quote=" "), "empty_quote")):
        with pytest.raises(Rejected) as error:
            store.add_card(batch, "decision", "pin pydantic < 3", {"rationale": "v3 breaks validators"}, evidence)
        assert error.value.code == code


# ---------------------------------------------------------------- versioning

def test_supersede_keeps_history_and_answers_as_of(store, tmp_path):
    batch = batch_for(store, tmp_path)
    old = store.add_card(batch, "world_update", "Japan's prime minister is Ishiba", {"source": "user"}, ev(line=1))
    before = store.clock()
    new = store.add_card(batch, "world_update", "Japan's prime minister is X",
                         {"source": "user", "replaces": "Ishiba is PM", "valid_from": "2026-10-01"}, ev(line=5),
                         supersedes=old)

    assert [c["card"] for c in store.current("world_update")] == [new]
    assert [(c["card"], c["version"], c["status"]) for c in store.history(new)] == [(old, 1, "superseded"), (new, 2, "active")]
    assert store.get(old)["superseded_by"] == new
    assert store.get(new)["valid_from"] == "2026-10-01"
    assert {"src": new, "dst": old, "kind": "supersedes"} in store.links_of(old)
    assert [c["card"] for c in store.as_of(before)] == [old]  # what we believed before the update
    assert [c["card"] for c in store.as_of(store.clock())] == [new]
    assert store.evidence(old)[0]["line"] == 1  # the old version keeps its own evidence


@pytest.mark.parametrize("setup, code", [
    ("superseded", "supersede_inactive_card"),
    ("other_bucket", "supersede_other_bucket"),
    ("unknown", "supersede_unknown_card"),
])
def test_supersede_rules(store, tmp_path, setup, code):
    batch = batch_for(store, tmp_path)
    first = store.add_card(batch, "decision", "use poetry", {"rationale": "lockfile"}, ev())
    target = {"unknown": "c_missing", "other_bucket": first}.get(setup, first)
    if setup == "superseded":
        store.add_card(batch, "decision", "use uv", {"rationale": "faster"}, ev(), supersedes=first)
    bucket = "gotcha" if setup == "other_bucket" else "decision"
    fields = {"symptom": "s", "fix": "f"} if bucket == "gotcha" else {"rationale": "r"}
    with pytest.raises(Rejected) as error:
        store.add_card(batch, bucket, "title", fields, ev(), supersedes=target)
    assert error.value.code == code
    assert len(store.current()) == 1  # the rejected proposal left nothing behind


def test_supersedes_links_only_through_add_card(store, tmp_path):
    batch = batch_for(store, tmp_path)
    a = store.add_card(batch, "decision", "a", {"rationale": "r"}, ev())
    b = store.add_card(batch, "decision", "b", {"rationale": "r"}, ev())
    with pytest.raises(Rejected):
        store.link(b, a, "supersedes")
    store.link(b, a, "relates_to")
    assert store.links_of(a) == [{"src": b, "dst": a, "kind": "relates_to"}]


def test_reinforce_counts_and_keeps_evidence(store, tmp_path):
    batch = batch_for(store, tmp_path)
    habit = store.add_card(batch, "workflow_habit", "branch before committing", {"trigger": "before git commit"}, ev(line=0))
    store.reinforce(batch, habit, ev(line=4, quote="made a branch first again"))
    card = store.get(habit)
    assert card["reinforcements"] == 1 and len(store.evidence(habit)) == 2
    new = store.add_card(batch, "workflow_habit", "branch, then test", {"trigger": "before git commit"}, ev(), supersedes=habit)
    with pytest.raises(Rejected) as error:
        store.reinforce(batch, habit, ev())
    assert error.value.code == "reinforce_inactive_card" and store.get(new)["reinforcements"] == 0


def test_search_returns_active_cards_only(store, tmp_path):
    batch = batch_for(store, tmp_path)
    old = store.add_card(batch, "decision", "pin pydantic below 3", {"rationale": "validators break on v3"}, ev())
    store.add_card(batch, "gotcha", "flaky cache test", {"symptom": "random failure", "fix": "await warmup"}, ev())
    assert [c["card"] for c in store.search("pydantic")] == [old]
    new = store.add_card(batch, "decision", "pydantic 3 is fine now", {"rationale": "validators fixed"}, ev(), supersedes=old)
    assert [c["card"] for c in store.search("pydantic")] == [new]
    assert [c["title"] for c in store.search("warmup")] == ["flaky cache test"]  # fields are searchable
    assert store.search('" ') == []


# ---------------------------------------------------------------- cursor + outcomes

def test_cursor_moves_only_when_the_batch_settles(store, tmp_path):
    path = transcript(tmp_path / "s1.jsonl", 5)
    store.enqueue("s1", path)
    batch = store.next_batch("s1")
    assert (batch.start_line, batch.end_line, len(batch.lines)) == (0, 5, 5)

    attempt = store.begin_attempt(batch)
    store.record_outcome(attempt, batch, "p1", "accepted", card=None)
    store.record_outcome(attempt, batch, "p2", "retryable", code="api_timeout")
    assert not store.finish_attempt(attempt, batch)
    assert store.cursor("s1") == 0  # held

    transcript(path, 3, start=5)  # the session went on; the retry sees a wider batch from the same line
    retry = store.next_batch("s1")
    assert (retry.start_line, retry.end_line) == (0, 8) and retry.batch != batch.batch
    attempt = store.begin_attempt(retry)
    store.record_outcome(attempt, retry, "p3", "rejected", code="no_evidence")  # final, not a retry reason
    assert store.finish_attempt(attempt, retry)
    assert store.cursor("s1") == 8 and store.next_batch("s1") is None

    store.finish_attempt(store.begin_attempt(batch), batch)  # a late, narrower batch settling...
    assert store.cursor("s1") == 8  # ...never moves the cursor backwards
    reasons = store.db.execute("select retry_reasons from attempts where status = 'retryable'").fetchone()[0]
    assert reasons == "api_timeout"


def test_failed_observer_turn_keeps_the_cursor(store, tmp_path):
    batch = batch_for(store, tmp_path)
    attempt = store.begin_attempt(batch)
    store.fail_attempt(attempt, "turn_failed")
    assert store.cursor("s1") == 0 and store.next_batch("s1").start_line == 0


def test_batches_are_capped_and_missing_transcripts_are_skipped(store, tmp_path):
    store.enqueue("s1", transcript(tmp_path / "s1.jsonl", 10))
    assert store.next_batch("s1", max_lines=4).end_line == 4
    store.enqueue("gone", tmp_path / "missing.jsonl")
    assert store.next_batch("gone") is None and store.next_batch("never-seen") is None


# ---------------------------------------------------------------- cli

def test_stop_hook_queues_and_never_crashes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TACIT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    cli.main(["hook", "stop"])
    assert "ignored" in capsys.readouterr().err
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": "s1", "transcript_path": "/x.jsonl"})))
    cli.main(["hook", "stop"])
    cli.main(["stats"])
    assert json.loads(capsys.readouterr().out)["totals"]["sessions"] == 1


def test_search_finds_japanese_by_any_piece(store, tmp_path):
    batch = batch_for(store, tmp_path)
    card = store.add_card(batch, "gotcha", "ステージングのデプロイはVPN必須",
                          {"symptom": "タイムアウト", "fix": "VPNをオンにする"}, ev())
    for query in ("VPN", "ステージング", "デプロイ", "vpn 必須", "ﾃﾞﾌﾟﾛｲ"):  # half-width kana normalise too
        assert [c["card"] for c in store.search(query)] == [card], query
    assert store.search("本番") == []


@pytest.mark.parametrize("value", ["N/A", "n/a", "none", "-", "Unknown."])
def test_placeholder_values_are_not_knowledge(value):
    # Live run: the model filled optional fields with "N/A"
    assert validate_fields("world_update", {"source": "tool", "replaces": value}) == {"source": "tool"}
    with pytest.raises(Rejected) as error:
        validate_fields("decision", {"rationale": value})
    assert error.value.code == "missing_field"


# Review finding: only complete records are read, from a committed byte offset
def test_a_half_written_last_line_waits_until_it_is_complete(store, tmp_path):
    from fake_llm import FakeClient
    from tacit import observer
    path = tmp_path / "s1.jsonl"
    raw = json.dumps({"type": "user", "uuid": "u1", "message": {"content": "remember that staging needs the VPN"}})
    path.write_text(raw[:40])
    store.enqueue("s1", path)
    assert store.next_batch("s1") is None and store.cursor("s1") == 0
    with path.open("a") as f:
        f.write(raw[40:] + "\n")
    batch = store.next_batch("s1")
    assert (batch.start_line, batch.end_line) == (0, 1) and json.loads(batch.lines[0])["uuid"] == "u1"
    assert [i["text"] for i in observer.detect_intents(store, batch, observer.transcript.events(batch))] == [
        "staging needs the VPN"]


def test_reading_resumes_from_the_settled_byte_offset(store, tmp_path):
    batch = batch_for(store, tmp_path, n=3)
    a = store.begin_attempt(batch)
    assert store.finish_attempt(a, batch)
    path = transcript(tmp_path / "s1.jsonl", 2, start=3)
    with path.open("a") as f:
        f.write('{"type": "user", "message": {"content": "half')
    nxt = store.next_batch("s1")
    assert (nxt.start_line, nxt.end_line) == (3, 5)
    assert [json.loads(x)["message"]["content"] for x in nxt.lines] == ["line 3", "line 4"]


def test_old_stores_without_an_offset_resume_at_their_line_cursor(store, tmp_path):
    batch = batch_for(store, tmp_path, n=3)
    store.finish_attempt(store.begin_attempt(batch), batch)
    store.db.execute("update sessions set byte_offset = 0")
    transcript(tmp_path / "s1.jsonl", 1, start=3)
    nxt = store.next_batch("s1")
    assert nxt.start_line == 3 and json.loads(nxt.lines[0])["message"]["content"] == "line 3"


def test_broken_complete_records_leave_a_diagnostic_not_a_silent_skip(store, tmp_path):
    from fake_llm import FakeClient
    from tacit import observer
    path = tmp_path / "s1.jsonl"
    path.write_text('{"type": "summary", "summary": "fine"}\n{not json\n')
    store.enqueue("s1", path)
    batch = store.next_batch("s1")
    assert observer.observe(store, batch, FakeClient()).settled
    observer.observe(store, batch, FakeClient())  # a rerun doesn't duplicate it
    assert [(d["session"], d["line"], d["code"]) for d in store.diagnostics()] == [("s1", 1, "malformed_record")]


def test_a_transcript_that_shrank_is_reported_not_reread(store, tmp_path):
    batch = batch_for(store, tmp_path, n=3)
    store.finish_attempt(store.begin_attempt(batch), batch)
    (tmp_path / "s1.jsonl").write_text("")
    assert store.next_batch("s1") is None
    assert [d["code"] for d in store.diagnostics()] == ["transcript_shrank"]
