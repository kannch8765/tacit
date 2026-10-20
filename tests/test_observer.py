"""Observer turns end to end with a scripted model: grounding, idempotent retries, intents, and the ownership-based
batch outcome (only owned work that isn't durably settled fails a batch)."""
import datetime
import json
import sqlite3

import pytest
from fake_llm import FakeClient

from tacit import observer, transcript
from tacit.runtime import Runtime
from tacit.store import Store, stable_id


class Clock:
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


def user(text, uid):
    return {"type": "user", "uuid": uid, "timestamp": "2026-10-19T01:00:00Z", "message": {"role": "user", "content": text}}


def assistant(text, uid):
    return {"type": "assistant", "uuid": uid, "timestamp": "2026-10-19T01:00:05Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def tool_result(text, uid, tool_use="t1"):
    return {"type": "user", "uuid": uid, "message": {"content": [{"type": "tool_result", "tool_use_id": tool_use, "content": text}]}}


DEFAULT = [user("we pin pydantic < 3 because v3 breaks our validators", "u1"), assistant("Pinned.", "a1")]


@pytest.fixture
def session(store, tmp_path):
    """Write transcript lines for a session (s1 by default) and return the next batch plus its events."""

    def write(lines=DEFAULT, name="s1"):
        path = tmp_path / f"{name}.jsonl"
        with open(path, "a") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        store.enqueue(name, path)
        batch = store.next_batch(name)
        return batch, {e.quote: e.evidence_id for e in transcript.events(batch)}
    return write


SEARCH = ("search", {"query": "pydantic", "purpose": "maintenance"})


def decision(ev, **extra):
    return ("propose_decision", {"title": "Pin pydantic < 3", "fields": {"rationale": "v3 breaks our validators"},
                                 "evidence_ids": [ev], **extra})


# ---------------------------------------------------------------- grounding

def test_grounded_card_copies_the_quote_from_the_transcript(store, session):
    batch, ids = session()
    ev = ids["we pin pydantic < 3 because v3 breaks our validators"]
    result = observer.observe(store, batch, FakeClient([SEARCH], [decision(ev)]))
    assert result.settled and store.cursor("s1") == 2
    (card,) = store.current()
    assert card["title"] == "Pin pydantic < 3"
    (e,) = store.evidence(card["card"])
    assert (e["quote"], e["role"], e["evidence_id"]) == ("we pin pydantic < 3 because v3 breaks our validators", "user", ev)


@pytest.mark.parametrize("args, code", [
    ({"evidence_ids": ["ev_made_up"]}, "unresolvable_evidence"),
    ({"evidence_ids": []}, "no_evidence"),
    ({"evidence_ids": "nope"}, "no_evidence"),
    ({"quote": "we pin pydantic"}, "unknown_argument"),  # a retyped quote is not evidence
    ({"title": " "}, "missing_title"),
    ({"intent_id": "intent_nope"}, "unknown_intent"),
])
def test_ungrounded_proposals_are_rejected_with_codes_and_still_settle(store, session, args, code):
    batch, ids = session()
    client = FakeClient([SEARCH], [decision(next(iter(ids.values())), **args)])
    result = observer.observe(store, batch, client)
    assert client.tool_results("propose_decision")[0]["code"] == code
    assert store.current() == []
    assert result.settled  # a rejection is final: retrying the same batch won't improve it


def test_world_update_cannot_be_grounded_by_the_assistant(store, session):
    batch, ids = session([user("who leads the project now?", "u1"), assistant("Mika took over as PM in October.", "a1"),
                          tool_result("Org chart: PM = Mika (since 2026-10-01)", "u2")])
    proposal = {"title": "PM is Mika", "fields": {"source": "tool"}}
    client = FakeClient([("search", {"query": "PM", "purpose": "maintenance"})],
                        [("propose_world_update", {**proposal, "evidence_ids": [ids["Mika took over as PM in October."]]}),
                         ("propose_world_update", {**proposal, "evidence_ids": [ids["Org chart: PM = Mika (since 2026-10-01)"]]})])
    observer.observe(store, batch, client)
    first, second = client.tool_results("propose_world_update")
    assert first["code"] == "evidence_role_not_allowed" and second["outcome"] == "accepted"


def test_secrets_never_reach_the_model(store, session):
    key = "sk-ant-api03-" + "k" * 40
    batch, _ = session([user(f"the staging key is {key}, never commit it", "u1")])
    client = FakeClient([("search", {"query": "staging key", "purpose": "maintenance"})])
    observer.observe(store, batch, client)
    assert key not in client.payload() and "[redacted:anthropic_key]" in client.payload()


# ---------------------------------------------------------------- dedupe, reinforce, supersede, idempotency

def test_the_same_knowledge_later_reinforces_instead_of_duplicating(store, session):
    batch, ids = session()
    observer.observe(store, batch, FakeClient([SEARCH], [decision(ids["we pin pydantic < 3 because v3 breaks our validators"])]))
    batch2, ids2 = session([user("reminder: pydantic stays < 3, v3 breaks our validators", "u2")])
    ev2 = ids2["reminder: pydantic stays < 3, v3 breaks our validators"]
    client = FakeClient([SEARCH], [decision(ev2)])
    observer.observe(store, batch2, client)
    (card,) = store.current()
    assert client.tool_results("propose_decision")[0]["outcome"] == "reinforced"
    assert card["reinforcements"] == 1 and len(store.evidence(card["card"])) == 2


def test_reinforce_tool_and_supersede(store, session):
    batch, ids = session()
    observer.observe(store, batch, FakeClient([SEARCH], [decision(ids["we pin pydantic < 3 because v3 breaks our validators"])]))
    (old,) = store.current()
    batch2, ids2 = session([user("v3 is fixed now; we moved to pydantic 3.2", "u2"), user("pin stays as a habit", "u3")])
    new_ev = ids2["v3 is fixed now; we moved to pydantic 3.2"]
    client = FakeClient([SEARCH], [
        ("reinforce", {"card": old["card"], "evidence_ids": [ids2["pin stays as a habit"]]}),
        ("propose_decision", {"title": "Use pydantic 3.2", "fields": {"rationale": "v3 is fixed now"},
                              "evidence_ids": [new_ev], "supersedes": old["card"]}),
        ("reinforce", {"card": old["card"], "evidence_ids": [new_ev]}),  # now inactive
    ])
    observer.observe(store, batch2, client)
    first, after = client.tool_results("reinforce")
    (superseded,) = client.tool_results("propose_decision")
    assert first["outcome"] == "reinforced" and superseded["outcome"] == "superseded"
    assert after["code"] == "reinforce_inactive_card"
    assert [c["title"] for c in store.history(superseded["card"])] == ["Pin pydantic < 3", "Use pydantic 3.2"]


def test_replaying_a_proposal_is_idempotent(store, session):
    batch, ids = session()
    ev = ids["we pin pydantic < 3 because v3 breaks our validators"]
    client = FakeClient([SEARCH], [decision(ev), decision(ev)])
    observer.observe(store, batch, client)
    assert [r["outcome"] for r in client.tool_results("propose_decision")] == ["accepted", "duplicate"]
    assert len(store.current()) == 1


def test_retry_after_a_storage_failure_converges_without_duplicates(store, session, monkeypatch):
    batch, ids = session()
    ev = ids["we pin pydantic < 3 because v3 breaks our validators"]
    real = store.add_card

    def flaky(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(store, "add_card", flaky)
    first = observer.observe(store, batch, FakeClient([SEARCH], [decision(ev)]))
    assert not first.settled and first.retry_reasons == ["attempt_retryable", "card_retryable"]
    assert store.cursor("s1") == 0  # held

    monkeypatch.setattr(store, "add_card", real)
    retry_batch, _ = session([assistant("anything else?", "a2")])  # the session went on: a wider batch, same start
    assert retry_batch.start_line == 0
    client = FakeClient([SEARCH], [decision(ev)])
    assert observer.observe(store, retry_batch, client).settled
    assert len(store.current()) == 1 and store.cursor("s1") == 3


# ---------------------------------------------------------------- ownership-based outcome (ported scenarios)

def test_rejected_whisper_then_normal_end_settles(store, session):
    batch, _ = session()
    client = FakeClient([SEARCH], [("deliver_whisper", {"card": "c_nope", "snippet_ids": ["s1"]})])
    result = observer.observe(store, batch, client)
    assert result.settled and store.cursor("s1") == 2
    assert [e.name for e in result.tool_errors] == ["deliver_whisper"]


def test_rejected_whisper_with_a_retryable_outcome_still_holds(store, session, monkeypatch):
    batch, ids = session()
    monkeypatch.setattr(store, "add_card", lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("disk I/O error")))
    client = FakeClient([SEARCH], [("deliver_whisper", {"card": "c_nope", "snippet_ids": ["s1"]}),
                                   decision(ids["we pin pydantic < 3 because v3 breaks our validators"])])
    result = observer.observe(store, batch, client)
    assert not result.settled and "card_retryable" in result.retry_reasons  # the ledger wins


def test_owned_tool_that_throws_holds_the_batch(store, session, monkeypatch):
    batch, ids = session()
    monkeypatch.setattr(Runtime, "propose", lambda self, bucket, raw: 1 / 0)
    result = observer.observe(store, batch, FakeClient([SEARCH], [decision("x")]))
    assert not result.settled and result.retry_reasons == ["attempt_retryable"]
    assert result.tool_errors[0].kind == "execution"


def test_link_provenance_rejection_alone_settles(store, session):
    batch, _ = session()
    client = FakeClient([SEARCH], [("suggest_link", {"cards": ["c_a", "c_b"], "relation": "relates_to", "reason": "same lib"})])
    result = observer.observe(store, batch, client)
    assert result.settled and "maintenance" in result.tool_errors[0].message


@pytest.mark.parametrize("client, reason", [
    (FakeClient(fail=1), "provider"),
    (FakeClient("nothing to save"), "required"),  # ended before the required search
    (FakeClient([("rm_rf", {})]), "unknown tool"),
    (FakeClient(*[[SEARCH]] * 20), "rounds"),
])
def test_transport_failures_fail_the_turn(store, session, client, reason, caplog):
    batch, _ = session()
    result = observer.observe(store, batch, client)
    assert not result.settled and result.retry_reasons == ["turn_failed"]
    assert reason in caplog.text and store.cursor("s1") == 0


def test_whisper_not_surfaced_error_lists_what_was_surfaced(store, session):
    batch, ids = session()
    observer.observe(store, batch, FakeClient([SEARCH], [decision(ids["we pin pydantic < 3 because v3 breaks our validators"])]))
    (card,) = store.current()
    batch2, _ = session([user("should we try pydantic 3?", "u2")], name="s2")
    client = FakeClient([("search", {"query": "pydantic", "purpose": "whisper"})],
                        [("deliver_whisper", {"card": card["card"], "snippet_ids": ["snip_wrong"]})])
    result = observer.observe(store, batch2, client)
    message = result.tool_errors[0].message
    assert card["card"] in message and "snippet_ids: snip_" in message


def test_second_whisper_is_already_resolved_not_an_error(store, session):
    batch, ids = session()
    observer.observe(store, batch, FakeClient([SEARCH], [decision(ids["we pin pydantic < 3 because v3 breaks our validators"])]))
    batch2, _ = session([user("should we try pydantic 3?", "u2")], name="s2")

    def pick(messages):
        (found,) = json.loads(messages[-1]["content"])["results"]
        args = {"card": found["card"], "snippet_ids": [found["snippets"][0]["snippet_id"]]}
        return [("deliver_whisper", args), ("deliver_whisper", args)]
    client = FakeClient([("search", {"query": "pydantic", "purpose": "whisper"})], pick)
    result = observer.observe(store, batch2, client)
    assert [r["status"] for r in client.tool_results("deliver_whisper")] == ["ok", "already_resolved"]
    assert result.tool_errors == [] and result.settled
    assert "Pin pydantic < 3" in result.whisper and "“we pin pydantic < 3" in result.whisper
    assert [w["card"] for w in store.pending_whispers("s2")] == [store.current()[0]["card"]]


@pytest.mark.parametrize("tool, args, settled", [
    ("propose_decision", "{not json", False),  # an unparseable owned-work call may hide a write
    ("deliver_whisper", "{not json", True),
])
def test_invalid_json_holds_only_owned_work(store, session, tool, args, settled):
    batch, _ = session()
    result = observer.observe(store, batch, FakeClient([SEARCH], [(tool, args)]))
    assert result.settled is settled
    assert result.tool_errors[0].kind == "invalid_arguments"


def test_batch_with_no_events_settles_without_a_model_call(store, session):
    batch, _ = session([{"type": "summary", "summary": "x"}])
    client = FakeClient()
    assert observer.observe(store, batch, client).settled and client.requests == []


def test_an_assistant_only_batch_does_not_require_search(store, session):
    batch, _ = session([assistant("Running the tests now.", "a1")])
    assert observer.observe(store, batch, FakeClient("nothing durable")).settled


# ---------------------------------------------------------------- intents

INTENT = "remember that staging deploys need the VPN on"


def test_intent_answered_by_a_card_settles(store, session):
    batch, ids = session([user(INTENT, "u1")])
    intent = stable_id("intent", {"evidence_id": ids[INTENT]})  # intent ids are stable, like evidence ids
    client = FakeClient([("search", {"query": "staging VPN", "purpose": "maintenance"})],
                        [("propose_gotcha", {"title": "Staging deploys need the VPN", "intent_id": intent,
                                             "fields": {"symptom": "staging deploy fails", "fix": "turn the VPN on"},
                                             "evidence_ids": [ids[INTENT]]})])
    assert observer.observe(store, batch, client).settled
    assert store.intent_resolved(intent)


def test_ignored_intent_holds_for_three_turns_then_releases(store, session):
    batch, _ = session([user(INTENT, "u1")])
    quiet = lambda: FakeClient([("search", {"query": "staging", "purpose": "maintenance"})])  # noqa: E731
    outcomes = []
    for _ in range(3):
        result = observer.observe(store, batch, quiet())
        outcomes.append((result.settled, result.retry_reasons))
    assert outcomes == [(False, ["intent_blocking"]), (False, ["intent_blocking"]), (True, [])]


def test_failed_turns_dont_spend_the_intent_budget(store, session):
    batch, _ = session([user(INTENT, "u1")])
    for _ in range(4):
        observer.observe(store, batch, FakeClient(fail=1))
    assert store.db.execute("select count(*) from intent_outcomes where outcome = 'deferred'").fetchone()[0] == 0


def test_declined_intent_settles_with_a_reason(store, session):
    batch, _ = session([user(INTENT, "u1")])

    def decline(messages):
        found = messages[1]["content"].split('intent_id="')[1].split('"')[0]
        return [("decline_intent", {"intent_id": found, "reason": "one-off, not team knowledge"})]
    client = FakeClient([("search", {"query": "staging", "purpose": "maintenance"})], decline)
    assert observer.observe(store, batch, client).settled
    assert client.tool_results("decline_intent") == [{"outcome": "declined"}]


# ---------------------------------------------------------------- links → review inbox

def test_link_suggestion_after_maintenance_search_goes_to_review(store, session):
    batch, ids = session([user("we pin pydantic < 3 because v3 breaks our validators", "u1"),
                          user("gotcha: pydantic v3 validators fail silently; fix by pinning < 3", "u2")])
    client = FakeClient([SEARCH], [decision(ids["we pin pydantic < 3 because v3 breaks our validators"]),
                                   ("propose_gotcha", {"title": "pydantic v3 validators fail silently",
                                                       "fields": {"symptom": "validators fail silently", "fix": "pin < 3"},
                                                       "evidence_ids": [ids["gotcha: pydantic v3 validators fail silently; fix by pinning < 3"]]})])
    observer.observe(store, batch, client)
    cards = [c["card"] for c in store.current()]
    batch2, _ = session([user("the pydantic pin and the validator gotcha are the same story", "u3")])
    link = {"cards": cards, "relation": "relates_to", "reason": "the pin is the gotcha's fix"}
    client = FakeClient([SEARCH], [("suggest_link", link), ("suggest_link", link)])
    assert observer.observe(store, batch2, client).settled
    assert [r["outcome"] for r in client.tool_results("suggest_link")] == ["queued", "duplicate"]
    (review,) = store.reviews()
    assert review["relation"] == "relates_to" and sorted(review["cards"]) == sorted(cards)


# ---------------------------------------------------------------- found in the live smoke test

def test_cards_are_dated_by_their_earliest_evidence(store, session):
    batch, ids = session()
    observer.observe(store, batch, FakeClient([SEARCH], [decision(ids["we pin pydantic < 3 because v3 breaks our validators"])]))
    assert store.current()[0]["valid_from"] == "2026-10-19T01:00:00Z"  # when it was said, not when tacit learned it


def test_cards_written_this_turn_can_be_linked(store, session):
    # Live run: the model linked the two cards it had just created and was rejected twice
    batch, ids = session([user("we pin pydantic < 3 because v3 breaks our validators", "u1"),
                          user("gotcha: pydantic v3 validators fail silently; fix by pinning < 3", "u2")])

    def link(messages):
        written = [json.loads(m["content"])["card"] for m in messages if m["role"] == "tool" and '"card"' in m["content"]]
        return [("suggest_link", {"cards": written, "relation": "relates_to", "reason": "the pin is the fix"})]
    client = FakeClient([SEARCH], [decision(ids["we pin pydantic < 3 because v3 breaks our validators"]),
                                   ("propose_gotcha", {"title": "pydantic v3 validators fail silently",
                                                       "fields": {"symptom": "validators fail silently", "fix": "pin < 3"},
                                                       "evidence_ids": [ids["gotcha: pydantic v3 validators fail silently; fix by pinning < 3"]]})],
                        link)
    result = observer.observe(store, batch, client)
    assert result.tool_errors == [] and client.tool_results("suggest_link")[0]["outcome"] == "queued"


def test_writing_before_searching_is_feedback_not_a_failure(store, session):
    # Live run: the model proposed three cards without searching, and the required-search rule failed the whole turn
    batch, ids = session()
    ev = ids["we pin pydantic < 3 because v3 breaks our validators"]
    client = FakeClient([decision(ev)], [SEARCH], [decision(ev)])
    result = observer.observe(store, batch, client)
    first, second = client.tool_results("propose_decision")
    assert "search with purpose=maintenance first" in first["error"] and second["outcome"] == "accepted"
    assert result.settled and result.retry_reasons == [] and len(store.current()) == 1


def test_a_session_is_never_whispered_what_it_just_said(store, session):
    # Live run: the observer whispered the gotcha it had just written from the same session
    batch, ids = session()
    observer.observe(store, batch, FakeClient([SEARCH], [decision(ids["we pin pydantic < 3 because v3 breaks our validators"])]))
    whisper_search = ("search", {"query": "pydantic", "purpose": "whisper"})
    same, _ = session([user("should we try pydantic 3?", "u2")])
    client = FakeClient([whisper_search])
    observer.observe(store, same, client)
    assert client.tool_results("search")[0]["results"] == []
    other, _ = session([user("should we try pydantic 3?", "u9")], name="s2")
    client = FakeClient([whisper_search])
    observer.observe(store, other, client)
    assert len(client.tool_results("search")[0]["results"]) == 1


def test_one_statement_one_card_is_in_the_prompt():
    assert "One piece of knowledge is one card" in observer.SYSTEM


def test_a_corrected_owned_call_resolves_its_invalid_json(store, session):
    # Review finding: a fixed formatting mistake must not hold work that settled
    batch, ids = session()
    ev = ids["we pin pydantic < 3 because v3 breaks our validators"]
    result = observer.observe(store, batch, FakeClient([SEARCH], [("propose_decision", "{bad")], [decision(ev)]))
    assert len(store.current()) == 1
    assert result.settled and result.retry_reasons == []
    assert result.tool_errors[0].resolved


def test_an_invalid_owned_call_still_holds_when_another_tool_succeeds(store, session):
    batch, ids = session()
    ev = ids["we pin pydantic < 3 because v3 breaks our validators"]
    result = observer.observe(store, batch, FakeClient([SEARCH], [("propose_gotcha", "{bad")], [decision(ev)]))
    assert not result.settled and result.retry_reasons == ["attempt_retryable"]
