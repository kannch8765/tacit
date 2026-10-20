"""Observability and the human inbox: daily rollup, healthcheck, the HTML report (aggregates only), and review."""
import datetime
import io
import json

import pytest
from fake_llm import FakeClient

from tacit import cli, observer, report, stats, transcript, whisper
from tacit.buckets import Rejected
from tacit.store import Evidence, Store

UTC = datetime.timezone.utc
SECRET_TITLE = "Pin pydantic below three"


class Clock:
    def __init__(self, start=datetime.datetime(2026, 10, 18, 9, tzinfo=UTC)):
        self.t = start

    def __call__(self):
        self.t += datetime.timedelta(seconds=30)
        return self.t.isoformat(timespec="seconds")


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    s = Store(tmp_path / "tacit.db", clock=clock)
    yield s
    s.close()


def session(store, tmp_path, name, *texts):
    path = tmp_path / f"{name}.jsonl"
    with open(path, "a") as f:
        for i, text in enumerate(texts):
            f.write(json.dumps({"type": "user", "uuid": f"{name}-{path.stat().st_size}-{i}", "message": {"content": text}}) + "\n")
    store.enqueue(name, path)
    batch = store.next_batch(name)
    return batch, {e.quote: e.evidence_id for e in transcript.events(batch)}


SEARCH = ("search", {"query": "pydantic", "purpose": "maintenance"})


def run_day(store, tmp_path):
    """One settled batch (1 accepted, 1 rejected), one held batch (an ignored intent)."""
    batch, ids = session(store, tmp_path, "s1", "we pin pydantic < 3 because v3 breaks validators")
    ev = next(iter(ids.values()))
    observer.observe(store, batch, FakeClient([SEARCH], [
        ("propose_decision", {"title": SECRET_TITLE, "fields": {"rationale": "v3 breaks validators"}, "evidence_ids": [ev]}),
        ("propose_decision", {"title": "made up", "fields": {"rationale": "x"}, "evidence_ids": ["ev_nope"]})]))
    held, _ = session(store, tmp_path, "s2", "remember that staging needs the VPN")
    observer.observe(store, held, FakeClient([SEARCH]))


# ---------------------------------------------------------------- rollup

def test_rollup_counts_the_day(store, tmp_path):
    run_day(store, tmp_path)
    (day, metrics), = stats.compute(store, UTC, ["2026-10-18"]).items()
    assert day == "2026-10-18"
    assert {k: metrics[k] for k in ("attempts", "completed", "retryable", "retry_reason:intent_blocking", "proposals",
                                    "proposal:accepted", "proposal:rejected", "rejection:unresolvable_evidence",
                                    "intents_captured", "cards_new")} == {
        "attempts": 2, "completed": 1, "retryable": 1, "retry_reason:intent_blocking": 1, "proposals": 2,
        "proposal:accepted": 1, "proposal:rejected": 1, "rejection:unresolvable_evidence": 1,
        "intents_captured": 1, "cards_new": 1}
    assert metrics["retryable_share"] == 0.5 and metrics["rejected_share"] == 0.5
    assert metrics["observer_turn_s_p90"] > 0


def test_days_follow_the_local_timezone(store, tmp_path, clock):
    clock.t = datetime.datetime(2026, 10, 18, 23, 50, tzinfo=UTC)  # 08:50 on the 19th in Tokyo
    run_day(store, tmp_path)
    tokyo = datetime.timezone(datetime.timedelta(hours=9))
    assert set(stats.compute(store, tokyo, ["2026-10-18", "2026-10-19"])) == {"2026-10-19"}


def test_stats_json_is_numbers_only(store, tmp_path):
    run_day(store, tmp_path)
    out = stats.summary_json(store, UTC, ["2026-10-18"])
    assert SECRET_TITLE not in out and "pydantic" not in out
    data = json.loads(out)
    assert data["totals"]["cards_active"] == 1 and data["totals"]["backlog_lines"] == 1  # s2 is held


# ---------------------------------------------------------------- health

def test_health_levels(store, tmp_path, clock):
    run_day(store, tmp_path)
    now = clock.t + datetime.timedelta(hours=2)
    result = stats.check(store, now)
    levels = {c["name"]: (c["level"], c["value"]) for c in result["checks"]}
    assert levels["batch_retryable_share_24h"] == ("alert", 0.5)
    assert levels["proposal_rejected_share_24h"] == ("warn", 0.5)
    assert levels["held_sessions"] == ("warn", 1)
    assert levels["unresolved_intent_age_h"][0] == "warn"
    assert levels["pending_reviews"] == ("ok", 0)
    assert result["overall"] == "alert" and result["info"]["unresolved_intents"] == 1


def test_health_is_ok_on_an_empty_ledger(store):
    result = stats.check(store)
    assert result["overall"] == "ok" and all(c["value"] in (None, 0) for c in result["checks"])


def test_health_cli_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TACIT_HOME", str(tmp_path / "home"))
    assert cli.main(["health"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "tacit_health_v1"


# ---------------------------------------------------------------- report

def test_report_has_charts_and_no_card_text(store, tmp_path):
    run_day(store, tmp_path)
    page, data = report.build(store, UTC, "2026-10-18", generated_at="x")
    assert page.count("<svg") >= 8 and "Proposals by outcome" in page
    assert SECRET_TITLE not in page and "pydantic" not in page and "VPN" not in page
    assert data["series"]["all"]["attempts"][-1] == 2
    assert data["summary"]["last_7d"]["rejected_share"] == 0.5


def test_report_on_an_empty_ledger(store, tmp_path):
    page, _ = report.build(store, UTC, "2026-10-18")
    assert "No data in this window yet" in page
    path = report.write(store, UTC, "2026-10-18", tmp_path / "out")
    assert path.exists() and json.loads((tmp_path / "out" / "metrics.json").read_text())["schema"] == "tacit_report_v1"


# ---------------------------------------------------------------- review

@pytest.fixture
def cards(store, tmp_path):
    batch, _ = session(store, tmp_path, "seed", *[f"line {i}" for i in range(5)])

    def make(title, line):
        return store.add_card(batch, "decision", title, {"rationale": "r"}, [Evidence(line, "user", title, f"e{line}", "user_text")])
    return [make("Use uv", 0), make("uv lockfile is committed", 1), make("Use uv for installs", 2)]


@pytest.mark.parametrize("relation", ["relates_to", "contradicts"])
def test_accepting_a_link_links_every_pair(store, cards, relation):
    store.add_review("r1", relation, cards, "same topic", type("B", (), {"batch": "b"})())
    store.decide_review("r1", accept=True)
    links = {(l["src"], l["dst"], l["kind"]) for c in cards for l in store.links_of(c)}
    assert len(links) == 3 and {k for _, _, k in links} == {relation}
    assert store.reviews() == [] and store.reviews("accepted")[0]["decided_at"]


def test_accepting_supersedes_closes_the_older_cards(store, cards):
    store.add_review("r1", "supersedes", cards[:2], "the lockfile rule replaces it", type("B", (), {"batch": "b"})())
    store.decide_review("r1", accept=True)
    old, new = store.get(cards[0]), store.get(cards[1])
    assert (old["status"], old["superseded_by"], new["status"]) == ("superseded", cards[1], "active")


def test_accepting_same_as_merges_into_the_oldest(store, cards):
    store.add_review("r1", "same_as", [cards[2], cards[0]], "same rule", type("B", (), {"batch": "b"})())
    store.decide_review("r1", accept=True)
    keep, dup = store.get(cards[0]), store.get(cards[2])
    assert dup["status"] == "retired" and keep["reinforcements"] == 1
    assert [e["quote"] for e in store.evidence(cards[0])] == ["Use uv", "Use uv for installs"]


# Review finding: review is a second writer, held to the same bucket and history rules
@pytest.mark.parametrize("relation", ["same_as", "supersedes"])
def test_review_cannot_merge_or_replace_across_buckets(store, cards, tmp_path, relation):
    batch, _ = session(store, tmp_path, "other", "the PM is now Mika")
    world = store.add_card(batch, "world_update", "PM is Mika", {"source": "user"}, [Evidence(0, "user", "x", "w0", "user_text")])
    store.add_review("r1", relation, [world, cards[0]], "same claim", type("B", (), {"batch": "b"})())
    with pytest.raises(Rejected, match="review_other_bucket"):
        store.decide_review("r1", accept=True)
    assert {e["role"] for e in store.evidence(world)} == {"user"} and store.get(cards[0])["status"] == "active"
    assert store.reviews()[0]["review"] == "r1"  # still pending: a human can dismiss it


def test_a_supersedes_review_shows_in_history_both_ways(store, cards):
    store.add_review("r1", "supersedes", cards[:2], "the lockfile rule replaces it", type("B", (), {"batch": "b"})())
    store.decide_review("r1", accept=True)
    assert [c["card"] for c in store.history(cards[1])] == cards[:2]
    assert [c["card"] for c in store.history(cards[0])] == cards[:2]
    lines = whisper.history_lines(store, store.get(cards[1]))
    assert lines[0].startswith("  was (until ") and lines[0].endswith("Use uv")


# Review finding: history is the whole replacement family, the same from every entry point
def test_history_is_the_whole_family_from_every_card(store, cards, tmp_path):
    a, b, c = cards
    store.add_review("r1", "supersedes", [a, b, c], "the installs rule replaces both", type("B", (), {"batch": "b"})())
    store.decide_review("r1", accept=True)
    batch, _ = session(store, tmp_path, "later", "uv for installs and CI")
    d = store.add_card(batch, "decision", "Use uv for installs and CI", {"rationale": "r"},
                       [Evidence(0, "user", "uv for installs and CI", "e9", "user_text")], supersedes=c)
    family = [x["card"] for x in store.history(a)]
    assert sorted(family) == sorted([a, b, c, d]) and len(family) == 4
    for card in (b, c, d):
        assert [x["card"] for x in store.history(card)] == family
    edges = {(l["src"], l["dst"]) for x in family for l in store.links_of(x) if l["kind"] == "supersedes"}
    assert edges == {(c, a), (c, b), (d, c)}
    assert all(family.index(old) < family.index(new) for new, old in edges)
    assert set(store.predecessors(d)) == {a, b, c} and store.predecessors(d)[0] == c
    assert store.predecessors(a) == []


# Review finding: history follows the replacement edge, not timestamp order (the clock has one-second resolution)
def test_history_follows_supersedes_links_when_timestamps_tie(tmp_path, monkeypatch):
    import tacit.store as storage
    names = {"c": iter(["c_z", "c_a"]), "l": iter(["l_a", "l_z"])}
    real = storage.new_id
    monkeypatch.setattr(storage, "new_id", lambda p: next(names[p]) if p in names else real(p))
    store = Store(tmp_path / "tacit.db", clock=lambda: "2026-10-19T00:00:00+00:00")
    batch, _ = session(store, tmp_path, "s", "the old and the new rule")
    ev = [Evidence(0, "user", "the old and the new rule", "e0", "user_text")]
    new = store.add_card(batch, "decision", "New rule", {"rationale": "r"}, ev)
    old = store.add_card(batch, "decision", "Old rule", {"rationale": "r"}, ev)
    store.add_review("r", "supersedes", [new, old], "replacement", batch)
    store.decide_review("r", True)
    assert store.get(old)["superseded_by"] == new
    assert [c["card"] for c in store.history(new)] == [old, new] == [c["card"] for c in store.history(old)]
    assert any(line.endswith("Old rule") for line in whisper.history_lines(store, store.get(new)))
    store.close()


def test_dismiss_and_guards(store, cards):
    batch = type("B", (), {"batch": "b"})()
    store.add_review("r1", "relates_to", cards[:2], "maybe", batch)
    store.decide_review("r1", accept=False)
    assert store.links_of(cards[0]) == [] and store.reviews("dismissed")
    with pytest.raises(Rejected, match="review_already_decided"):
        store.decide_review("r1", accept=True)
    with pytest.raises(Rejected, match="unknown_review"):
        store.decide_review("nope", accept=True)
    store.add_review("r2", "same_as", cards[1:], "dup", batch)
    store.add_review("r3", "relates_to", cards[1:], "x", batch)
    store.decide_review("r2", accept=True)
    with pytest.raises(Rejected, match="review_cards_not_current"):  # cards[2] retired meanwhile
        store.decide_review("r3", accept=True)


def test_review_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TACIT_HOME", str(tmp_path / "home"))
    s = Store(tmp_path / "home" / "tacit.db")
    batch, _ = session(s, tmp_path, "seed", "a", "b")
    a = s.add_card(batch, "decision", "Use uv", {"rationale": "r"}, [Evidence(0, "user", "a")])
    b = s.add_card(batch, "gotcha", "uv cache fills the disk", {"symptom": "disk full", "fix": "uv cache prune"}, [Evidence(1, "user", "b")])
    s.add_review("r1", "relates_to", [a, b], "both about uv", batch)
    s.close()
    cli.main(["review"])
    listed = capsys.readouterr().out
    assert "r1  relates_to" in listed and "uv cache fills the disk" in listed
    cli.main(["review", "accept", "r1"])
    assert "r1 accepted" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["review", "accept", "r1"])
    assert "review_already_decided" in capsys.readouterr().err
