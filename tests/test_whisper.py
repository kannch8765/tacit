"""Whispers end to end: snippets, weaving (history + related cards), the sync lane, the prompt hook and the worker."""
import datetime
import io
import json

import pytest
from fake_llm import FakeClient

from tacit import cli, whisper, worker
from tacit.store import Evidence, Store


class Clock:
    def __init__(self):
        self.t = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)

    def __call__(self):
        self.t += datetime.timedelta(hours=1)
        return self.t.isoformat(timespec="seconds")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "tacit.db", clock=Clock())
    yield s
    s.close()


@pytest.fixture
def batch(store, tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text("\n".join(json.dumps({"type": "user", "message": {"content": f"l{i}"}}) for i in range(50)) + "\n")
    store.enqueue("seed", path)
    return store.next_batch("seed")


def card(store, batch, title, bucket="decision", quote=None, line=0, **kw):
    fields = {"decision": {"rationale": "because"}, "gotcha": {"symptom": "s", "fix": "f"}}[bucket]
    return store.add_card(batch, bucket, title, fields, [Evidence(line, "user", quote or title, f"ev{line}", "user_text")], **kw)


# ---------------------------------------------------------------- snippets

def test_split_keeps_exact_text_and_caps_length():
    text = "First point. Second point!\n\n" + "x" * 1300
    pieces = whisper.split_source_text(text)
    assert pieces[:2] == ["First point.", "Second point!"]
    assert [len(p) for p in pieces[2:]] == [600, 600, 100] and "".join(pieces[2:]) == "x" * 1300


def test_bound_keeps_oldest_and_newest():
    items = [{"snippet_id": str(i)} for i in range(20)]
    assert [s["snippet_id"] for s in whisper.bound(items, 4)] == ["0", "1", "18", "19"]


# The fast lane asks for one snippet: `unique[-0:]` would have returned the whole list
@pytest.mark.parametrize("limit, expected", [(1, ["0"]), (2, ["0", "19"]), (3, ["0", "1", "19"]), (0, [])])
def test_bound_returns_at_most_limit(limit, expected):
    items = [{"snippet_id": str(i)} for i in range(20)]
    assert [s["snippet_id"] for s in whisper.bound(items, limit)] == expected


def test_raw_tool_calls_are_never_quoted(store, batch):
    c = store.add_card(batch, "decision", "Use uv", {"rationale": "fast"},
                       [Evidence(0, "assistant", '{"command": "uv sync"}', "e0", "assistant_tool_use"),
                        Evidence(1, "user", "use uv from now on", "e1", "user_text")])
    assert [s["quote"] for s in whisper.snippets(store, c)] == ["use uv from now on"]


# ---------------------------------------------------------------- weaving

def test_render_weaves_history_with_dates_and_quotes(store, batch):
    old = card(store, batch, "Pin pydantic < 3", quote="we pin pydantic < 3. v3 breaks validators", line=1)
    new = card(store, batch, "Use pydantic 3.2", quote="v3.2 fixed it, we moved", line=2, supersedes=old)
    text, related = whisper.render(store, store.get(new), whisper.snippets(store, new, 1))
    assert text.splitlines()[0].startswith("tacit · decision (as of 2026-10-01, v2, ")
    assert "  was (until 2026-10-01): Pin pydantic < 3" in text
    assert "    [user] “we pin pydantic < 3.”" in text  # one exact sentence, never a cut-off quote
    assert related == []


def test_related_cards_are_mmr_selected_and_marked(store, batch):
    main = card(store, batch, "Deploy staging through the VPN")
    on_topic = card(store, batch, "Staging VPN profile lives in 1Password", bucket="gotcha", line=1)
    redundant = card(store, batch, "Deploy staging through the VPN gateway", line=2)
    off_topic = card(store, batch, "Lint with ruff", line=3)
    clash = card(store, batch, "Staging deploys go straight to the cloud console", line=4)
    for other, kind in [(on_topic, "relates_to"), (redundant, "relates_to"), (off_topic, "relates_to"), (clash, "contradicts")]:
        store.link(main, other, kind)
    text, related = whisper.render(store, store.get(main), [], topic="how do I deploy staging over VPN")
    # The contradiction is a warning, so it always comes first. MMR then picks the two most relevant related cards;
    # the off-topic one is left out.
    assert related == [clash, redundant, on_topic]
    assert text.splitlines()[3] == f"  - ⚠ contradicts: Staging deploys go straight to the cloud console (decision, {clash})"


def test_related_skips_cards_the_session_already_knows(store, batch):
    main, other = card(store, batch, "Use uv"), card(store, batch, "uv lockfile is committed", line=1)
    store.link(main, other, "relates_to")
    assert whisper.render(store, store.get(main), [], known={other})[1] == []
    assert whisper.render(store, store.get(main), [])[1] == [other]


def test_cjk_tokens_are_bigrams():
    assert {"ステ", "テー", "ージ", "vpn"} <= whisper.tokens("ステージ VPN")


# ---------------------------------------------------------------- sync lane

def test_prefetch_needs_two_meaningful_terms(store, batch):
    c = card(store, batch, "Staging deploys need the VPN on")
    assert whisper.prefetch(store, "s1", "how do I do this?") == []
    assert whisper.prefetch(store, "s1", "the staging thing") == []  # one shared term in a multi-term prompt
    ((found, text),) = whisper.prefetch(store, "s1", "deploy to staging needs what again? the VPN?")
    assert found["card"] == c and "Staging deploys need the VPN on" in text


def test_prompt_hook_delivers_pending_once_and_never_repeats_a_card(store, batch, tmp_path, monkeypatch):
    monkeypatch.setenv("TACIT_HOME", str(tmp_path))
    c = card(store, batch, "Staging deploys need the VPN on")
    store.queue_whisper("w1", "s1", batch.batch, c, "tacit · decision: queued whisper")

    def prompt(text):
        out = io.StringIO()
        cli.hook_prompt(store, io.StringIO(json.dumps({"session_id": "s1", "prompt": text})), out)
        return json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"] if out.getvalue() else None

    first = prompt("anything")
    assert first.startswith(cli.PREAMBLE) and "queued whisper" in first
    assert prompt("staging VPN deploy again?") is None  # already whispered to this session: no repeat
    assert prompt("anything") is None


def test_prompt_hook_sync_lane_and_bad_input(store, batch, capsys):
    card(store, batch, "Staging deploys need the VPN on")
    out = io.StringIO()
    cli.hook_prompt(store, io.StringIO(json.dumps({"session_id": "s2", "prompt": "staging deploy fails, VPN?"})), out)
    assert "Staging deploys need the VPN on" in out.getvalue()
    cli.hook_prompt(store, io.StringIO("{oops"), io.StringIO())
    assert "prompt hook ignored" in capsys.readouterr().err


# ---------------------------------------------------------------- worker

def write_session(store, tmp_path, session, text):
    path = tmp_path / f"{session}.jsonl"
    path.write_text(json.dumps({"type": "user", "uuid": f"{session}-u", "message": {"content": text}}) + "\n")
    store.enqueue(session, path)


def test_drain_settles_what_it_can_and_skips_held_sessions(store, tmp_path):
    write_session(store, tmp_path, "good", "the build uses uv")
    write_session(store, tmp_path, "bad", "remember that the build uses uv")  # an intent the model ignores: held

    class Router:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools, session=None):
            self.calls += 1
            return FakeClient([("search", {"query": "uv", "purpose": "maintenance"})]).chat(messages, tools) \
                if messages[-1]["role"] == "user" else {"role": "assistant", "content": "done"}
    stats = worker.drain(store, Router())
    assert stats == {"batches": 2, "settled": 1, "held": 1}
    assert store.cursor("good") == 1 and store.cursor("bad") == 0


def test_worker_is_single_and_needs_a_model(store, tmp_path, monkeypatch):
    for key in ("TACIT_LLM_MODEL", "TACIT_LLM_API_KEY", "TACIT_LLM_API_KEY_FILE"):
        monkeypatch.delenv(key, raising=False)
    write_session(store, tmp_path, "s1", "the build uses uv")
    assert worker.run(store, str(tmp_path)) is None  # no model configured: nothing observed, still queued
    assert store.cursor("s1") == 0
    with worker.Lock(str(tmp_path / "worker.lock")) as held:
        assert held
        assert worker.run(store, str(tmp_path), client=FakeClient()) is None  # another worker holds the lock
    assert worker.run(store, str(tmp_path), client=FakeClient([("search", {"query": "uv", "purpose": "maintenance"})])) \
        == {"batches": 1, "settled": 1, "held": 0}


def test_stop_hook_spawns_the_worker_unless_disabled(store, tmp_path, monkeypatch):
    spawned = []
    monkeypatch.setattr(worker, "spawn", spawned.append)
    monkeypatch.setenv("TACIT_HOME", str(tmp_path))
    hook = json.dumps({"session_id": "s1", "transcript_path": str(tmp_path / "s1.jsonl")})
    cli.hook_stop(store, io.StringIO(hook))
    assert spawned == []
    monkeypatch.delenv("TACIT_NO_WORKER")
    cli.hook_stop(store, io.StringIO(hook))
    assert spawned == [str(tmp_path)]


def test_version_numbers_never_split_a_quote():
    # Live run, gemini flash-lite: "...when 3.2 ships the fix" was quoted as “2 ships the fix ...”
    text = "Let's pin pydantic < 3 for now, we can revisit when 3.2 ships the fix. See e.g.the docs!  直した。次へ"
    assert whisper.split_source_text(text) == [
        "Let's pin pydantic < 3 for now, we can revisit when 3.2 ships the fix.", "See e.g.the docs!", "直した。", "次へ"]
    assert whisper.first_sentence("v3.2 is out. Upgrade later.") == "v3.2 is out."


def test_fast_lane_skips_cards_the_session_wrote(store, tmp_path):
    path = tmp_path / "s1.jsonl"
    path.write_text(json.dumps({"type": "user", "message": {"content": "staging deploys need the VPN on"}}) + "\n")
    store.enqueue("s1", path)
    own = store.next_batch("s1")
    store.add_card(own, "gotcha", "Staging deploys need the VPN on", {"symptom": "s", "fix": "f"}, [Evidence(0, "user", "x")])
    assert whisper.prefetch(store, "s1", "staging deploy needs the VPN?") == []
    assert len(whisper.prefetch(store, "s2", "staging deploy needs the VPN?")) == 1
