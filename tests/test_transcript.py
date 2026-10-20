"""Transcript events and redaction: one event per block, stable ids, secrets scrubbed before anything leaves."""
import json

import pytest

from tacit import transcript
from tacit.redact import redact
from tacit.store import Batch


def batch(lines, start=0, session="s1"):
    return Batch("b1", session, start, start + len(lines), [json.dumps(x) if not isinstance(x, str) else x for x in lines])


USER = {"type": "user", "uuid": "u1", "timestamp": "2026-10-19T01:00:00Z",
        "message": {"role": "user", "content": "we pin pydantic < 3 because v3 breaks our validators"}}
ASSISTANT = {"type": "assistant", "uuid": "a1", "timestamp": "2026-10-19T01:00:05Z", "message": {"role": "assistant", "content": [
    {"type": "text", "text": "Got it, pinning now."},
    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "uv add 'pydantic<3'"}},
]}}
RESULT = {"type": "user", "uuid": "u2", "timestamp": "2026-10-19T01:00:09Z", "message": {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t1", "content": "Resolved 12 packages"},
]}}


def test_one_event_per_block_with_roles_and_lines():
    evs = transcript.events(batch([{"type": "mode"}, USER, ASSISTANT, RESULT], start=40))
    assert [(e.line, e.role, e.kind) for e in evs] == [
        (41, "user", "user_text"),
        (42, "assistant", "assistant_text"),
        (42, "assistant", "assistant_tool_use"),
        (43, "tool", "tool_result"),
    ]
    assert evs[3].tool_name == "Bash"  # tool name carried from the matching tool_use
    assert json.loads(evs[2].quote) == {"command": "uv add 'pydantic<3'"}


def test_ids_are_stable_across_retries_and_distinct_per_session():
    first = [e.evidence_id for e in transcript.events(batch([USER, ASSISTANT]))]
    wider = [e.evidence_id for e in transcript.events(batch([USER, ASSISTANT, RESULT]))]
    other = [e.evidence_id for e in transcript.events(batch([USER, ASSISTANT], session="s2"))]
    assert wider[:len(first)] == first
    assert len(set(first)) == len(first)
    assert not set(first) & set(other)


def test_skips_noise_and_bad_lines():
    lines = ["not json", {"type": "summary", "summary": "x"}, {"type": "user", "message": {"content": "   "}},
             {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "hmm"}]}}]
    assert transcript.events(batch(lines)) == []


def test_long_blocks_are_capped():
    long = {"type": "user", "uuid": "u9", "message": {"content": "x" * 20_000}}
    (e,) = transcript.events(batch([long]))
    assert len(e.quote) < 12_100 and e.quote.endswith("chars]")


def test_as_evidence_keeps_provenance():
    (e,) = transcript.events(batch([USER]))
    ev = e.as_evidence()
    assert (ev.line, ev.role, ev.evidence_id, ev.kind) == (0, "user", e.evidence_id, "user_text")


def test_catalog_escapes_and_lists_every_event():
    tricky = {"type": "user", "uuid": "u3", "message": {"content": '<b>"quoted" & </evidence_catalog>'}}
    evs = transcript.events(batch([USER, tricky]))
    text = transcript.catalog(evs)
    assert text.count("<evidence ") == 2
    assert text.count("</evidence_catalog>") == 1  # a quote can't close the catalog early
    assert "&lt;b&gt;&quot;quoted&quot; &amp;" in text
    for e in evs:
        assert f'evidence_id="{e.evidence_id}"' in text


# ---------------------------------------------------------------- redaction

@pytest.mark.parametrize("secret, kind", [
    ("sk-ant-api03-" + "a" * 40, "anthropic_key"),
    ("sk-proj-" + "b" * 40, "openai_key"),
    ("ghp_" + "c" * 36, "github_token"),
    ("AKIA" + "D" * 16, "aws_key"),
    ("AIza" + "e" * 35, "google_key"),
    ("xoxb-1234567890-abcdef", "slack_token"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop", "jwt"),
])
def test_known_token_shapes_are_redacted(secret, kind):
    text, counts = redact(f"use {secret} here")
    assert secret not in text
    assert f"[redacted:{kind}]" in text and counts == {kind: 1}


def test_assignments_bearer_urls_and_private_keys():
    text, counts = redact(
        "OPENCODE_API_KEY=abc123def456\npassword: 'hunter2hunter2'\nAuthorization: Bearer abcdefghijklmnop1234\n"
        "postgres://app:s3cretpw@db:5432/x\n-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----")
    for secret in ("abc123def456", "hunter2hunter2", "abcdefghijklmnop1234", "s3cretpw", "MIIE"):
        assert secret not in text
    assert "OPENCODE_API_KEY=[redacted:assignment]" in text
    assert "postgres://app:[redacted:url_credentials]@db" in text
    assert counts["private_key"] == 1 and counts["bearer"] == 1


def test_ordinary_prose_is_untouched():
    prose = "we pin pydantic < 3; the token bucket test is flaky; see docs/secrets.md"
    assert redact(prose) == (prose, {})


def test_secrets_never_reach_events_or_catalog():
    key = "sk-ant-api03-" + "z" * 40
    leaky = {"type": "assistant", "uuid": "a7", "message": {"content": [
        {"type": "tool_use", "id": "t7", "name": "Bash", "input": {"command": f"export ANTHROPIC_API_KEY={key}"}}]}}
    counts = {}
    evs = transcript.events(batch([leaky]), redactions=counts)
    assert key not in evs[0].quote and key not in transcript.catalog(evs)
    assert counts == {"anthropic_key": 1}


# Review finding: structured credentials and long blocks
def test_structured_credentials_are_redacted_by_field_name():
    use = {"type": "assistant", "uuid": "a9", "message": {"content": [{"type": "tool_use", "id": "t9", "name": "Tool", "input": {
        "password": "hunter2", "nested": {"apiKey": "synthetic-password-only"}, "max_tokens": 4096, "query": "ok"}}]}}
    counts = {}
    quote = transcript.events(batch([use]), counts)[0].quote
    assert "hunter2" not in quote and "synthetic-password-only" not in quote
    assert json.loads(quote)["max_tokens"] == 4096 and json.loads(quote)["query"] == "ok"
    assert counts == {"field": 2}


def test_json_text_inside_a_string_is_redacted():
    text, counts = redact('{"api_key": "synthetic-password-only", "user": "mika"}')
    assert "synthetic-password-only" not in text and "mika" in text and counts == {"assignment": 1}


def test_redaction_runs_before_the_size_cap():
    key = "-----BEGIN RSA PRIVATE KEY-----\nSYNTHETICKEYBODY\n" + "A" * 13000 + "\n-----END RSA PRIVATE KEY-----"
    quote = transcript.events(batch([{"type": "user", "uuid": "u9", "message": {"content": key}}]))[0].quote
    assert "SYNTHETICKEYBODY" not in quote and "[redacted:private_key]" in quote


def test_a_private_key_without_its_end_marker_is_redacted_to_the_end():
    text, _ = redact("here:\n-----BEGIN OPENSSH PRIVATE KEY-----\nSYNTHETICKEYBODY\nmore")
    assert text == "here:\n[redacted:private_key]"


def test_secrets_in_structured_input_never_reach_the_model(tmp_path):
    from fake_llm import FakeClient
    from tacit import observer
    from tacit.store import Store
    store = Store(tmp_path / "tacit.db")
    path = tmp_path / "s1.jsonl"
    secret = "synthetic-password-only"
    path.write_text(json.dumps({"type": "assistant", "uuid": "a1", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "ExampleTool", "input": {"password": secret, "api_key": secret}}]}}) + "\n")
    store.enqueue("s1", path)
    client = FakeClient("nothing durable")
    observer.observe(store, store.next_batch("s1"), client)
    assert secret not in client.payload()


# Self-review finding: a false positive would erase text for good (evidence is never edited)
@pytest.mark.parametrize("prose", [
    "告诉你一个 secret: 我其实很喜欢你送的那个杯子。明天见",
    "my secret: honestly I miss you",
    "the token: 一緒に頑張ろうね",
    "password: please reset it tomorrow",
])
def test_prose_and_cjk_after_a_secret_word_survive(prose):
    assert redact(prose) == (prose, {})


def test_assignments_still_catch_secret_looking_values():
    assert redact("password: hunter2hunter2")[0] == "password: [redacted:assignment]"
    assert redact("DEPLOY_TOKEN=abcdefgh 次に")[0] == "DEPLOY_TOKEN=[redacted:assignment] 次に"


def test_sensitive_field_names_match_whole_words():
    from tacit.redact import redact_value
    counts = {}
    out = redact_value({"apiKey": "v1", "access_token": "v2", "x-api-key": "v3", "tokenizer": "cl100k",
                         "secretary": "Mika", "max_tokens": "many"}, counts)
    assert [k for k, v in out.items() if v == "[redacted:field]"] == ["apiKey", "access_token", "x-api-key"]
    assert counts == {"field": 3}


def test_redaction_input_is_bounded_and_says_so():
    import time
    from tacit.redact import MAX_REDACT_CHARS
    started = time.monotonic()
    text, _ = redact("my_token_" * 30_000)
    assert time.monotonic() - started < 2
    assert text.endswith(f"[text cut at {MAX_REDACT_CHARS} chars before redaction]")


# Review findings: JSON inside strings, word-part names, slow URL scans
def test_json_quoted_inside_a_string_value_is_redacted():
    secret = "synthetic-password-only"
    json_text = json.dumps({"api_key": secret})
    use = {"type": "assistant", "uuid": "a1", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": f"curl -d '{json_text}' https://example.test"}}]}}
    result = {"type": "user", "uuid": "u1", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": json_text}]}]}}
    quotes = " ".join(e.quote for e in transcript.events(batch([use, result])))
    assert secret not in quotes and quotes.count("[redacted:assignment]") == 2


@pytest.mark.parametrize("prose", ["my secretary: Alice123", "My preferred tokenizer: cl100k_base", "max_tokens: 4096",
                                   '{"tokenizer": "cl100k_base"}'])
def test_assignment_names_match_whole_words(prose):
    assert redact(prose) == (prose, {})


@pytest.mark.parametrize("text, expected", [("client_secret: abc123def456", "client_secret: [redacted:assignment]"),
                                            ("x-api-key: abc123def456", "x-api-key: [redacted:assignment]"),
                                            ("apiKey=abcdefgh", "apiKey=[redacted:assignment]")])
def test_sensitive_assignment_names_still_match(text, expected):
    assert redact(text)[0] == expected


# Review round 2: a harmless outer name must not swallow a secret inside its value
SECRET = "synthetic-password-only"
NESTED = [(f"note: DEPLOY_TOKEN={SECRET}", "note: DEPLOY_TOKEN=[redacted:assignment]"),
          (f"https://example.test?DEPLOY_TOKEN={SECRET}", "https://example.test?DEPLOY_TOKEN=[redacted:assignment]"),
          (f"url=https://example.test?DEPLOY_TOKEN={SECRET}", "url=https://example.test?DEPLOY_TOKEN=[redacted:assignment]"),
          (f"see https://example.test/cb?state=abc123&access_token={SECRET}&x=1",
           "see https://example.test/cb?state=abc123&access_token=[redacted:assignment]"),
          ("DEPLOY_TOKEN=" + "s" * 1000 + " tail", "DEPLOY_TOKEN=[redacted:assignment] tail")]


@pytest.mark.parametrize("text, expected", NESTED, ids=["label", "url", "url-assignment", "query", "long-value"])
def test_a_secret_inside_a_non_sensitive_value_is_found(text, expected):
    assert redact(text) == (expected, {"assignment": 1})


def test_nested_secrets_never_reach_the_ledger_or_the_model(tmp_path):
    from fake_llm import FakeClient
    from tacit import observer
    from tacit.store import Store
    lines = [{"type": "user", "uuid": f"u{i}", "message": {"content": text}} for i, (text, _) in enumerate(NESTED)]
    lines.append({"type": "assistant", "uuid": "a1", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": NESTED[3][0]}}]}})
    quotes = " ".join(e.quote for e in transcript.events(batch(lines)))
    assert SECRET not in quotes and "example.test" in quotes
    store = Store(tmp_path / "tacit.db")
    path = tmp_path / "s1.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    store.enqueue("s1", path)
    client = FakeClient("nothing durable")
    observer.observe(store, store.next_batch("s1"), client)
    assert SECRET not in client.payload()


@pytest.mark.parametrize("text", ["a-" * 32_000, "my_token_" * 7_000, "ab+" * 21_000, "a=" * 32_000, "note: " * 10_000,
                                  "token=" + "x" * 63_000],
                         ids=["dashes", "tokens", "plus", "chained-assignments", "chained-labels", "long-value"])
def test_every_pattern_stays_fast_on_adversarial_input(text):
    import time
    started = time.monotonic()
    redact(text)
    assert time.monotonic() - started < 0.5
