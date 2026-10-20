"""Redaction: scrub secrets from transcript text before it is stored or sent to any model.

Runs once, when transcript events are built, so the ledger, the observer prompt and every whisper see the same
scrubbed text. It runs on the whole block, before the block is cut to size, and structured values are redacted leaf by
leaf before they are serialized, so JSON quoting can't hide a secret. It is pattern-based (known token shapes,
`key = value` assignments, credentials in URLs, private keys, sensitive field names), so it lowers the risk but can't
promise every possible secret is caught. A redacted span becomes `[redacted:<kind>]` so a reader still sees that
something was there.

Evidence is never edited, so a false positive loses text for good: sensitive names match on whole words, and secret
values are ASCII-only, so prose and CJK text are never swallowed.
"""
import re

# Longest text the patterns run on. Longer text is cut here first, so a pathological single line can't make the
# regexes slow; evidence is capped far below this anyway, and a private key cut off here still redacts to the end.
MAX_REDACT_CHARS = 64_000

# Names whose values are secrets: whole words of a key or field name (`DEPLOY_TOKEN`, `apiKey`, `x-api-key`,
# `client_secret`), never parts of a word (`secretary`, `tokenizer`, `max_tokens`).
SENSITIVE_WORDS = {"password", "passwd", "passphrase", "secret", "token", "credential", "credentials", "authorization",
                   "apikey", "privatekey", "accesskey"}


def sensitive_name(name):
    words = [w for w in re.split(r"[^a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()) if w]
    return any(w in SENSITIVE_WORDS or w + nxt in SENSITIVE_WORDS for w, nxt in zip(words, words[1:] + [""]))


# (kind, pattern): known token shapes, in order. Every variable-length part is bounded, so no pattern can backtrack
# quadratically.
PATTERNS = [
    # a BEGIN marker with no END redacts to the end of the text: the rest may be key material
    ("private_key", re.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----(?:.*?-----END [A-Z ]{0,40}PRIVATE KEY-----|.*\Z)",
                               re.S)),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("bearer", re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._\-+/=]{16,}")),
    ("url_credentials", re.compile(r"(?i)(\b[a-z][a-z0-9+.\-]{0,31}://[^\s:/@]{1,256}:)[^\s@/]{1,256}(@)")),
]

# A secret value: printable ASCII with no spaces, quotes or separators.
VALUE = r"[\x21\x23-\x26\x28-\x2b\x2d-\x3a\x3c-\x7e]"
# `name = value` / `name: value` / JSON `"name": "value"`. The name is captured whole and checked with sensitive_name.
# `=` (env style) takes any value of 6+ characters; `:` only one that looks like a secret (a digit, or 16+ long), so
# "my secret: honestly…" stays prose. The value is matched at most 256 characters at a time so the scan stays linear;
# VALUE_REST takes the rest of a sensitive one.
ASSIGNMENT = re.compile(rf"(?<![A-Za-z0-9_.\-])([A-Za-z][A-Za-z0-9_.\-]{{0,63}})([\"']?\s*([=:])\s*[\"']?)"
                        rf"(?!\[redacted)({VALUE}{{6,256}})")
VALUE_REST = re.compile(rf"{VALUE}*")


def redact(text):
    """Return (scrubbed_text, {kind: count})."""
    counts = {}
    if len(text) > MAX_REDACT_CHARS:
        text = f"{text[:MAX_REDACT_CHARS]}\n... [text cut at {MAX_REDACT_CHARS} chars before redaction]"
    for kind, pattern in PATTERNS:
        def replace(match, kind=kind):
            counts[kind] = counts.get(kind, 0) + 1
            # Keep a captured prefix ("Bearer ", "scheme://user:") and suffix ("@") so the text stays readable
            groups = match.groups()
            prefix = groups[0] if groups else ""
            suffix = groups[1] if len(groups) > 1 else ""
            return f"{prefix}[redacted:{kind}]{suffix}"
        text = pattern.sub(replace, text)

    out, kept, pos = [], 0, 0
    while match := ASSIGNMENT.search(text, pos):
        name, sep, op, value = match.groups()
        value_start = match.start(4)
        if not sensitive_name(name) or (op == ":" and not re.search(r"[0-9]", value) and len(value) < 16):
            # Not a secret itself, but its value may hold one (`note: DEPLOY_TOKEN=…`, `https://…?token=…`): scan on
            # inside it
            pos = value_start
            continue
        counts["assignment"] = counts.get("assignment", 0) + 1
        out.append(f"{text[kept:value_start]}[redacted:assignment]")
        kept = pos = VALUE_REST.match(text, match.end()).end()
    return "".join(out) + text[kept:], counts


def redact_value(value, counts):
    """A copy of a JSON value with every string leaf redacted, and the string values of sensitive fields replaced.
    Counts go to `counts`."""
    if isinstance(value, str):
        text, found = redact(value)
        for kind, n in found.items():
            counts[kind] = counts.get(kind, 0) + n
        return text
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, str) and sensitive_name(key) and isinstance(item, str) and item:
                counts["field"] = counts.get("field", 0) + 1
                out[key] = "[redacted:field]"
            else:
                out[key] = redact_value(item, counts)
        return out
    if isinstance(value, list):
        return [redact_value(item, counts) for item in value]
    return value
